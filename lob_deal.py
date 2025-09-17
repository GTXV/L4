# lob_pipeline.py
from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import polars as pl

# ========= 配置 =========
TIME_UNIT = "ms"                 # 统一时间单位
DEFAULT_FREQ = "1s"              # 统一栅格
MAX_STALENESS_MS = 3000          # 小缺口允许的最大“陈旧”时长
MIN_SEG_GAP_MS = 5000            # 段切分阈值：超过视为大缺口 -> 新 segment
MANIFEST_PATH = "manifest.parquet"
GRID_CACHE_DIR = "grid_parquet"  # 栅格化后缓存输出目录

# 典型 L2 列；按你实际列名微调
ASK_PX = [f"ap{i}" for i in range(1, 21)]
ASK_VOL = [f"av{i}" for i in range(1, 21)]
BID_PX = [f"bp{i}" for i in range(1, 21)]
BID_VOL = [f"bv{i}" for i in range(1, 21)]
LOB_COLS = ASK_PX + ASK_VOL + BID_PX + BID_VOL
TS_COL = "T"   # 原始毫秒时间戳

# ========= 工具 =========
def parse_symbol_and_date(p: Path) -> Tuple[str, str]:
    # 目录名就是 YYYY-MM-DD
    date_str = p.parent.name
    # 文件名形如 "AAVE-2025-05-06-..." -> 取首段为 symbol
    m = re.match(r"([A-Za-z0-9_]+)-", p.name)
    sym = m.group(1) if m else p.stem.split("-")[0]
    return sym, date_str

def read_one(p: Path) -> pl.DataFrame:
    # 自行按实际格式改：示例兼容 csv/parquet/feather
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        df = pl.read_parquet(p)
    elif suffix in (".feather", ".ft"):
        df = pl.read_ipc(p)
    else:
        # 假定 csv，含表头
        df = pl.read_csv(p, infer_schema_length=1000)
    if TS_COL not in df.columns:
        # 兜底兼容：尝试 'ts'/'timestamp' 列
        for alt in ["ts", "timestamp"]:
            if alt in df.columns:
                df = df.rename({alt: TS_COL})
                break
    return df

def basic_clean(df: pl.DataFrame) -> pl.DataFrame:
    # 仅保留 LOB 列与 T
    keep = [c for c in df.columns if c in LOB_COLS] + [TS_COL]
    df = df.select(keep).drop_nulls(subset=[TS_COL])
    # 确保类型正确
    df = df.with_columns(pl.col(TS_COL).cast(pl.Int64))
    # 排序&按时间去重（同 T 取最后一条）
    df = df.sort(TS_COL).unique(subset=[TS_COL], keep="last")
    # 加 datetime 列
    df = df.with_columns(
        pl.from_epoch(pl.col(TS_COL), unit=TIME_UNIT).alias("ts")
    )
    return df

def quality_stats(df: pl.DataFrame) -> Dict:
    if df.is_empty():
        return {
            "rows": 0,
            "t_min": None,
            "t_max": None,
            "median_dt_ms": None,
            "crossed_ratio": None,
            "mean_spread_bps": None,
        }
    # 顶档
    has_best = ("ap1" in df.columns) and ("bp1" in df.columns)
    crossed = pl.lit(None)
    spread_bps = pl.lit(None)
    if has_best:
        crossed = (pl.col("bp1") >= pl.col("ap1")).mean()  # 交叉比例
        spread_bps = ( (pl.col("ap1") - pl.col("bp1")) / pl.col("bp1").abs() * 1e4 ).mean()

    dt = pl.col(TS_COL).diff().alias("dt")
    q = df.select(
        pl.len().alias("rows"),
        pl.col(TS_COL).min().alias("t_min"),
        pl.col(TS_COL).max().alias("t_max"),
        dt
    )
    med_dt = q["dt"].drop_nulls().median() if q["rows"][0] > 1 else None

    return {
        "rows": int(q["rows"][0]),
        "t_min": int(q["t_min"][0]) if q["rows"][0] > 0 else None,
        "t_max": int(q["t_max"][0]) if q["rows"][0] > 0 else None,
        "median_dt_ms": int(med_dt) if med_dt is not None else None,
        "crossed_ratio": float(df.select(crossed)["bp1"].item()) if has_best else None,
        "mean_spread_bps": float(df.select(spread_bps)["ap1"].item()) if has_best else None,
    }

def resample_to_grid(
    df: pl.DataFrame,
    every: str = DEFAULT_FREQ,
    max_staleness_ms: int = MAX_STALENESS_MS,
    min_seg_gap_ms: int = MIN_SEG_GAP_MS,
) -> pl.DataFrame:
    if df.is_empty():
        return df

    t0 = df["ts"].min()
    t1 = df["ts"].max()
    grid = pl.DataFrame({
        "ts": pl.datetime_range(start=t0, end=t1, interval=every, time_unit=TIME_UNIT, eager=True)
    })
    # 为 asof 准备：右表的匹配键改名为 last_ts
    right = df.rename({"ts": "last_ts"})

    joined = grid.join_asof(
        right,
        left_on="ts", right_on="last_ts",
        strategy="backward"
    )

    # 陈旧度（可能为 null：表示此前没有任何观测）
    staleness = (pl.col("ts").cast(pl.Int64) - pl.col("last_ts").cast(pl.Int64)).alias("staleness_ms")
    joined = joined.with_columns(staleness)

    # 掩码：小缺口以内视为 live
    joined = joined.with_columns(
        (pl.col("staleness_ms").is_not_null() & (pl.col("staleness_ms") <= max_staleness_ms)).alias("mask_live")
    )

    # 段切分：超过 min_seg_gap_ms 视为断档
    gap_flag = (pl.col("staleness_ms").is_null() | (pl.col("staleness_ms") > min_seg_gap_ms))
    joined = joined.with_columns(
        gap_flag.fill_null(True).cast(pl.Int8).diff().fill_null(0).ne(0).cumsum().alias("segment_id")
    )

    # 常用派生：mid / spread
    if "ap1" in joined.columns and "bp1" in joined.columns:
        joined = joined.with_columns([
            ((pl.col("ap1") + pl.col("bp1")) * 0.5).alias("mid"),
            (pl.col("ap1") - pl.col("bp1")).alias("spread")
        ])

    return joined

def grid_quality(joined: pl.DataFrame) -> Dict:
    if joined.is_empty():
        return {"grid_rows": 0, "coverage_ratio": 0.0}
    n = joined.height
    cov = float(joined.select(pl.col("mask_live").fill_null(False).mean()).item())
    return {"grid_rows": n, "coverage_ratio": cov}

# ========= 主流程 =========
def build_manifest(root_dir: str, every: str = DEFAULT_FREQ) -> pl.DataFrame:
    root = Path(root_dir)
    rows = []
    for date_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        date_str = date_dir.name
        for f in sorted(date_dir.iterdir()):
            if f.is_dir():
                continue
            sym, _ = parse_symbol_and_date(f)
            try:
                raw = read_one(f)
                raw = basic_clean(raw)
                q = quality_stats(raw)
                grd = resample_to_grid(raw, every=every)
                gq = grid_quality(grd)
                rows.append({
                    "date": date_str,
                    "symbol": sym,
                    "path": str(f),
                    **q, **gq
                })
            except Exception as e:
                rows.append({
                    "date": date_str, "symbol": sym, "path": str(f),
                    "rows": 0, "t_min": None, "t_max": None, "median_dt_ms": None,
                    "crossed_ratio": None, "mean_spread_bps": None,
                    "grid_rows": 0, "coverage_ratio": 0.0,
                    "error": str(e)
                })
    mf = pl.DataFrame(rows)
    mf.write_parquet(MANIFEST_PATH)
    return mf

def materialize_grid_parquet(
    root_dir: str,
    out_dir: str = GRID_CACHE_DIR,
    every: str = DEFAULT_FREQ,
    max_staleness_ms: int = MAX_STALENESS_MS,
    min_seg_gap_ms: int = MIN_SEG_GAP_MS,
    overwrite: bool = False
):
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for date_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        date_str = date_dir.name
        (out / date_str).mkdir(exist_ok=True)
        for f in sorted(date_dir.iterdir()):
            if f.is_dir():
                continue
            sym, _ = parse_symbol_and_date(f)
            out_path = out / date_str / f"{sym}.parquet"
            if out_path.exists() and not overwrite:
                continue
            raw = basic_clean(read_one(f))
            joined = resample_to_grid(raw, every=every,
                                      max_staleness_ms=max_staleness_ms,
                                      min_seg_gap_ms=min_seg_gap_ms)
            # 附带 symbol，方便后续 concat
            joined = joined.with_columns(pl.lit(sym).alias("symbol"))
            joined.write_parquet(out_path)

def select_tradable_universe(
    manifest_path: str = MANIFEST_PATH,
    min_coverage: float = 0.90,
    min_rows: int = 3600  # 对应 1Hz 栅格下 ≥1小时
) -> pl.DataFrame:
    mf = pl.read_parquet(manifest_path)
    return (
        mf.filter(
            (pl.col("coverage_ratio") >= min_coverage) &
            (pl.col("grid_rows") >= min_rows)
        )
        .select(["date", "symbol", "coverage_ratio", "grid_rows", "mean_spread_bps", "crossed_ratio"])
        .sort(["date", "symbol"])
    )

def load_panel(
    cache_dir: str,
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None
) -> pl.DataFrame:
    root = Path(cache_dir)
    dates = [p.name for p in sorted(root.iterdir()) if p.is_dir() and (start_date <= p.name <= end_date)]
    dfs = []
    for d in dates:
        day_dir = root / d
        files = list(day_dir.glob("*.parquet"))
        for fp in files:
            sym = fp.stem
            if (symbols is not None) and (sym not in symbols):
                continue
            df = pl.read_parquet(fp)
            dfs.append(df.select(["ts","symbol","mid","spread","mask_live","staleness_ms","segment_id"] + [c for c in LOB_COLS if c in df.columns]))
    return pl.concat(dfs, how="diagonal_relaxed") if dfs else pl.DataFrame()

# ========= 示例：横截面因子（顶档不平衡） =========
def top1_imbalance(df: pl.DataFrame) -> pl.DataFrame:
    # 仅在 mask_live 的时刻计算
    expr = None
    if all(c in df.columns for c in ("av1","bv1")):
        expr = ( (pl.col("bv1") - pl.col("av1")) / (pl.col("bv1") + pl.col("av1")).clip_min(1e-12) ).alias("imb1")
    else:
        return df
    return df.with_columns(expr).with_columns(
        pl.when(pl.col("mask_live")).then(pl.col("imb1")).otherwise(None).alias("imb1")
    )

if __name__ == "__main__":
    ROOT = "/path/to/your/LOB_root"   # 改成你的根目录
    # 1) 构建清单
    mf = build_manifest(ROOT, every=DEFAULT_FREQ)
    print(mf.head())

    # 2) 栅格化并缓存
    materialize_grid_parquet(ROOT, out_dir=GRID_CACHE_DIR, every=DEFAULT_FREQ)

    # 3) 选 tradable 宇宙
    tradable = select_tradable_universe(MANIFEST_PATH, min_coverage=0.9, min_rows=12*3600)
    print(tradable.head())

    # 4) 载入一段时间的面板并计算示例因子
    panel = load_panel(GRID_CACHE_DIR, "2025-05-01", "2025-05-07", symbols=None)
    panel = top1_imbalance(panel)
    print(panel.head())
