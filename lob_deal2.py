# lob_trades_integration.py
from __future__ import annotations
from pathlib import Path
import re
import polars as pl

# ========== 全局配置（与 LOB 保持一致） ==========
TIME_UNIT = "ms"
DEFAULT_FREQ = "1s"
TRADES_GRID_DIR = "trades_grid_parquet"   # 输出：trades按日栅格
MERGED_GRID_DIR = "merged_grid_parquet"   # 输出：LOB+Trades合并后
# LOB 缓存目录（来自上一版管线 materialize_grid_parquet 的输出）
LOB_GRID_DIR = "grid_parquet"

# Binance aggTrades 列（你给过）
TRADE_COLS = ["agg_trade_id","price","quantity","first_trade_id","last_trade_id","transact_time","is_buyer_maker"]

# ---------- 读与清洗 ----------
def read_month_file(p: Path) -> pl.DataFrame:
    # feather -> polars ipc
    df = pl.read_ipc(p)
    # 统一列名（防止大小写）
    ren = {c: c.lower() for c in df.columns}
    df = df.rename(ren)
    # 只保留需要的列
    keep = [c for c in TRADE_COLS if c in df.columns]
    df = df.select(keep)
    # 类型修正
    df = df.with_columns([
        pl.col("transact_time").cast(pl.Int64),
        pl.col("price").cast(pl.Float64),
        pl.col("quantity").cast(pl.Float64),
        pl.col("is_buyer_maker").cast(pl.Boolean)
    ])
    # 时间列
    df = df.with_columns(
        pl.from_epoch(pl.col("transact_time"), unit=TIME_UNIT).alias("ts")
    ).sort("ts")
    return df

def parse_symbol_from_filename(p: Path) -> str:
    # e.g. BTCUSDT-aggTrades-2025-05.feather -> BTCUSDT
    m = re.match(r"([A-Z0-9_]+)-aggTrades-", p.name, flags=re.I)
    return m.group(1).upper() if m else p.stem.split("-")[0].upper()

# ---------- 月文件 -> 按日栅格 ----------
def month_to_daily_trade_bins(month_file: str, out_dir: str = TRADES_GRID_DIR, every: str = DEFAULT_FREQ):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    p = Path(month_file)
    sym = parse_symbol_from_filename(p)
    df = read_month_file(p)

    if df.is_empty():
        return

    # 分日切片
    df = df.with_columns(pl.col("ts").dt.date().alias("d"))
    for day, sub in df.group_by("d", maintain_order=True):
        date_str = str(day)
        day_dir = out_root / date_str
        day_dir.mkdir(exist_ok=True)
        out_path = day_dir / f"{sym}.parquet"
        # 按秒分箱（右闭右标记，落到bin右端）
        bins = (
            sub.group_by_dynamic(index_column="ts", every=every, label="right", closed="right")
               .agg([
                   pl.len().alias("n_trades"),
                   pl.sum("quantity").alias("vol"),
                   pl.sum( pl.when(~pl.col("is_buyer_maker")).then(pl.col("quantity")).otherwise(0.0) ).alias("vol_buy"),
                   pl.sum( pl.when(pl.col("is_buyer_maker")).then(pl.col("quantity")).otherwise(0.0) ).alias("vol_sell"),
                   pl.sum(pl.col("price") * pl.col("quantity")).alias("dollar_vol"),
                   (pl.sum(pl.col("price") * pl.col("quantity")) / pl.sum("quantity")).alias("vwap"),
                   pl.col("price").tail(1).alias("last_price")
               ])
               .with_columns([
                   pl.col("vol").fill_null(0.0),
                   (pl.col("vol_buy") - pl.col("vol_sell")).alias("vol_net"),
                   (pl.col("vol_buy") / (pl.col("vol_buy") + pl.col("vol_sell")).clip_min(1e-12)).alias("of_buy_ratio"),
                   (pl.col("dollar_vol").fill_null(0.0)).alias("signed_dv")  # 注意：这里是总美元量；若要“带方向”的美元流见下行
               ])
        )

        # 方向美元流（买主动为正，卖主动为负）
        # is_buyer_maker==False => buy-agg
        signed_dv = (
            sub.with_columns( 
                (pl.when(~pl.col("is_buyer_maker")).then(1).otherwise(-1) * pl.col("price") * pl.col("quantity")).alias("sdv")
            )
            .group_by_dynamic(index_column="ts", every=every, label="right", closed="right")
            .agg(pl.sum("sdv").alias("signed_dv"))
        )

        bins = bins.join(signed_dv, on="ts", how="left").with_columns(pl.col("signed_dv").fill_null(0.0))

        # 计算 bin 收益（基于 vwap 或 last_price）
        bins = bins.with_columns(
            pl.col("last_price").pct_change().alias("ret_last")
        )

        # 加 symbol & 掩码
        bins = bins.with_columns([
            pl.lit(sym).alias("symbol"),
            (pl.col("n_trades") > 0).alias("mask_trade")
        ])

        # 补齐整天的空bin：让每秒都有一行（便于与 LOB 对齐）
        t0 = pl.datetime(day.year, day.month, day.day, 0, 0, 0, time_unit=TIME_UNIT)
        t1 = pl.datetime(day.year, day.month, day.day, 23, 59, 59, time_unit=TIME_UNIT)
        full_grid = pl.DataFrame({"ts": pl.datetime_range(t0, t1, every=every, eager=True, time_unit=TIME_UNIT)})
        bins = full_grid.join(bins, on="ts", how="left").with_columns([
            pl.col("symbol").fill_null(sym),
            pl.col("n_trades").fill_null(0),
            pl.col("vol").fill_null(0.0),
            pl.col("vol_buy").fill_null(0.0),
            pl.col("vol_sell").fill_null(0.0),
            pl.col("vol_net").fill_null(0.0),
            pl.col("dollar_vol").fill_null(0.0),
            pl.col("vwap"),
            pl.col("last_price"),
            pl.col("ret_last").fill_null(0.0),
            pl.col("of_buy_ratio").fill_null(0.5),
            pl.col("signed_dv").fill_null(0.0),
            pl.col("mask_trade").fill_null(False)
        ])

        bins.write_parquet(out_path)

# ---------- 批量把 trades 月文件物化为“按日” ----------
def materialize_trades_month_dir(trades_dir: str, out_dir: str = TRADES_GRID_DIR, every: str = DEFAULT_FREQ):
    td = Path(trades_dir)
    files = sorted([p for p in td.iterdir() if p.suffix.lower() in (".feather", ".ft", ".ipc", ".parquet")])
    for f in files:
        month_to_daily_trade_bins(str(f), out_dir=out_dir, every=every)

# ---------- 合并 LOB 栅格 与 日级 trades 栅格 ----------
def merge_lob_and_trades_per_day(date_str: str, symbols: list[str] | None = None,
                                 lob_dir: str = LOB_GRID_DIR, trades_dir: str = TRADES_GRID_DIR,
                                 out_dir: str = MERGED_GRID_DIR):
    out_day = Path(out_dir) / date_str
    out_day.mkdir(parents=True, exist_ok=True)

    lob_day = Path(lob_dir) / date_str
    tr_day = Path(trades_dir) / date_str
    if not lob_day.exists():
        return

    # 遍历该日的 LOB 符号（以 LOB 为主，只有 trades 没有 LOB 的不合并）
    for fp in sorted(lob_day.glob("*.parquet")):
        sym = fp.stem
        if symbols and sym not in symbols:
            continue
        lob = pl.read_parquet(fp)  # 含 ts,symbol,mid,spread,mask_live,staleness_ms,segment_id,...
        # 尝试加载 trades 栅格
        tfile = tr_day / f"{sym}.parquet"
        if tfile.exists():
            tb = pl.read_parquet(tfile)  # 含 ts,symbol,n_trades,vol,of_buy_ratio,signed_dv,...
        else:
            # 构造空的占位表（以便左连接后列存在）
            tb = lob.select("ts").with_columns([
                pl.lit(sym).alias("symbol"),
                pl.lit(0).alias("n_trades"),
                pl.lit(0.0).alias("vol"),
                pl.lit(0.0).alias("vol_buy"),
                pl.lit(0.0).alias("vol_sell"),
                pl.lit(0.0).alias("vol_net"),
                pl.lit(0.0).alias("dollar_vol"),
                pl.lit(None).alias("vwap"),
                pl.lit(None).alias("last_price"),
                pl.lit(0.0).alias("ret_last"),
                pl.lit(0.5).alias("of_buy_ratio"),
                pl.lit(0.0).alias("signed_dv"),
                pl.lit(False).alias("mask_trade"),
            ])

        merged = lob.join(tb.drop("symbol"), on="ts", how="left")

        # 一些常见派生（仅在 mask_live 内有效）
        merged = merged.with_columns([
            pl.when(pl.col("mask_live")).then( ((pl.col("ap1")+pl.col("bp1"))*0.5) ).otherwise(pl.col("mid")).alias("mid"),  # 保底
            pl.when(pl.col("mask_live")).then(pl.col("spread")).otherwise(None).alias("spread")
        ])

        # 保存
        (out_day / f"{sym}.parquet").write_bytes(merged.write_parquet(file=None))

# ---------- 质量统计：trades 覆盖 ----------
def trade_day_coverage(date_str: str, trades_dir: str = TRADES_GRID_DIR) -> pl.DataFrame:
    day_dir = Path(trades_dir) / date_str
    rows = []
    if not day_dir.exists():
        return pl.DataFrame()
    for fp in sorted(day_dir.glob("*.parquet")):
        sym = fp.stem
        df = pl.read_parquet(fp).select(["ts","n_trades","mask_trade"])
        cov = float(df.select(pl.col("mask_trade").mean()).item())
        med_dt = (
            df.filter(pl.col("mask_trade"))
              .select(pl.col("ts").cast(pl.Int64).diff().drop_nulls())
              .to_series().median()
        )
        rows.append({"date": date_str, "symbol": sym, "trade_coverage": cov,
                     "median_trade_dt_ms": int(med_dt) if med_dt is not None else None})
    return pl.DataFrame(rows)

if __name__ == "__main__":
    # 1) 先把月度 aggTrades 物化为“按日栅格”
    materialize_trades_month_dir(trades_dir="trades", out_dir=TRADES_GRID_DIR, every=DEFAULT_FREQ)

    # 2) 把某天 LOB 与 trades 合并为统一面板
    merge_lob_and_trades_per_day("2025-05-06", lob_dir=LOB_GRID_DIR, trades_dir=TRADES_GRID_DIR, out_dir=MERGED_GRID_DIR)

    # 3) 看看该日 trades 覆盖度
    print(trade_day_coverage("2025-05-06"))
