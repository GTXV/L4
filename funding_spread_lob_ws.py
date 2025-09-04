#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, asyncio, time, json, logging, argparse, itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Callable
import requests
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    from pytz import timezone as ZoneInfo

try:
    import orjson
    def jdump(obj): return orjson.dumps(obj).decode()
except Exception:
    def jdump(obj): return json.dumps(obj, separators=(',', ':'), ensure_ascii=False)
from pprint import pprint


LOG = logging.getLogger("funding_spread_lob")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
LOG.addHandler(_handler)

SG_TZ = ZoneInfo("Asia/Singapore")

# ---------------- Symbol normalization ----------------
def to_canon(symbol: str, exchange: str) -> str:
    s = symbol.upper().strip()
    ex = exchange.lower()
    if ex == "okx":
        s = s.replace("-USDT-SWAP", "USDT").replace("-", "")
    elif ex == "bitget":
        s = s.replace("_UMCBL", "")
    elif ex == "gate":
        s = s.replace("_", "")
    return s

def canon_to_gate_contract(canon: str) -> str:
    return f"{canon[:-4]}_USDT"

def compute_next_scan_time(now_utc: datetime) -> datetime:
    now_sg = now_utc.astimezone(SG_TZ)
    target_sg = now_sg.replace(minute=59, second=0, microsecond=0)
    window_end = target_sg + timedelta(minutes=1)
    if target_sg <= now_sg < window_end:
        return now_utc
    if now_sg < target_sg:
        next_target_sg = target_sg
    else:
        next_target_sg = (target_sg + timedelta(hours=1))
    return next_target_sg.astimezone(timezone.utc)

# ---------------- Funding fetchers ----------------
""" def fetch_okx_funding_all(timeout=20) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        url = "https://www.okx.com/api/v5/public/funding-rate"
        params = {"instType": "SWAP"}
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "0":
            raise RuntimeError(f"Error fetching OKX funding rates: {data.get('msg')}")
        arr = data.get("data", [])
        if not arr:
            raise RuntimeError("No data found in OKX funding rates response")
        for d0 in arr:
            instId = d0.get("instId", "")
            if not instId.endswith("-USDT-SWAP"): continue
            fr = float(d0.get("fundingRate"))
            nxt = int(d0.get("nextFundingTime"))
            canon = to_canon(instId, "okx")
            out[canon] = {"exchange": "okx", "raw_symbol": instId, "rate": fr, "next_ts": nxt}
    except Exception as e:
        LOG.debug("OKX funding fetch error: %s", e)
    return out """

def fetch_okx_funding_all(timeout=20) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    try:
        url = "https://www.okx.com/api/v5/public/funding-rate"
        # ✅ 关键：一次性获取全部 SWAP 的资金费
        params = {"instId": "ANY"}   # 不要再传 instType
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "0":
            raise RuntimeError(f"Error fetching OKX funding rates: {data.get('msg')}")
        arr = data.get("data", [])
        if not arr:
            raise RuntimeError("No data found in OKX funding rates response")
        for d0 in arr:
            instId = d0.get("instId", "")
            # OKX 会把所有 SWAP 都返回（含 -USDT- / -USD- / 反向），你如果只要 USDT，可以保留这个筛选
            if not instId.endswith("-USDT-SWAP"):
                continue
            fr  = float(d0.get("fundingRate"))
            nxt = int(d0.get("nextFundingTime"))
            canon = to_canon(instId, "okx")
            out[canon] = {
                "exchange":   "okx",
                "raw_symbol": instId,
                "rate":       fr,
                "next_ts":    nxt,
            }
    except Exception as e:
        LOG.debug("OKX funding fetch error: %s", e, exc_info=True)
    return out


def fetch_binance_funding_all(timeout=20) -> Dict[str, Dict[str, Any]]:
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    now_ms = int(time.time() * 1000)
    out = {}
    for d in data:
        try:
            sym = d["symbol"]
            if not sym.endswith("USDT"): continue
            fr_raw = d.get("lastFundingRate")
            if fr_raw in (None, ""): continue
            fr = float(fr_raw)
            nxt = int(d.get("nextFundingTime", now_ms))
            canon = to_canon(sym, "binance")
            out[canon] = {"exchange": "binance", "raw_symbol": sym, "rate": fr, "next_ts": nxt}
        except Exception as e:
            LOG.debug("Binance parse skip: %s | %s", d, e)
    return out

def fetch_bitget_funding_all(timeout=20) -> Dict[str, Dict[str, Any]]:
    url = "https://api.bitget.com/api/v2/mix/market/current-fund-rate"
    params = {"productType": "usdt-futures"}
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if data.get("code") != "00000":
        raise RuntimeError(f"Bitget error: {data.get('msg')}")
    now_ms = int(time.time() * 1000)
    out = {}
    for d in data.get("data", []):
        try:
            sym = d["symbol"]
            fr_raw = d.get("fundingRate")
            if fr_raw in (None, ""): continue
            fr = float(fr_raw)
            nxt = int(d.get("nextUpdate", now_ms))
            canon = to_canon(sym, "bitget")
            out[canon] = {"exchange": "bitget", "raw_symbol": sym, "rate": fr, "next_ts": nxt}
        except Exception as e:
            LOG.debug("Bitget parse skip: %s | %s", d, e)
    return out

def fetch_gate_funding_all(timeout=20) -> Dict[str, Dict[str, Any]]:
    url = "https://api.gateio.ws/api/v4/futures/usdt/contracts"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    out = {}
    now_s = int(time.time())
    for d in data:
        try:
            nm = d["name"]
            if not nm.endswith("_USDT"): continue
            fr_raw = d.get("funding_rate")
            if fr_raw in (None, ""): continue
            fr = float(fr_raw)
            nxt_s = int(d.get("funding_next_apply", now_s))
            nxt_ms = nxt_s * 1000
            canon = to_canon(nm, "gate")
            out[canon] = {"exchange": "gate", "raw_symbol": nm, "rate": fr, "next_ts": nxt_ms}
        except Exception as e:
            LOG.debug("Gate parse skip: %s | %s", d, e)
    return out

def fetch_all_exchanges() -> Dict[str, Dict[str, Dict[str, Any]]]:
    return {
        "okx": fetch_okx_funding_all(),
        "binance": fetch_binance_funding_all(),
        "bitget": fetch_bitget_funding_all(),
        "gate": fetch_gate_funding_all(),
    }

# ---------------- Spreads ----------------
@dataclass
class PairSpread:
    canon: str
    ex_a: str
    ex_b: str
    rate_a: float
    rate_b: float
    spread: float
    abs_spread: float
    next_ts: int
    raw_a: str
    raw_b: str
    def funding_local_str(self) -> str:
        dt = datetime.fromtimestamp(self.next_ts / 1000, SG_TZ)
        return dt.strftime("%H_%M")
    def filename(self) -> str:
        pct_abs_for_name = round(abs(self.spread) * 100, 4)
        return f"{self.canon}-{self.funding_local_str()}-{self.ex_a}-{self.ex_b}-{pct_abs_for_name}"

def compute_pair_spreads(all_data: Dict[str, Dict[str, Dict[str, Any]]]) -> List['PairSpread']:
    exchanges = list(all_data.keys())
    universe = set().union(*[set(v.keys()) for v in all_data.values()])
    pairs: List[PairSpread] = []
    for canon in universe:
        avail = [(ex, all_data[ex][canon]) for ex in exchanges if canon in all_data[ex]]
        if len(avail) < 2:
            continue
        for (ex_a, a), (ex_b, b) in itertools.combinations(avail, 2):
            rate_a, rate_b = float(a["rate"]), float(b["rate"])
            spread = rate_a - rate_b
            next_ts = min(int(a["next_ts"]), int(b["next_ts"]))
            pairs.append(PairSpread(
                canon=canon, ex_a=ex_a, ex_b=ex_b,
                rate_a=rate_a, rate_b=rate_b,
                spread=spread, abs_spread=abs(spread),
                next_ts=next_ts, raw_a=a["raw_symbol"], raw_b=b["raw_symbol"]
            ))
    return pairs

def top_n_pairs(pairs: List[PairSpread], n: int) -> List[PairSpread]:
    return sorted(pairs, key=lambda p: p.abs_spread, reverse=True)[:n]

# ---------------- LOB capture infra ----------------
@dataclass
class LOBRecord:
    ts_ms: int
    exchange: str
    symbol: str
    payload: Any

class SubscriptionHub:
    def __init__(self, use_rest_fallback: bool = False):
        self.use_rest_fallback = use_rest_fallback
        self.okx_order_books: Dict[str, Dict[str, Any]] = {}
        self.bitget_order_books: Dict[str, Dict[str, Any]] = {}
        self.gate_order_books: Dict[str, Dict[str, Any]] = {}
        self.gate_level: Dict[str, int] = {}
        self._buffers: Dict[Tuple[str, str], List[LOBRecord]] = {}
        self._tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        self._adapters: Dict[str, Callable[[str, Callable[[LOBRecord], None], bool], asyncio.Future]] = {
            "okx": self._subscribe_okx,
            "binance": self._subscribe_binance,
            "bitget": self._subscribe_bitget,
            "gate": self._subscribe_gate,
        }
    def get_buffer(self, exchange: str, raw_symbol: str) -> List[LOBRecord]:
        return self._buffers.setdefault((exchange, raw_symbol), [])
    def _append(self, rec: LOBRecord):
        self.get_buffer(rec.exchange, rec.symbol).append(rec)
    async def ensure(self, exchange: str, raw_symbol: str):
        key = (exchange, raw_symbol)
        if key in self._tasks and not self._tasks[key].done():
            return
        coro = self._adapters[exchange](raw_symbol, self._append, self.use_rest_fallback)
        task = asyncio.create_task(coro, name=f"sub:{exchange}:{raw_symbol}")
        self._tasks[key] = task
    async def stop_all(self):
        for t in self._tasks.values():
            t.cancel()
        await asyncio.sleep(0.05)

    def _handle_gate_full(
        self,
        raw_symbol: str,
        obj: Dict[str, Any],
        sink: Callable[[LOBRecord], None],
    ):
        """
        Gate.io futures.order_book（全量推送）处理：
        - 每帧都是整本 Top-N，直接规范化并下沉
        - 统一输出 payload: {'a': [[p,q],...], 'b': [[p,q],...], 'ts': int}
        """
        res = obj.get("result") or {}
        if not res:
            return

        # 时间戳
        ts_raw = res.get("t") or obj.get("time_ms") or int(time.time() * 1000)
        try:
            ts = int(ts_raw)
        except Exception:
            ts = int(time.time() * 1000)

        # 读取订阅的 level（若消息没带 l 就回退）
        try:
            top_n = int(res.get("l")) if res.get("l") is not None else None
        except Exception:
            top_n = None
        if top_n is None:
            top_n = getattr(self, "gate_level", {}).get(raw_symbol)

        # 取 asks/bids，兼容两种格式：list[dict{p,s}] 或 list[list[p,q]]
        asks_raw = res.get("asks") or res.get("a") or []
        bids_raw = res.get("bids") or res.get("b") or []

        def _norm_side(arr):
            out = []
            for it in arr:
                if isinstance(it, dict):
                    p, q = it.get("p"), it.get("s")
                else:
                    if not it: 
                        continue
                    p = it[0]; q = it[1] if len(it) > 1 else "0"
                if p is None:
                    continue
                # 跳过 0 数量（有些帧会夹杂为 0 的价位）
                try:
                    if float(q) == 0.0:
                        continue
                except Exception:
                    pass
                out.append([str(p), str(q)])
            return out

        asks = _norm_side(asks_raw)
        bids = _norm_side(bids_raw)

        # 排序并裁到固定档数
        bids = sorted(bids, key=lambda x: float(x[0]), reverse=True)
        asks = sorted(asks, key=lambda x: float(x[0]))
        if isinstance(top_n, int) and top_n > 0:
            bids = bids[:top_n]
            asks = asks[:top_n]

        payload = {"a": asks, "b": bids, "ts": ts}
        #print(payload)
        sink(LOBRecord(ts_ms=ts, exchange="gate", symbol=raw_symbol, payload=payload))

    def _handle_bitget_update(
        self,
        raw_symbol: str,
        obj: Dict[str, Any],
        sink: Callable[[LOBRecord], None],
    ):
        """
        维护本地 Bitget 订单簿（price -> qty），将 snapshot / update 合并成一份完整订单簿。
        输出 payload 统一规范为 {'a': [[p, q], ...], 'b': [[p, q], ...], 'ts': ts}，以复用现有 formatter。
        """
        book = self.bitget_order_books.setdefault(raw_symbol, {'bids': {}, 'asks': {}, 'ts': 0})

        action = obj.get('action')
        if not obj.get('data'):
            return
        data = obj['data'][0]

        if action == 'snapshot':
            # 直接重建整本
            book['bids'] = {p: q for p, q, *rest in data.get('bids', [])}
            book['asks'] = {p: q for p, q, *rest in data.get('asks', [])}
        elif action == 'update':
            # 增量合并：数量=0 删除；否则覆盖
            for p, q, *_ in data.get('bids', []):
                if float(q) == 0:
                    book['bids'].pop(p, None)
                else:
                    book['bids'][p] = q
            for p, q, *_ in data.get('asks', []):
                if float(q) == 0:
                    book['asks'].pop(p, None)
                else:
                    book['asks'][p] = q
        else:
            return

        ts = int(data.get('ts', int(time.time() * 1000)))
        book['ts'] = ts

        # 归一化输出为 a/b（与 binance 分支一致的两列 [price, qty]）
        sorted_bids = sorted(book['bids'].items(), key=lambda x: float(x[0]), reverse=True)
        sorted_asks = sorted(book['asks'].items(), key=lambda x: float(x[0]))

        payload = {
            'b': [[p, v] for p, v in sorted_bids],
            'a': [[p, v] for p, v in sorted_asks],
            'ts': ts,
        }
        sink(LOBRecord(ts_ms=ts, exchange='bitget', symbol=raw_symbol, payload=payload))



    def _handle_okx_update(self, raw_symbol: str, data: Dict, sink: Callable[[LOBRecord], None]):
        """Maintains a local OKX order book (price -> [qty, num_orders]) and sinks the full book on each update."""
        local_book = self.okx_order_books.setdefault(raw_symbol, {'bids': {}, 'asks': {}, 'ts': 0})

        action = data.get('action')
        if action == 'snapshot':
            # In snapshot, the 4th element is num_orders. We store [qty, num_orders].
            local_book['bids'] = {p: [q, n] for p, q, _, n in data['bids']}
            local_book['asks'] = {p: [q, n] for p, q, _, n in data['asks']}
        elif action == 'update':
            # In update, the 4th element is also num_orders.
            for p, q, _, n in data.get('bids', []):
                if float(q) == 0: local_book['bids'].pop(p, None)
                else: local_book['bids'][p] = [q, n]
            for p, q, _, n in data.get('asks', []):
                if float(q) == 0: local_book['asks'].pop(p, None)
                else: local_book['asks'][p] = [q, n]
        else:
            return

        local_book['ts'] = int(data.get('ts', int(time.time() * 1000)))

        sorted_bids = sorted(local_book['bids'].items(), key=lambda x: float(x[0]), reverse=True)
        sorted_asks = sorted(local_book['asks'].items(), key=lambda x: float(x[0]))

        # Reconstruct payload: [price, qty, liquidations_placeholder, num_orders]
        reconstructed_payload = {
            'bids': [[p, v[0], '0', v[1]] for p, v in sorted_bids],
            'asks': [[p, v[0], '0', v[1]] for p, v in sorted_asks],
            'ts': local_book['ts']
        }
        
        sink(LOBRecord(ts_ms=local_book['ts'], exchange='okx', symbol=raw_symbol, payload=reconstructed_payload))

    async def _subscribe_okx(self, raw_symbol: str, sink: Callable[[LOBRecord], None], rest_fallback: bool):
        import websockets
        if rest_fallback:
            url = "https://www.okx.com/api/v5/market/books"
            params = {"instId": raw_symbol, "sz": "400"}
            while True:
                try:
                    r = requests.get(url, params=params, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    item = data["data"][0]
                    ts = int(item.get("ts", int(time.time() * 1000)))
                    sink(LOBRecord(ts_ms=ts, exchange="okx", symbol=raw_symbol, payload=item))
                except Exception as e:
                    LOG.warning("OKX REST poll error: %s", e)
                await asyncio.sleep(1)
        else:
            url = "wss://ws.okx.com:8443/ws/v5/public"
            sub = {"op": "subscribe", "args": [{"channel": "books", "instId": raw_symbol}]}
            while True:
                try:
                    async with websockets.connect(url, ping_interval=20, open_timeout=10) as ws:
                        await ws.send(jdump(sub))
                        while True:
                            msg = await ws.recv()
                            obj = orjson.loads(msg)

                            if obj.get("event") == "subscribe": continue
                            if 'data' in obj and obj['data'] and 'action' in obj:
                                #self._handle_okx_update(raw_symbol, obj['data'][0], sink)
                                self._handle_okx_update(raw_symbol, obj['data'][0], sink, obj['action'])
                except Exception as e:
                    LOG.error("OKX WS connection failed for %s: %s", raw_symbol, e, exc_info=True)
                    if raw_symbol in self.okx_order_books:
                        del self.okx_order_books[raw_symbol]
                    await asyncio.sleep(5)

    async def _subscribe_binance(self, raw_symbol: str, sink: Callable[[LOBRecord], None], rest_fallback: bool):
        import websockets
        if rest_fallback:
            url = "https://fapi.binance.com/fapi/v1/depth"
            params = {"symbol": raw_symbol, "limit": 1000}
            while True:
                try:
                    r = requests.get(url, params=params, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    ts = int(time.time() * 1000)
                    sink(LOBRecord(ts_ms=ts, exchange="binance", symbol=raw_symbol, payload=data))
                except Exception as e:
                    LOG.warning("Binance REST poll error: %s", e)
                await asyncio.sleep(1)
        else:
            stream = f"{raw_symbol.lower()}@depth"
            url = f"wss://fstream.binance.com/ws/{stream}"
            while True:
                try:
                    async with websockets.connect(url, ping_interval=20, open_timeout=10) as ws:
                        LOG.info("Binance WS connected for %s", raw_symbol)
                        while True:
                            msg = await ws.recv()
                            obj = orjson.loads(msg)
                            ts = int(obj.get("E", int(time.time() * 1000)))
                            sink(LOBRecord(ts_ms=ts, exchange="binance", symbol=raw_symbol, payload=obj))
                except Exception as e:
                    LOG.error("Binance WS connection failed for %s: %s", raw_symbol, e, exc_info=True)
                    await asyncio.sleep(5)

    async def _subscribe_bitget(self, raw_symbol: str, sink: Callable[[LOBRecord], None], rest_fallback: bool):
        import websockets
        if rest_fallback:
            url = "https://api.bitget.com/api/v2/mix/market/merge-depth"
            params = {"productType": "usdt-futures", "symbol": raw_symbol, "limit": "200"}
            while True:
                try:
                    r = requests.get(url, params=params, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    ts = int(data.get("data", {}).get("ts", int(time.time() * 1000)))
                    sink(LOBRecord(ts_ms=ts, exchange="bitget", symbol=raw_symbol, payload=data.get("data")))
                except Exception as e:
                    LOG.warning("Bitget REST poll error: %s", e)
                await asyncio.sleep(1)
        else:
            url = "wss://ws.bitget.com/v2/ws/public"
            sub = {"op": "subscribe", "args": [{"instType": "USDT-FUTURES", "channel": "books", "instId": raw_symbol}]}
            while True:
                try:
                    async with websockets.connect(url, ping_interval=20, open_timeout=10) as ws:
                        await ws.send(jdump(sub))
                        while True:
                            msg = await ws.recv()
                            obj = orjson.loads(msg)

                            # 忽略订阅确认等
                            if obj.get("event") == "subscribe":
                                continue

                            # snapshot / update 都走同一个处理器
                            if 'action' in obj and obj.get('data'):
                                self._handle_bitget_update(raw_symbol, obj, sink)
                except Exception as e:
                    LOG.error("Bitget WS connection failed for %s: %s", raw_symbol, e, exc_info=True)
                    if raw_symbol in self.bitget_order_books:
                        del self.bitget_order_books[raw_symbol]
                    await asyncio.sleep(5)


    async def _subscribe_gate(self, raw_symbol: str, sink: Callable[[LOBRecord], None], rest_fallback: bool):
        import websockets
        if rest_fallback:
            url = "https://api.gateio.ws/api/v4/futures/usdt/order_book"
            params = {"contract": raw_symbol, "limit": 100}
            while True:
                try:
                    r = requests.get(url, params=params, timeout=10)
                    r.raise_for_status()
                    data = r.json()
                    ts = int(data.get("t", int(time.time() * 1000)))
                    sink(LOBRecord(ts_ms=ts, exchange="gate", symbol=raw_symbol, payload=data))
                except Exception as e:
                    LOG.warning("Gate REST poll error: %s", e)
                await asyncio.sleep(1)
        else:
            #{"channel":"futures.order_book_update","event":"subscribe","payload":["BTC_USDT","100ms","100"]}
            #{"time": int(time.time()), "channel": "futures.order_book", "event": "subscribe", "payload": [raw_symbol, "100ms", "100"]}
            try:
                # Gate USDT 永续 WS
                url = "wss://fx-ws.gateio.ws/v4/ws/usdt"
                sub = {
                    "channel": "futures.order_book",
                    "event": "subscribe",
                    # payload = [contract, level, interval]；interval 固定 "0"
                    "payload": [raw_symbol, "100", "0"]  # 100 档
                }
                level = 100
                self.gate_level[raw_symbol] = level  # 记录，供裁切用

                async with websockets.connect(url, ping_interval=20, open_timeout=10) as ws:
                    await ws.send(jdump(sub))
                    while True:
                        msg = await ws.recv()
                        obj = orjson.loads(msg)
                        #print(obj)
                        if obj.get("event") in ("subscribe", "pong"):
                            continue
                        if obj.get("channel") == "futures.order_book" and obj.get("event") == "all":
                            self._handle_gate_full(raw_symbol, obj, sink)

            except Exception as e:
                print(e)
                print(')'*20)
                LOG.error("Gate WS connection failed for %s: %s", raw_symbol, e, exc_info=True)
                await asyncio.sleep(5)

# ---------------- Capture orchestration ----------------

def format_lob_to_ragged_csv(record: LOBRecord, depth_n: int) -> List[str]:
    """Parses a raw LOB record into a single 'wide' CSV row."""
    payload = record.payload
    if not payload: return []

    line_parts = [record.ts_ms, record.exchange, record.symbol]
    
    try:
        bids, asks = payload.get('bids', []), payload.get('asks', [])
        
        if record.exchange == 'binance' or record.exchange == 'bitget':
            bids, asks = payload.get('b', []), payload.get('a', [])
            for i in range(depth_n):
                if i < len(asks): line_parts.extend(asks[i][:2])
                else: line_parts.extend(['', ''])
                if i < len(bids): line_parts.extend(bids[i][:2])
                else: line_parts.extend(['', ''])
            if record.exchange == 'binance':
                line_parts.append(payload.get('E', ''))

        elif record.exchange == 'okx':
            for i in range(depth_n):
                if i < len(asks): line_parts.extend(asks[i][:4])
                else: line_parts.extend(['', '', '', ''])
                if i < len(bids): line_parts.extend(bids[i][:4])
                else: line_parts.extend(['', '', '', ''])
            
        elif record.exchange == 'gate':
            # 同时兼容 'a'/'b' 与 'asks'/'bids'
            asks_raw = payload.get('a', payload.get('asks', []))
            bids_raw = payload.get('b', payload.get('bids', []))

            def _pq(item):
                # 支持 dict {'p','s'} 或 list/tuple [p,q]
                if isinstance(item, dict):
                    p, q = item.get('p'), item.get('s')
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    p, q = item[0], item[1]
                else:
                    return '', ''
                # qty 统一转成数值（失败则原样返回）
                try:
                    q = abs(float(q))
                except Exception:
                    pass
                return p, q

            for i in range(depth_n):
                if i < len(asks_raw):
                    p, q = _pq(asks_raw[i])
                    line_parts.extend([p, q])
                else:
                    line_parts.extend(['', ''])
                if i < len(bids_raw):
                    p, q = _pq(bids_raw[i])
                    line_parts.extend([p, q])
                else:
                    line_parts.extend(['', ''])


        csv_line = ','.join(map(str, line_parts)) + '\n'
        #print(csv_line)
        return [csv_line]

    except (IndexError, KeyError, TypeError, ValueError) as e:
        LOG.debug("Failed to parse LOB for %s %s: %s | payload: %s", record.exchange, record.symbol, e, payload)
        return []


async def collect_lob_for_pairs(pairs: List[PairSpread], hub: SubscriptionHub, depth_n: int):
    if not pairs: return
    next_ts = min(p.next_ts for p in pairs)
    start_ts = next_ts - 60_000
    end_ts = next_ts + 120_000
    legs = set((p.ex_a, p.raw_a) for p in pairs) | set((p.ex_b, p.raw_b) for p in pairs)
    LOG.info("Capture window [%s, %s] for %d pairs; %d legs",
             datetime.fromtimestamp(start_ts/1000, SG_TZ).isoformat(),
             datetime.fromtimestamp(end_ts/1000, SG_TZ).isoformat(),
             len(pairs), len(legs))
    for ex, raw in legs:
        try:
            await hub.ensure(ex, raw)
        except Exception as e:
            LOG.error("Ensure subscription failed for %s %s: %s", ex, raw, e)
    now_ms = int(time.time() * 1000)
    if now_ms < start_ts:
        await asyncio.sleep((start_ts - now_ms) / 1000)
    while int(time.time() * 1000) < end_ts:
        await asyncio.sleep(0.1)
    
    all_records = []
    for p in pairs:
        all_records.extend(hub.get_buffer(p.ex_a, p.raw_a))
        all_records.extend(hub.get_buffer(p.ex_b, p.raw_b))
    
    unique_records = {id(r): r for r in all_records}.values()
    filtered_records = [r for r in unique_records if start_ts <= r.ts_ms <= end_ts]
    filtered_records.sort(key=lambda r: (r.ts_ms, r.exchange))

    if not filtered_records:
        LOG.warning("No LOB records captured for pairs: %s", [p.canon for p in pairs])
        return

    fname = pairs[0].filename() + ".csv"
    path = os.path.abspath(fname)
    with open(path, "w", encoding="utf-8") as f:
        for record in filtered_records:
            csv_lines = format_lob_to_ragged_csv(record, depth_n)
            f.writelines(csv_lines)
    LOG.info("Wrote %s (%d records processed)", path, len(filtered_records))

def group_pairs_by_funding_time(pairs: List[PairSpread], threshold_ms: int = 1000) -> List[List[PairSpread]]:
    if not pairs: return []
    pairs_sorted = sorted(pairs, key=lambda p: p.next_ts)
    groups = []
    current = [pairs_sorted[0]]
    for p in pairs_sorted[1:]:
        if abs(p.next_ts - current[-1].next_ts) <= threshold_ms:
            current.append(p)
        else:
            groups.append(current)
            current = [p]
    groups.append(current)
    return groups

async def run_once(topN: int, hub: SubscriptionHub, depth_n: int):
    all_data = fetch_all_exchanges()
    pairs = compute_pair_spreads(all_data)
    top_pairs = top_n_pairs(pairs, topN)
    now_ms = int(time.time() * 1000)
    LOG.info("Top-%d spreads:", topN)
    for p in top_pairs:
        tleft = max(0, (p.next_ts - now_ms) // 1000)
        LOG.info("%s %s-%s spread=%.6f (%.4f%%) next=%s (+%ss)",
                 p.canon, p.ex_a, p.ex_b, p.spread, p.spread*100,
                 datetime.fromtimestamp(p.next_ts/1000, SG_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                 tleft)
    trigger = [p for p in top_pairs if (0 <= p.next_ts - now_ms <= 60_000)]
    if trigger:
        for group in group_pairs_by_funding_time(trigger, threshold_ms=1_000):
            await collect_lob_for_pairs(group, hub, depth_n)
    else:
        LOG.info("No pair within 60s to funding; nothing to capture this run.")

async def hourly_loop(topN: int, rest_fallback: bool, once: bool, depth_n: int):
    hub = SubscriptionHub(use_rest_fallback=rest_fallback)
    if once:
        await run_once(topN, hub, depth_n)
        return
    while True:
        now_utc = datetime.now(timezone.utc)
        next_scan_utc = compute_next_scan_time(now_utc)
        sleep_s = (next_scan_utc - now_utc).total_seconds()
        if sleep_s > 0:
            LOG.info("Sleeping %.1fs until next scan at %s (SGT %s:59:00)",
                     sleep_s,
                     next_scan_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
                     next_scan_utc.astimezone(SG_TZ).strftime("%H"))
            await asyncio.sleep(sleep_s)
        await run_once(topN, hub, depth_n)

# ---------------- Test Functions ----------------
async def test_immediate_collection():
    """Bypasses all logic to immediately collect LOB data for a hardcoded pair."""
    print("--- Running Immediate LOB Collection Test ---")
    hub = SubscriptionHub(use_rest_fallback=False)
    depth_n_to_save = 20
    collection_duration_seconds = 5

    '''
    exchange_a, symbol_a = "binance", "BTCUSDT"
    exchange_b, symbol_b = "okx", "BTC-USDT-SWAP"
    '''
    exchange_a, symbol_a = "gate", "BTC_USDT"
    exchange_b, symbol_b = "bitget", "BTCUSDT"

    print(f"Subscribing to {exchange_a}/{symbol_a} and {exchange_b}/{symbol_b}...")
    await hub.ensure(exchange_a, symbol_a)
    await hub.ensure(exchange_b, symbol_b)

    print(f"Waiting for {collection_duration_seconds} seconds to collect data...")
    await asyncio.sleep(collection_duration_seconds)

    print("Collection finished. Stopping subscriptions...")
    await hub.stop_all()
    print(hub.get_buffer(exchange_a, symbol_a))
    all_records = hub.get_buffer(exchange_a, symbol_a) + hub.get_buffer(exchange_b, symbol_b)
    #print(hub.get_buffer(exchange_a, symbol_a))
    all_records.sort(key=lambda r: r.ts_ms)

    if not all_records:
        print("\n!!! No data was collected. Check WebSocket connections or symbols. !!!")
        return

    filename = f"TEST-collection-{int(time.time())}.txt"
    filepath = os.path.abspath(filename)
    
    print(f"\nSaving {len(all_records)} captured records to {filepath}...")
    with open(filepath, "w", encoding="utf-8") as f:
        for record in all_records:
            csv_lines = format_lob_to_ragged_csv(record, depth_n_to_save)
            f.writelines(csv_lines)
    
    print("--- Immediate Collection Test Complete ---")



def test_top_funding_spreads_pairs(topN: int = 20, min_exchanges: int = 2):
    """
    返回“跨交易所两两资金费差”的 TopN 结果。
    每条记录结构: {
        symbol, ex1, rate1, ex2, rate2, spread, abs_spread,
        rates(该symbol各交易所费率dict), raw_symbols(各交易所原始合约名), next_ts_by_ex
    }
    """
    try:
        okx = fetch_okx_funding_all(timeout=15)

    except Exception as e:
        LOG.warning("fetch_okx_funding_all failed: %s", e); okx = {}
    try:
        binance = fetch_binance_funding_all(timeout=15)
    except Exception as e:
        LOG.warning("fetch_binance_funding_all failed: %s", e); binance = {}
    try:
        bitget = fetch_bitget_funding_all(timeout=15)
    except Exception as e:
        LOG.warning("fetch_bitget_funding_all failed: %s", e); bitget = {}
    try:
        gate = fetch_gate_funding_all(timeout=15)
    except Exception as e:
        LOG.warning("fetch_gate_funding_all failed: %s", e); gate = {}

    books = {"okx": okx, "binance": binance, "bitget": bitget, "gate": gate}

    # 汇总所有 canonical symbols
    all_symbols = set().union(*[set(d.keys()) for d in books.values()])

    rows = []
    for sym in all_symbols:
        rates = {}
        next_ts_by_ex = {}
        raw_symbols = {}
        for ex, d in books.items():
            v = d.get(sym)
            if not v:
                continue
            try:
                r = float(v.get("rate"))
            except Exception:
                continue
            rates[ex] = r
            next_ts_by_ex[ex] = int(v.get("next_ts", 0))
            raw_symbols[ex] = v.get("raw_symbol", "")

        if len(rates) < min_exchanges:
            continue

        ex_items = list(rates.items())  # [(ex, rate), ...]
        for i in range(len(ex_items)):
            for j in range(i + 1, len(ex_items)):
                ex1, r1 = ex_items[i]
                ex2, r2 = ex_items[j]
                spread = r1 - r2
                rows.append({
                    "symbol": sym,
                    "ex1": ex1, "rate1": r1,
                    "ex2": ex2, "rate2": r2,
                    "spread": spread,
                    "abs_spread": abs(spread),
                    "rates": rates,
                    "raw_symbols": raw_symbols,
                    "next_ts_by_ex": next_ts_by_ex,
                })

    rows.sort(key=lambda r: r["abs_spread"], reverse=True)
    return rows[:topN]


def print_top_funding_spreads_pairs(topN: int = 20, unit: str = "decimal"):
    """
    unit: 'decimal'（小数）, 'pct'（百分比）, 'bps'
    """
    rows = test_top_funding_spreads_pairs(topN=topN)
    def fmt(x):
        if unit == "pct": return f"{x*100:.4f}%"
        if unit == "bps": return f"{x*10000:.1f} bps"
        return f"{x:.6f}"

    for i, r in enumerate(rows, 1):
        rates_str = ", ".join(f"{ex}: {fmt(rate)}" for ex, rate in r["rates"].items())
        print(
            f"{i:02d}. {r['symbol']}: spread={fmt(r['spread'])} "
            f"[{r['ex1']} {fmt(r['rate1'])} ↔ {r['ex2']} {fmt(r['rate2'])}] "
            f"rates={{ {rates_str} }}"
        )




def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--topN", type=int, default=10, help="Select top-N spreads")
    ap.add_argument("--depthN", type=int, default=20, help="Select depth-N of LOB to parse and save")
    ap.add_argument("--rest-fallback", action="store_true", help="Force REST polling instead of WebSockets")
    ap.add_argument("--once", action="store_true", help="Run once and exit (no hourly loop)")
    args = ap.parse_args()
    try:
        asyncio.run(hourly_loop(args.topN, rest_fallback=args.rest_fallback, once=args.once, depth_n=args.depthN))
    except KeyboardInterrupt:
        LOG.info("Exit by user")

if __name__ == "__main__":
    # To run the main application, comment out the active test and uncomment main()
    try:
        asyncio.run(main())
        #print(test_full_formatter())
        #print_top_funding_spreads_pairs(20, unit="pct")

    except KeyboardInterrupt:
        print("\nExit by user.")