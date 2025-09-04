import asyncio
import websockets
import ujson as json
import os
import time
from datetime import datetime, timezone
import aiofiles
from collections import defaultdict

# -------------------- Config --------------------
GATE_WS_URL = "wss://fx-ws.gateio.ws/v4/ws/usdt"

# 头部输出模式：
#   'gate'         -> recv_ts_sec,engine_ts_ms,recv_ts_ms,event,u,
#   'binance_like' -> T,U,u,pu,
HEADER_MODE = 'gate'

symbols = [
    'BTC_USDT','ETH_USDT','SOL_USDT','XRP_USDT','DOGE_USDT','LTC_USDT','BNB_USDT',
    'ADA_USDT','AVAX_USDT','LINK_USDT','SUI_USDT','ETC_USDT','ATOM_USDT','UNI_USDT',
    'TON_USDT','DOT_USDT','NEAR_USDT','TRX_USDT','BCH_USDT','AAVE_USDT','FIL_USDT',
    'OP_USDT','ARB_USDT','APT_USDT','MATIC_USDT','TIA_USDT'
]

# -------------------- Buffered writer --------------------
class BufferedDailyFileWriter:
    def __init__(self, root: str = "gate_data", buffer_size: int = 1000, flush_interval: int = 60):
        self.root = root
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.fp_map = {}
        self.buffers = defaultdict(list)
        self.locks = defaultdict(asyncio.Lock)
        self.cur_date = self._today()
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush())

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def write(self, contract: str, line: str):
        if not line:
            return
        today = self._today()
        if today != self.cur_date:
            await self._rollover(today)

        key = (contract, today)
        self.buffers[key].append(line)
        if len(self.buffers[key]) >= self.buffer_size:
            asyncio.create_task(self.flush(key))

    async def flush(self, key: tuple):
        async with self.locks[key]:
            if not self.buffers[key]:
                return
            contract, date_str = key
            lines_to_write = self.buffers.pop(key)
            try:
                if key not in self.fp_map:
                    path = os.path.join(self.root, date_str)
                    os.makedirs(path, exist_ok=True)
                    fp = await aiofiles.open(f"{path}/{contract}.txt", "a")
                    self.fp_map[key] = fp
                content = "".join(lines_to_write)
                await self.fp_map[key].write(content)
            except Exception as e:
                print(f"Failed to flush buffer for key {key}: {e}")
                self.buffers[key].extend(lines_to_write)

    async def _periodic_flush(self):
        while self._running:
            await asyncio.sleep(self.flush_interval)
            keys_to_flush = list(self.buffers.keys())
            for key in keys_to_flush:
                if self.buffers[key]:
                    await self.flush(key)

    async def _rollover(self, new_date: str):
        await self.flush_all()
        for fp in self.fp_map.values():
            await fp.close()
        self.fp_map.clear()
        self.cur_date = new_date

    async def flush_all(self):
        keys_to_flush = list(self.buffers.keys())
        await asyncio.gather(*(self.flush(key) for key in keys_to_flush))

    async def close_all(self):
        self._running = False
        self._flush_task.cancel()
        await self.flush_all()
        for fp in self.fp_map.values():
            await fp.close()
        self.fp_map.clear()

# -------------------- Diff builder --------------------
def contrast_diff_line(meta_prefix: str, new_a, new_b, old_a, old_b) -> str:
    """
    对比 new 与 old 的前 20 档，只记录变化；
    - 存在->存在且不同：写 new_price,new_size,idx
    - 不存在->存在：写 new_price,new_size,idx
    - 存在->不存在：写 old_price,0,idx
    - 不存在->不存在：忽略
    """
    changes = []

    def handle_side(new_side, old_side, base_idx):
        for i in range(20):
            has_new = i < len(new_side)
            has_old = i < len(old_side)
            if has_new and has_old:
                if new_side[i] != old_side[i]:
                    p, s = new_side[i]
                    changes.append([str(p), str(s), str(base_idx + i)])
            elif has_new and not has_old:
                p, s = new_side[i]
                changes.append([str(p), str(s), str(base_idx + i)])
            elif has_old and not has_new:
                p, _ = old_side[i]
                changes.append([str(p), "0", str(base_idx + i)])

    handle_side(new_a, old_a, 0)      # asks -> 0..19
    handle_side(new_b, old_b, 20)     # bids -> 20..39

    if not changes:
        return ""
    flattened = ",".join(sum(changes, []))
    return f"{meta_prefix}{flattened}\n"

# -------------------- Helpers --------------------
async def subscribe(ws):
    now = int(time.time())
    for sym in symbols:
        sub = {
            "time": now,
            "channel": "futures.order_book",
            "event": "subscribe",
            "payload": [sym, "100ms"]
        }
        await ws.send(json.dumps(sub))
        await asyncio.sleep(0.05)

async def heartbeat(ws):
    while True:
        try:
            await ws.ping()
        except Exception:
            break
        await asyncio.sleep(25)

def _normalize_top20(asks, bids):
    try:
        asks_sorted = sorted(asks, key=lambda x: float(x[0]))
    except Exception:
        asks_sorted = asks
    try:
        bids_sorted = sorted(bids, key=lambda x: float(x[0]), reverse=True)
    except Exception:
        bids_sorted = bids
    a20 = [[str(p), str(s)] for p, s in asks_sorted[:20]]
    b20 = [[str(p), str(s)] for p, s in bids_sorted[:20]]
    return a20, b20

# -------------------- Main --------------------
async def run():
    fw = BufferedDailyFileWriter(buffer_size=1000, flush_interval=60)
    prev_book = {}  # contract -> {"a":[[p,s],...],"b":[[p,s],...]}
    prev_seq = {}   # contract -> last U for pu

    while True:
        try:
            async with websockets.connect(GATE_WS_URL, ping_interval=None, ping_timeout=None) as ws:
                await subscribe(ws)
                asyncio.create_task(heartbeat(ws))

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue

                    if data.get("channel") != "futures.order_book":
                        continue
                    if data.get("event") not in ("update", "all"):
                        continue

                    res = data.get("result") or {}
                    contract = res.get("contract", "UNKNOWN")
                    engine_ts = int(res.get("t", 0))  # exchange engine ms timestamp if provided
                    U_val = res.get("u")              # gate's update id if provided

                    asks = res.get("asks", [])
                    bids = res.get("bids", [])
                    a20, b20 = _normalize_top20(asks, bids)

                    recv_ts = datetime.utcnow()
                    recv_ts_sec = int(recv_ts.timestamp())
                    recv_ts_ms = int(recv_ts.timestamp() * 1000)

                    if HEADER_MODE == 'gate':
                        meta_prefix = f"{recv_ts_sec},{engine_ts},{recv_ts_ms},{data.get('event','')},{U_val if U_val is not None else ''},"
                    else:
                        last = prev_seq.get(contract, 0)
                        if U_val is None:
                            # fallback: simple local counter
                            U_val = last + 1 if last else 1
                        U = int(U_val)
                        T = int(engine_ts)
                        u = U
                        pu = int(last)
                        meta_prefix = f"{T},{U},{u},{pu},"
                        prev_seq[contract] = U

                    old = prev_book.get(contract, {"a": [], "b": []})
                    line = contrast_diff_line(meta_prefix, a20, b20, old.get("a", []), old.get("b", []))
                    if line:
                        await fw.write(contract, line)
                    prev_book[contract] = {"a": a20, "b": b20}

        except Exception as e:
            print("连接断开，重连中...", e)
            await asyncio.sleep(3)

if __name__ == "__main__":
    asyncio.run(run())
