"""
DEX-DEX Arbitrage Bot Framework
语言: Python 3.11+
并发模型: asyncio + uvloop
依赖: web3.py, eth_abi, aiohttp, pydantic, numpy, pandas, msgspec(or orjson), python-decouple, tenacity
(可选) flashbots, mev-share, web3-eth-account
区块链: EVM 链 (以太坊/Arbitrum/OP/BSC/Polygon)，可扩展至非EVM (需适配器)
"""

# ─────────────────────────────── 1. 架构总览 ───────────────────────────────
# 发现 → 定价 → 路由 → 模拟校验 → 打包/执行 → 监控回滚
#              └── 风险控制 & 日志 → 持久化
#
# ASCII Diagram
#
#                +-------------------+
#  mempool/ws  → |  Event Listener   | → on_new_block / on_tx
#                +---------+---------+
#                          |
#                          v
#                +-------------------+
#                |  Quote Engine     |  估算两边DEX的买卖价、滑点、gas
#                +---------+---------+
#                          |
#                          v
#                +-------------------+
#                |  Opportunity Fn   | → (expected_profit > threshold?)
#                +---------+---------+
#                          |
#              yes         v        no
#                    +-----------+
#                    | Simulator |  逐笔/逐步状态机重放, 包含 MEV 风险/失败回滚
#                    +-----------+
#                          |
#                        pass
#                          v
#                +-------------------+
#                |  Executor         | → EOA or Flashbots/MEV-Share bundle
#                +---------+---------+
#                          |
#                          v
#                +-------------------+
#                |  Monitor/Logger   |
#                +-------------------+


# ─────────────────────────────── 2. 目录结构 ───────────────────────────────
# project_root/
#   config/
#       chains.yaml            # RPC、gas 参数、私钥路径(只在本地 .env)
#       dex.yaml               # DEX 列表、池子地址、fee tier 等
#   core/
#       amm.py                 # 各类AMM公式(CP, StableSwap, UniV3区间流动性)
#       quote.py               # 估算成交价格 & 滑点
#       state.py               # 缓存链上状态, block snapshot
#       math_utils.py          # 精度、Q64.96、sqrtPriceX96工具
#   dex/
#       base.py                # DEX 接口抽象类
#       uniswap_v2.py
#       uniswap_v3.py
#       curve.py
#   infra/
#       web3ext.py             # 异步web3封装(或使用 "web3-async")
#       mempool.py             # 监听pending tx、new block
#       db.py                  # SQLite/ClickHouse/Parquet 持久化
#       logging.py             # 结构化日志
#   engine/
#       discovery.py           # 枚举池子/路径 pair graph 建图
#       opportunity.py         # 机会检测器 (Bellman-Ford/DFS)
#       simulator.py           # 交易模拟、滑点、gas、MEV 风险
#       executor.py            # 发送交易/Flashbots/MultiCall
#       risk.py                # 风险控制模块
#   scripts/
#       backfill_pools.py      # 一次性拉取池子列表
#       warmup_state.py        # 预拉当前价格、liquidity
#   main.py                    # 入口
#   requirements.txt
#   .env                       # 私钥/Flashbots key，不要提交Git


# ─────────────────────────────── 3. 配置示例 ───────────────────────────────
# config/chains.yaml
chains_yaml = """
ethereum:
  rpc: https://mainnet.infura.io/v3/${INFURA_KEY}
  ws:  wss://mainnet.infura.io/ws/v3/${INFURA_KEY}
  chain_id: 1
  max_gas_price_wei: 150000000000  # 150 gwei
  flashbots_relay: https://relay.flashbots.net
arbitrum:
  rpc: https://arb1.arbitrum.io/rpc
  ws:  wss://arb1.arbitrum.io/ws
  chain_id: 42161
"""

# config/dex.yaml
.dex_yaml = """
uniswap_v2:
  factory: 0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f
  init_code_hash: 0x96e8ac427...
  fee: 0.003
uniswap_v3:
  factory: 0x1F98431c8aD98523631AE4a59f267346ea31F984
  quoter:  0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6
  fees: [500, 3000, 10000]
curve:
  registry: 0x0000000022d53366457F9d5E68Ec105046FC4383
"""


# ─────────────────────────────── 4. 抽象接口 ───────────────────────────────
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, List, Dict, Tuple, Optional
from decimal import Decimal

@dataclass
class SwapResult:
    amount_in: int
    amount_out: int
    gas_used: int
    success: bool
    route: List[str]

class DEX(Protocol):
    name: str
    async def get_price(self, token_in: str, token_out: str, amount_in: int, block: int | str = "latest") -> int: ...
    async def simulate_swap(self, token_in: str, token_out: str, amount_in: int, block: int | str = "latest") -> SwapResult: ...
    async def build_tx(self, token_in: str, token_out: str, amount_in: int, min_out: int, to: str, nonce: int) -> Dict: ...


# ─────────────────────────────── 5. AMM 数学工具 (core/amm.py) ───────────────────────────────
# 1) Uniswap V2 (恒定积 x*y=k)

def univ2_get_amount_out(amount_in: int, reserve_in: int, reserve_out: int, fee_bps: int = 30) -> int:
    fee_adj = amount_in * (10000 - fee_bps)
    numerator = fee_adj * reserve_out
    denominator = reserve_in * 10000 + fee_adj
    return numerator // denominator

# 2) Uniswap V3: 使用 sqrtPriceX96 和 liquidity
# Δy = L * (ΔsqrtP), Δx = L * (Δ(1/sqrtP))
# 这里仅提供骨架；具体实现需要精度处理

# 3) Curve StableSwap 近似: 使用 invariant D 的迭代法求解


# ─────────────────────────────── 6. 机会检测 (engine/opportunity.py) ───────────────────────────────
import asyncio
from itertools import product

@dataclass
class Opportunity:
    path: Tuple[str, str]
    amount_in: int
    expected_profit: int
    dex_in: str
    dex_out: str
    gas_cost: int

class OpportunityFinder:
    def __init__(self, dex_a: DEX, dex_b: DEX, tokens: List[str], amount_grid: List[int], gas_oracle):
        self.dex_a = dex_a
        self.dex_b = dex_b
        self.tokens = tokens
        self.amount_grid = amount_grid
        self.gas_oracle = gas_oracle

    async def scan_once(self) -> List[Opportunity]:
        ops: List[Opportunity] = []
        tasks = []
        for t_in, t_out in product(self.tokens, repeat=2):
            if t_in == t_out: 
                continue
            for amt in self.amount_grid:
                tasks.append(self._check_pair(t_in, t_out, amt))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Opportunity) and r.expected_profit > 0:
                ops.append(r)
        return ops

    async def _check_pair(self, token_in, token_out, amt) -> Optional[Opportunity]:
        price_a = await self.dex_a.get_price(token_in, token_out, amt)
        price_b = await self.dex_b.get_price(token_in, token_out, amt)
        # route1: buy on A → sell on B
        profit1 = price_b - amt  # 假设token_in是基准
        gas_cost = await self.gas_oracle.estimate()
        if profit1 > gas_cost:
            return Opportunity((token_in, token_out), amt, profit1 - gas_cost, self.dex_a.name, self.dex_b.name, gas_cost)
        # route2: buy on B → sell on A
        profit2 = price_a - amt
        if profit2 > gas_cost:
            return Opportunity((token_in, token_out), amt, profit2 - gas_cost, self.dex_b.name, self.dex_a.name, gas_cost)
        return None


# ─────────────────────────────── 7. 模拟执行 (engine/simulator.py) ───────────────────────────────
class Simulator:
    def __init__(self, web3, dex_map: Dict[str, DEX]):
        self.web3 = web3
        self.dex_map = dex_map

    async def simulate(self, op: Opportunity) -> SwapResult:
        dex_in = self.dex_map[op.dex_in]
        dex_out = self.dex_map[op.dex_out]
        # 1. 模拟第一个swap
        r1 = await dex_in.simulate_swap(op.path[0], op.path[1], op.amount_in)
        if not r1.success:
            return r1
        # 2. 模拟第二个swap (反向)
        r2 = await dex_out.simulate_swap(op.path[1], op.path[0], r1.amount_out)
        return SwapResult(op.amount_in, r2.amount_out, r1.gas_used + r2.gas_used, r1.success and r2.success, [dex_in.name, dex_out.name])


# ─────────────────────────────── 8. 执行器 (engine/executor.py) ───────────────────────────────
class Executor:
    def __init__(self, web3, priv_key: str, flashbots=None):
        self.web3 = web3
        self.account = web3.eth.account.from_key(priv_key)
        self.flashbots = flashbots

    async def send_bundle(self, txs: List[Dict], block_tag: int):
        # 调用 flashbots 提交保护交易，略
        pass

    async def send_tx(self, tx: Dict):
        tx['nonce'] = self.web3.eth.get_transaction_count(self.account.address)
        signed = self.account.sign_transaction(tx)
        tx_hash = self.web3.eth.send_raw_transaction(signed.rawTransaction)
        return tx_hash.hex()


# ─────────────────────────────── 9. main.py (入口) ───────────────────────────────
import asyncio
import uvloop

async def main():
    # 1. 加载配置
    # 2. 初始化 web3, dex 实例
    # 3. 暖启动 (预拉池子信息/链上状态)
    # 4. 事件循环: 每个block或固定时间扫描一次
    opportunity_finder = ...
    simulator = ...
    executor = ...

    while True:
        ops = await opportunity_finder.scan_once()
        for op in ops:
            sim_res = await simulator.simulate(op)
            if sim_res.success and sim_res.amount_out - op.amount_in > op.gas_cost:
                # 构建真实交易
                txs = []  # build route tx list (approve -> swap1 -> swap2)
                # 尽量合并Call, 使用 multicall/permit2 减少gas
                # 执行
                await executor.send_tx(txs[0])
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())


# ─────────────────────────────── 10. 风险控制要点 ───────────────────────────────
"""
1. Slippage/滑点: 设置严格的 minOut, 并在模拟时加入池子深度变化的容忍度。
2. Gas/MEV: gas 预估 + Flashbots/私有RPC，避免被夹。(也可透过 mev-share 监听 sandwich 风险)
3. Token Transfer Tax/钓鱼token: 白名单token；读取 bytecode & events 检测。 
4. Revert 风险: 对每一步 on-chain 调用 try/callStatic 先行模拟，必要时多链路 fallback。
5. 价格刷新频率: 高频扫描会消耗RPC，需做缓存/快照。
6. 资金管理: 拆分多钱包/多链部署, 控制单笔风险。
"""

# ─────────────────────────────── 11. 后续扩展 ───────────────────────────────
"""
- 多跳路径: tokenA → tokenB → tokenC → tokenA，利用路径非对称性
- 组合型套利: DEX↔CEX, perp↔spot, options↔futures
- ZK/Optimistic Rollup: 处理跨域消息延迟, 使用跨链桥快速转移资金
- Rust 实现: 将核心数学与执行模块用 Rust 编写，Python 仅做策略层
- Backtest: 用历史事件日志 (Swap topics) 重建池子状态，离线回测套利收益
"""


# ─────────────────────────────── 12. L2 多链套利扫描器 ───────────────────────────────
"""
目标：同时在多条 L2 (Arbitrum/OP/Base/zkSync...) 上抓取 DEX-DEX 价格差机会。
设计要点：
1. **多链并发**：每条链一个 `ChainContext`，共享统一任务队列与日志层。
2. **抽象 RPC 层**：统一 `AsyncWeb3` 接口，支持 failover（Alchemy/Infura/自建节点）。
3. **池子图缓存**：每条链启动时构建 token graph，周期性刷新（监听新池事件）。
4. **批量 callStatic**：对 UniV3 Quoter、Curve Calculator 做 multicall，降低 RPC 次数。
5. **精细 Gas 估计**：各链不同 gasToken（ETH/ARB/OP），统一换算到 USD 或稳定币。
6. **回测/线上共用模块**：扫描逻辑与模拟模块保持纯函数，方便离线重放。
"""

# core/chain.py
from dataclasses import dataclass
from typing import Any, Callable
import asyncio

@dataclass
class ChainConfig:
    name: str
    rpc: str
    ws: str | None
    chain_id: int
    native_symbol: str
    max_gas_price_wei: int

class ChainContext:
    def __init__(self, cfg: ChainConfig, web3_async, dex_map: dict[str, DEX], gas_oracle):
        self.cfg = cfg
        self.web3 = web3_async
        self.dex_map = dex_map
        self.gas_oracle = gas_oracle
        self.block_subscribers: list[Callable[[int], Any]] = []
        self._latest_block = 0

    async def start_block_listener(self):
        while True:
            try:
                blk = await self.web3.eth_block_number()
                if blk != self._latest_block:
                    self._latest_block = blk
                    for cb in self.block_subscribers:
                        asyncio.create_task(cb(blk))
            except Exception as e:
                # log error
                await asyncio.sleep(1)
            await asyncio.sleep(0.3)

    def on_new_block(self, cb: Callable[[int], Any]):
        self.block_subscribers.append(cb)


# engine/multi_chain_scanner.py
from typing import Sequence

class MultiChainScanner:
    def __init__(self, chains: Sequence[ChainContext], opp_finders: dict[str, OpportunityFinder], simulator: Simulator, executor: Executor, profit_thresh_usd: float):
        self.chains = chains
        self.opp_finders = opp_finders  # key = chain_name
        self.simulator = simulator
        self.executor = executor
        self.profit_thresh_usd = profit_thresh_usd
        self.price_feed = None  # 可插入 Chainlink/自建报价，用于折算USD

    async def run_forever(self):
        tasks = [asyncio.create_task(self._loop_chain(chain)) for chain in self.chains]
        await asyncio.gather(*tasks)

    async def _loop_chain(self, chain: ChainContext):
        finder = self.opp_finders[chain.cfg.name]
        # 挂在 new block 回调，或用固定频率轮询
        async def on_block(_blk: int):
            ops = await finder.scan_once()
            if not ops:
                return
            sims = await asyncio.gather(*(self.simulator.simulate(op) for op in ops))
            for op, sim_res in zip(ops, sims):
                if not sim_res.success:
                    continue
                profit = sim_res.amount_out - op.amount_in - op.gas_cost
                if profit <= 0:
                    continue
                profit_usd = await self._to_usd(chain, profit, op.path[0])
                if profit_usd >= self.profit_thresh_usd:
                    await self._execute(chain, op, sim_res)
        chain.on_new_block(on_block)
        await chain.start_block_listener()

    async def _to_usd(self, chain: ChainContext, amount: int, token: str) -> float:
        # 插入价格预言机 or off-chain价格
        return 0.0

    async def _execute(self, chain: ChainContext, op: Opportunity, sim_res: SwapResult):
        # 构建 & 发送交易（也可 bundle）
        pass


# infra/web3_async.py (简化版)
import aiohttp
import json

class AsyncWeb3:
    def __init__(self, rpc_url: str, session: aiohttp.ClientSession):
        self.url = rpc_url
        self.session = session
        self._id = 0

    async def _rpc(self, method: str, params: list[Any]):
        self._id += 1
        payload = {"jsonrpc":"2.0","id":self._id,"method":method,"params":params}
        async with self.session.post(self.url, json=payload, timeout=10) as resp:
            data = await resp.json()
            if 'error' in data:
                raise RuntimeError(data['error'])
            return data['result']

    async def eth_block_number(self) -> int:
        return int(await self._rpc('eth_blockNumber', []), 16)

    async def call(self, call_obj: dict, block_tag: str = 'latest') -> bytes:
        return bytes.fromhex((await self._rpc('eth_call', [call_obj, block_tag]))[2:])

    # ... 其他需要的 async 方法


# main_multichain.py
import asyncio
import uvloop
import aiohttp

async def main():
    # 1. 读取 chains.yaml, dex.yaml
    # 2. 初始化 aiohttp session & AsyncWeb3
    session = aiohttp.ClientSession()
    chains_cfg = [
        ChainConfig("arbitrum", "https://arb1.arbitrum.io/rpc", None, 42161, "ETH", 5_000_000_000),
        ChainConfig("optimism", "https://mainnet.optimism.io", None, 10, "ETH", 5_000_000_000),
        ChainConfig("base", "https://base-mainnet.g.alchemy.com/v2/${KEY}", None, 8453, "ETH", 5_000_000_000),
    ]
    chains: list[ChainContext] = []

    for cfg in chains_cfg:
        web3a = AsyncWeb3(cfg.rpc, session)
        dex_map = await build_dex_map(cfg, web3a)   # 你实现: 返回 {"univ3": UniswapV3Dex(...), ...}
        gas_oracle = await build_gas_oracle(cfg, web3a)
        chains.append(ChainContext(cfg, web3a, dex_map, gas_oracle))

    # 3. 为每条链创建 OpportunityFinder
    opp_finders = {}
    for chain in chains:
        tokens = await load_tokens_for_chain(chain.cfg.name)
        amount_grid = [10**18, 5*10**18, 1*10**19]  # 1,5,10 ETH 等级
        finder = OpportunityFinder(
            dex_a=list(chain.dex_map.values())[0],
            dex_b=list(chain.dex_map.values())[1],
            tokens=tokens,
            amount_grid=amount_grid,
            gas_oracle=chain.gas_oracle
        )
        opp_finders[chain.cfg.name] = finder

    simulator = Simulator(None, {})  # 这里可用一个全局 simulator 或每链一个
    executor = Executor(None, priv_key="${PRIVATE_KEY}")

    mscanner = MultiChainScanner(chains, opp_finders, simulator, executor, profit_thresh_usd=5.0)
    await mscanner.run_forever()

if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())


# ─────────────────────────────── 13. 性能与可靠性建议 ───────────────────────────────
"""
- **批量 multicall**：对 UniV3 quoter 一次性估价一组 token/amount；Curve 也可合成 invocations。
- **限流 & 熔断**：RPC 错误率上升时及时 failover；缓存最近一次价格结果。
- **并发窗口**：扫描→模拟→执行要流水线化，避免阻塞；使用 `asyncio.Queue` 解耦。
- **持久化**：把发现的机会、模拟结果写入 ClickHouse/Parquet，方便复盘。
- **测试**：
  - 单元测试：AMM 数学函数、路径搜索算法；
  - 集成测试：在 Forked chain (anvil, hardhat) 自动回放真实区块。
- **监控**：Prometheus/Grafana 记录吞吐、成功率、利润分布。
"""
