# WebSocket跨所三角套利策略
# 支持Bitget、Binance、Gate.io的实时深度数据和WebSocket下单
# 作者：千千量化

import asyncio
import websockets
import json
import time
import logging
import hmac
import hashlib
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import threading
import queue

@dataclass
class OrderBookData:
    """订单簿数据结构"""
    symbol: str
    exchange: str
    bids: List[List[float]]  # [[price, size], ...]
    asks: List[List[float]]  # [[price, size], ...]
    timestamp: float

@dataclass
class ArbitrageSignal:
    """套利信号数据结构"""
    mode: str  # '112', '211', '221', '122'
    base_currency: str  # X
    quote_currency: str  # Y
    target_currency: str  # Z
    profit_ratio: float
    estimated_profit: float
    orders: List[Dict]  # 需要执行的订单

class WebSocketExchange:
    """WebSocket交易所基类"""
    
    def __init__(self, name: str, api_key: str, secret_key: str, passphrase: str = ""):
        self.name = name
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.ws_url = ""
        self.order_ws_url = ""
        self.orderbook_data = {}
        self.balance_data = {}
        self.logger = logging.getLogger(f"{name}_ws")
        
    async def connect_orderbook(self, symbols: List[str]):
        """连接订单簿WebSocket"""
        raise NotImplementedError
        
    async def connect_orders(self):
        """连接订单WebSocket"""
        raise NotImplementedError
        
    async def place_order(self, symbol: str, side: str, order_type: str, amount: float, price: float = None):
        """下单"""
        raise NotImplementedError
        
    def get_orderbook(self, symbol: str) -> Optional[OrderBookData]:
        """获取订单簿数据"""
        return self.orderbook_data.get(symbol)

class BinanceWebSocket(WebSocketExchange):
    """Binance WebSocket实现"""
    
    def __init__(self, api_key: str, secret_key: str):
        super().__init__("binance", api_key, secret_key)
        self.ws_url = "wss://stream.binance.com:9443/ws"
        self.order_ws_url = "wss://ws-api.binance.com:443/ws-api/v3"
        
    async def connect_orderbook(self, symbols: List[str]):
        """连接Binance订单簿WebSocket"""
        # 构建订阅消息
        streams = [f"{symbol.lower()}@depth20@100ms" for symbol in symbols]
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        
        async with websockets.connect(self.ws_url) as websocket:
            await websocket.send(json.dumps(subscribe_msg))
            self.logger.info(f"Binance订单簿WebSocket已连接，订阅: {symbols}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if 'stream' in data and 'data' in data:
                        await self._handle_orderbook_data(data['data'], data['stream'])
                except Exception as e:
                    self.logger.error(f"处理Binance订单簿数据错误: {e}")
                    
    async def _handle_orderbook_data(self, data: Dict, stream: str):
        """处理订单簿数据"""
        symbol = stream.split('@')[0].upper()
        
        # 转换为标准格式
        bids = [[float(bid[0]), float(bid[1])] for bid in data['bids']]
        asks = [[float(ask[0]), float(ask[1])] for ask in data['asks']]
        
        orderbook = OrderBookData(
            symbol=symbol,
            exchange="binance",
            bids=bids,
            asks=asks,
            timestamp=time.time()
        )
        
        self.orderbook_data[symbol] = orderbook
        
    async def place_order(self, symbol: str, side: str, order_type: str, amount: float, price: float = None):
        """Binance WebSocket下单"""
        timestamp = int(time.time() * 1000)
        
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(amount),
            "timestamp": timestamp
        }
        
        if price:
            params["price"] = str(price)
            params["timeInForce"] = "GTC"
            
        # 生成签名
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        order_msg = {
            "id": f"order_{int(time.time() * 1000)}",
            "method": "order.place",
            "params": {
                **params,
                "signature": signature
            }
        }
        
        # 这里需要建立认证的WebSocket连接
        # 实际实现中需要处理认证流程
        return order_msg

class BitgetWebSocket(WebSocketExchange):
    """Bitget WebSocket实现"""
    
    def __init__(self, api_key: str, secret_key: str, passphrase: str):
        super().__init__("bitget", api_key, secret_key, passphrase)
        self.ws_url = "wss://ws.bitget.com/spot/v1/stream"
        
    async def connect_orderbook(self, symbols: List[str]):
        """连接Bitget订单簿WebSocket"""
        subscribe_msg = {
            "op": "subscribe",
            "args": [{"instType": "SPOT", "channel": "books", "instId": symbol} for symbol in symbols]
        }
        
        async with websockets.connect(self.ws_url) as websocket:
            await websocket.send(json.dumps(subscribe_msg))
            self.logger.info(f"Bitget订单簿WebSocket已连接，订阅: {symbols}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if 'data' in data:
                        await self._handle_orderbook_data(data['data'][0])
                except Exception as e:
                    self.logger.error(f"处理Bitget订单簿数据错误: {e}")
                    
    async def _handle_orderbook_data(self, data: Dict):
        """处理订单簿数据"""
        symbol = data['instId']
        
        bids = [[float(bid[0]), float(bid[1])] for bid in data['bids']]
        asks = [[float(ask[0]), float(ask[1])] for ask in data['asks']]
        
        orderbook = OrderBookData(
            symbol=symbol,
            exchange="bitget",
            bids=bids,
            asks=asks,
            timestamp=time.time()
        )
        
        self.orderbook_data[symbol] = orderbook
        
    async def place_order(self, symbol: str, side: str, order_type: str, amount: float, price: float = None):
        """Bitget WebSocket下单"""
        timestamp = str(int(time.time() * 1000))
        
        order_data = {
            "instId": symbol,
            "tdMode": "cash",
            "side": side.lower(),
            "ordType": order_type.lower(),
            "sz": str(amount)
        }
        
        if price:
            order_data["px"] = str(price)
            
        # 生成签名
        message = timestamp + 'POST' + '/api/spot/v1/trade/orders' + json.dumps(order_data)
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        
        order_msg = {
            "op": "order",
            "args": [order_data],
            "timestamp": timestamp,
            "signature": signature
        }
        
        return order_msg

class GateioWebSocket(WebSocketExchange):
    """Gate.io WebSocket实现"""
    
    def __init__(self, api_key: str, secret_key: str):
        super().__init__("gateio", api_key, secret_key)
        self.ws_url = "wss://api.gateio.ws/ws/v4/"
        
    async def connect_orderbook(self, symbols: List[str]):
        """连接Gate.io订单簿WebSocket"""
        subscribe_msg = {
            "method": "spot.order_book",
            "params": symbols + ["20", "100ms"],
            "id": 1
        }
        
        async with websockets.connect(self.ws_url) as websocket:
            await websocket.send(json.dumps(subscribe_msg))
            self.logger.info(f"Gate.io订单簿WebSocket已连接，订阅: {symbols}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if 'method' in data and data['method'] == 'spot.order_book':
                        await self._handle_orderbook_data(data['params'])
                except Exception as e:
                    self.logger.error(f"处理Gate.io订单簿数据错误: {e}")
                    
    async def _handle_orderbook_data(self, params: List):
        """处理订单簿数据"""
        symbol = params[2]  # currency_pair
        orderbook_data = params[1]
        
        bids = [[float(bid[0]), float(bid[1])] for bid in orderbook_data['bids']]
        asks = [[float(ask[0]), float(ask[1])] for ask in orderbook_data['asks']]
        
        orderbook = OrderBookData(
            symbol=symbol,
            exchange="gateio",
            bids=bids,
            asks=asks,
            timestamp=time.time()
        )
        
        self.orderbook_data[symbol] = orderbook
        
    async def place_order(self, symbol: str, side: str, order_type: str, amount: float, price: float = None):
        """Gate.io WebSocket下单"""
        timestamp = str(int(time.time()))
        
        order_data = {
            "currency_pair": symbol,
            "side": side.lower(),
            "type": order_type.lower(),
            "amount": str(amount)
        }
        
        if price:
            order_data["price"] = str(price)
            
        # 生成签名
        body = json.dumps(order_data)
        message = f"POST\n/spot/orders\n\n{body}"
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
        
        order_msg = {
            "method": "spot.order",
            "params": [order_data],
            "timestamp": timestamp,
            "signature": signature
        }
        
        return order_msg

class WebSocketTriangularArbitrage:
    """基于WebSocket的跨所三角套利策略"""
    
    def __init__(self, exchanges: List[WebSocketExchange], ratio: float = 0.8):
        self.exchanges = {ex.name: ex for ex in exchanges}
        self.ratio = ratio
        self.logger = logging.getLogger("arbitrage")
        self.signal_queue = queue.Queue()
        self.running = False
        
        # 统计信息
        self.signal_count = 0
        self.executed_count = 0
        self.profit_total = 0.0
        
    async def start(self, symbols: List[str]):
        """启动套利策略"""
        self.running = True
        self.logger.info("启动WebSocket跨所三角套利策略")
        
        # 启动所有交易所的WebSocket连接
        tasks = []
        for exchange in self.exchanges.values():
            task = asyncio.create_task(exchange.connect_orderbook(symbols))
            tasks.append(task)
            
        # 启动套利检测任务
        arbitrage_task = asyncio.create_task(self._arbitrage_loop())
        tasks.append(arbitrage_task)
        
        # 启动订单执行任务
        execution_task = asyncio.create_task(self._execution_loop())
        tasks.append(execution_task)
        
        await asyncio.gather(*tasks)
        
    async def _arbitrage_loop(self):
        """套利检测循环"""
        while self.running:
            try:
                # 检查三角套利机会
                await self._check_triangular_opportunities()
                await asyncio.sleep(0.01)  # 10ms检测间隔
            except Exception as e:
                self.logger.error(f"套利检测错误: {e}")
                await asyncio.sleep(0.1)
                
    async def _check_triangular_opportunities(self):
        """检查三角套利机会"""
        # 常见的三角套利组合
        triangular_pairs = [
            ('USDT', 'BTC', 'ETH'),
            ('USDT', 'BTC', 'TRX'),
            ('USDT', 'ETH', 'TRX'),
        ]
        
        for base, quote, target in triangular_pairs:
            await self._check_cross_exchange_triangular(base, quote, target)
            
    async def _check_cross_exchange_triangular(self, X: str, Y: str, Z: str):
        """检查跨所三角套利"""
        # 构建交易对
        symbol_A = f"{Y}{X}"  # 如 BTCUSDT
        symbol_B = f"{Y}{Z}"  # 如 BTCETH  
        symbol_C = f"{Z}{X}"  # 如 ETHUSDT
        
        # 获取所有交易所的订单簿数据
        orderbooks = {}
        for ex_name, exchange in self.exchanges.items():
            orderbooks[ex_name] = {
                'A': exchange.get_orderbook(symbol_A),
                'B': exchange.get_orderbook(symbol_B),
                'C': exchange.get_orderbook(symbol_C)
            }
            
        # 检查数据完整性
        for ex_name, books in orderbooks.items():
            if not all(books.values()):
                return
                
        # 计算跨所套利机会
        await self._calculate_cross_arbitrage(X, Y, Z, orderbooks)
        
    async def _calculate_cross_arbitrage(self, X: str, Y: str, Z: str, orderbooks: Dict):
        """计算跨所套利机会"""
        exchanges = list(self.exchanges.keys())
        
        # 检查所有可能的跨所组合
        for i, ex1 in enumerate(exchanges):
            for j, ex2 in enumerate(exchanges):
                if i != j:
                    await self._check_cross_pair(X, Y, Z, ex1, ex2, orderbooks)
                    
    async def _check_cross_pair(self, X: str, Y: str, Z: str, ex1: str, ex2: str, orderbooks: Dict):
        """检查特定交易所对的套利机会"""
        books1 = orderbooks[ex1]
        books2 = orderbooks[ex2]
        
        # 模式112: ex1买A卖B, ex2卖C
        if books1['A'] and books1['B'] and books2['C']:
            A_ask = books1['A'].asks[0][0]
            B_bid = books1['B'].bids[0][0]
            C_bid = books2['C'].bids[0][0]
            
            ratio = (C_bid * B_bid) / A_ask
            if ratio > 1.002:  # 0.2%以上的套利空间
                signal = ArbitrageSignal(
                    mode="112",
                    base_currency=X,
                    quote_currency=Y,
                    target_currency=Z,
                    profit_ratio=ratio,
                    estimated_profit=0,  # 需要根据实际资金计算
                    orders=[
                        {"exchange": ex1, "symbol": f"{Y}{X}", "side": "buy", "type": "limit", "price": A_ask},
                        {"exchange": ex1, "symbol": f"{Y}{Z}", "side": "sell", "type": "limit", "price": B_bid},
                        {"exchange": ex2, "symbol": f"{Z}{X}", "side": "sell", "type": "limit", "price": C_bid}
                    ]
                )
                
                self.signal_queue.put(signal)
                self.signal_count += 1
                self.logger.info(f"发现套利机会 {ex1}-{ex2} 模式112: {ratio:.6f}")
                
    async def _execution_loop(self):
        """订单执行循环"""
        while self.running:
            try:
                if not self.signal_queue.empty():
                    signal = self.signal_queue.get()
                    await self._execute_arbitrage(signal)
                await asyncio.sleep(0.001)  # 1ms检查间隔
            except Exception as e:
                self.logger.error(f"订单执行错误: {e}")
                
    async def _execute_arbitrage(self, signal: ArbitrageSignal):
        """执行套利订单"""
        self.logger.info(f"执行套利信号: {signal.mode} {signal.profit_ratio:.6f}")
        
        # 并发执行所有订单
        tasks = []
        for order in signal.orders:
            exchange = self.exchanges[order["exchange"]]
            task = asyncio.create_task(
                exchange.place_order(
                    symbol=order["symbol"],
                    side=order["side"],
                    order_type=order["type"],
                    amount=100,  # 需要根据实际情况计算
                    price=order.get("price")
                )
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 检查执行结果
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        if success_count == len(results):
            self.executed_count += 1
            self.logger.info(f"套利订单执行成功: {signal.mode}")
        else:
            self.logger.error(f"套利订单执行失败: {results}")
            
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "signal_count": self.signal_count,
            "executed_count": self.executed_count,
            "profit_total": self.profit_total,
            "success_rate": self.executed_count / max(self.signal_count, 1)
        }

# 使用示例
async def main():
    """主函数示例"""
    # 配置交易所
    binance = BinanceWebSocket(
        api_key="your_binance_api_key",
        secret_key="your_binance_secret_key"
    )
    
    bitget = BitgetWebSocket(
        api_key="your_bitget_api_key",
        secret_key="your_bitget_secret_key",
        passphrase="your_bitget_passphrase"
    )
    
    gateio = GateioWebSocket(
        api_key="your_gateio_api_key",
        secret_key="your_gateio_secret_key"
    )
    
    # 创建套利策略
    arbitrage = WebSocketTriangularArbitrage(
        exchanges=[binance, bitget, gateio],
        ratio=0.8
    )
    
    # 监控的交易对
    symbols = [
        "BTCUSDT", "ETHUSDT", "TRXUSDT",
        "BTCETH", "TRXBTC", "TRXETH"
    ]
    
    # 启动策略
    await arbitrage.start(symbols)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
