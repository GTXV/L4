# 跨所三角套利策略
# 作者：千千量化
# 说明：从BanZhuanKing.py中提取的跨所三角套利策略，包含同市三角、跨市三角、跨市双边套利

import ccxt
import time
import logging
import threading
from typing import Tuple, List, Dict, Any, Optional

class MyThread(threading.Thread):
    """多线程调用类，用于并发询价和下单"""
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        """返回多线程运行结果"""
        try:
            return self.result
        except Exception as e:
            return e

class CrossExchangeTriangularArbitrage:
    """跨所三角套利策略类"""
    
    def __init__(self, exchange1, exchange2, logger=None, ratio=0.5, fee_ratio_1=1, fee_ratio_2=1):
        """
        初始化跨所三角套利策略
        
        Args:
            exchange1: 第一个交易所实例
            exchange2: 第二个交易所实例
            logger: 日志记录器
            ratio: 吃单比例，默认0.5
            fee_ratio_1: 交易所1的手续费比率
            fee_ratio_2: 交易所2的手续费比率
        """
        self.exchange_1 = exchange1
        self.exchange_2 = exchange2
        self.ratio = ratio
        self.fee_ratio_1 = fee_ratio_1
        self.fee_ratio_2 = fee_ratio_2
        
        # 设置日志
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger(__name__)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)
        
        # 获取交易所信息
        self.markets_1 = self.exchange_1.fetch_markets()
        self.markets_2 = self.exchange_2.fetch_markets()
        self.fee_1 = self.exchange_1.fees
        self.fee_2 = self.exchange_2.fees
        
        # 统计信息
        self.signal_num = 0
        self.open_num = 0
        self.open_fail = 0
        self.win = {'BTC': 0, 'USDT': 0, 'ETH': 0}
    
    def get_balance(self, X: str, Y: str, Z: str) -> Tuple[float, float, float, float, float, float]:
        """获取两个交易所的余额"""
        balance1 = self.exchange_1.fetch_balance()
        balance2 = self.exchange_2.fetch_balance()
        
        cur_size_11 = balance1[X]['free'] if X in balance1 else 0
        cur_size_12 = balance1[Y]['free'] if Y in balance1 else 0
        cur_size_13 = balance1[Z]['free'] if Z in balance1 else 0
        cur_size_21 = balance2[X]['free'] if X in balance2 else 0
        cur_size_22 = balance2[Y]['free'] if Y in balance2 else 0
        cur_size_23 = balance2[Z]['free'] if Z in balance2 else 0
        
        return cur_size_11, cur_size_12, cur_size_13, cur_size_21, cur_size_22, cur_size_23
    
    def get_limit(self, symbol: str, mode: int) -> Tuple[float, float, float]:
        """获取交易对的限制信息"""
        base = symbol.split('/')[0]
        quote = symbol.split('/')[1]
        
        if mode == 1:
            markets = self.markets_1
            exchange_name = self.exchange_1.id
        else:
            markets = self.markets_2
            exchange_name = self.exchange_2.id
        
        min_amt = 0
        min_price = 0
        min_cost = 0
        
        for market in markets:
            if market['symbol'] == symbol:
                min_amt = market['limits']['amount']['min'] if 'amount' in market['limits'] else 0
                min_price = market['limits']['price']['min'] if 'price' in market['limits'] else 0
                min_cost = market['limits']['cost']['min'] if 'cost' in market['limits'] else 0
                break
        
        # 特殊交易所的最小交易量调整
        if exchange_name == 'okex3':
            min_amt = min_cost
        elif exchange_name == 'fcoin':
            if base == 'TRX': min_amt = 50
            elif base == 'XLM': min_amt = 5
            elif base == 'ETH' and quote == 'BTC': min_amt = 0.001
            elif base == 'ETC' and quote == 'BTC': min_amt = 0.001
            elif base == 'LTC': min_amt = 0.01
            elif base == 'EOS': min_amt = 0.1
            elif base == 'XRP': min_amt = 1
        elif exchange_name == 'gateio':
            if base == 'ETH': min_amt = 0.005
            elif base == 'TRX': min_amt = 60
            elif base == 'XLM': min_amt = 15
            elif base == 'EOS': min_amt = 0.3
            elif base == 'ETC': min_amt = 0.6
        
        return float(min_amt) * 1.05, float(min_price), float(min_cost)
    
    def get_order_book(self, symbol: str, exchange: int) -> Tuple[float, float, float, float]:
        """获取订单簿信息"""
        if exchange == 1:
            result = self.exchange_1.fetch_order_book(symbol=symbol, limit=None)
        else:
            result = self.exchange_2.fetch_order_book(symbol=symbol, limit=None)
        
        return result['bids'][0][0], result['bids'][0][1], result['asks'][0][0], result['asks'][0][1]
    
    def create_order(self, symbol: str, exchange: int, type: str, side: str, amount: float, price: float) -> Dict:
        """创建订单"""
        try:
            if exchange == 1:
                result = self.exchange_1.create_order(symbol, type, side, amount, price)
            else:
                result = self.exchange_2.create_order(symbol, type, side, amount, price)
            return result
        except Exception as e:
            self.log.error(f"下单出错: {e}")
            self.open_fail += 1
            return {}
    
    def check_triangular_arbitrage(self, X: str, Y: str, Z: str) -> bool:
        """
        检查跨所三角套利机会
        
        Args:
            X: 基础货币 (如 USDT)
            Y: 中间货币 (如 TRX)
            Z: 目标货币 (如 BTC/ETH)
        
        Returns:
            bool: 是否发现并执行了套利机会
        """
        time.sleep(0.2)
        self.log.debug('--------------------')
        self.log.debug(f'当前检测三角对：{X} {Y} {Z}')
        
        # 获取余额
        cur_size_11, cur_size_12, cur_size_13, cur_size_21, cur_size_22, cur_size_23 = self.get_balance(X, Y, Z)
        self.log.debug(f'交易所1当前币量：{cur_size_11:.6f} {cur_size_12:.6f} {cur_size_13:.6f}')
        self.log.debug(f'交易所2当前币量：{cur_size_21:.6f} {cur_size_22:.6f} {cur_size_23:.6f}')
        
        # 构建交易对
        symbol_A = f'{Y}/{X}'  # 如 TRX/USDT
        symbol_B = f'{Y}/{Z}'  # 如 TRX/BTC
        symbol_C = f'{Z}/{X}'  # 如 BTC/USDT
        
        # 获取交易限制
        min_amt_A1, min_price_A1, min_cost_A1 = self.get_limit(symbol_A, 1)
        min_amt_B1, min_price_B1, min_cost_B1 = self.get_limit(symbol_B, 1)
        min_amt_C1, min_price_C1, min_cost_C1 = self.get_limit(symbol_C, 1)
        min_amt_A2, min_price_A2, min_cost_A2 = self.get_limit(symbol_A, 2)
        min_amt_B2, min_price_B2, min_cost_B2 = self.get_limit(symbol_B, 2)
        min_amt_C2, min_price_C2, min_cost_C2 = self.get_limit(symbol_C, 2)
        
        # 多线程获取订单簿
        threads = []
        data = []
        
        threads.append(MyThread(self.get_order_book, args=(symbol_A, 1)))
        threads.append(MyThread(self.get_order_book, args=(symbol_B, 1)))
        threads.append(MyThread(self.get_order_book, args=(symbol_C, 1)))
        threads.append(MyThread(self.get_order_book, args=(symbol_A, 2)))
        threads.append(MyThread(self.get_order_book, args=(symbol_B, 2)))
        threads.append(MyThread(self.get_order_book, args=(symbol_C, 2)))
        
        begin = time.time()
        for t in threads:
            t.setDaemon(True)
            t.start()
        
        for t in threads:
            t.join()
            data.append(t.get_result())
        
        end = time.time()
        delay = float((end - begin) // 0.001)
        
        # 解析订单簿数据
        A_bestbid_1, A_bestbid_size_1, A_bestask_1, A_bestask_size_1 = data[0]
        B_bestbid_1, B_bestbid_size_1, B_bestask_1, B_bestask_size_1 = data[1]
        C_bestbid_1, C_bestbid_size_1, C_bestask_1, C_bestask_size_1 = data[2]
        A_bestbid_2, A_bestbid_size_2, A_bestask_2, A_bestask_size_2 = data[3]
        B_bestbid_2, B_bestbid_size_2, B_bestask_2, B_bestask_size_2 = data[4]
        C_bestbid_2, C_bestbid_size_2, C_bestask_2, C_bestask_size_2 = data[5]
        
        # 计算套利机会
        # 跨市三角套利 - 4种模式
        # 模式112: A1 B1 C2 (交易所1买A卖B，交易所2卖C)
        Surplus_112 = (C_bestbid_2 * B_bestbid_1 / A_bestask_1 - 
                      2 * self.fee_1['trading']['maker'] * self.fee_ratio_1 - 
                      self.fee_2['trading']['maker'] * self.fee_ratio_2)
        
        # 模式211: C2 B1 A1 (交易所2买C，交易所1买B卖A)
        Deficit_211 = (A_bestbid_1 / B_bestask_1 / C_bestask_2 - 
                      2 * self.fee_1['trading']['maker'] * self.fee_ratio_1 - 
                      self.fee_2['trading']['maker'] * self.fee_ratio_2)
        
        # 模式221: A2 B2 C1 (交易所2买A卖B，交易所1卖C)
        Surplus_221 = (C_bestbid_1 * B_bestbid_2 / A_bestask_2 - 
                      2 * self.fee_2['trading']['maker'] * self.fee_ratio_2 - 
                      self.fee_1['trading']['maker'] * self.fee_ratio_1)
        
        # 模式122: C1 B2 A2 (交易所1买C，交易所2买B卖A)
        Deficit_122 = (A_bestbid_2 / B_bestask_2 / C_bestask_1 - 
                      2 * self.fee_2['trading']['maker'] * self.fee_ratio_2 - 
                      self.fee_1['trading']['maker'] * self.fee_ratio_1)
        
        # 检查套利机会并执行
        if self._execute_cross_triangular_112(Surplus_112, X, Y, Z, symbol_A, symbol_B, symbol_C,
                                             cur_size_11, cur_size_12, cur_size_23,
                                             A_bestask_1, A_bestask_size_1, B_bestbid_1, B_bestbid_size_1,
                                             C_bestbid_2, C_bestbid_size_2,
                                             min_amt_A1, min_amt_B1, min_amt_C2, delay):
            return True
        
        if self._execute_cross_triangular_211(Deficit_211, X, Y, Z, symbol_A, symbol_B, symbol_C,
                                             cur_size_21, cur_size_13, cur_size_12,
                                             C_bestask_2, C_bestask_size_2, B_bestask_1, B_bestask_size_1,
                                             A_bestbid_1, A_bestbid_size_1,
                                             min_amt_C2, min_amt_B1, min_amt_A1, delay):
            return True
        
        if self._execute_cross_triangular_221(Surplus_221, X, Y, Z, symbol_A, symbol_B, symbol_C,
                                             cur_size_21, cur_size_22, cur_size_13,
                                             A_bestask_2, A_bestask_size_2, B_bestbid_2, B_bestbid_size_2,
                                             C_bestbid_1, C_bestbid_size_1,
                                             min_amt_A2, min_amt_B2, min_amt_C1, delay):
            return True
        
        if self._execute_cross_triangular_122(Deficit_122, X, Y, Z, symbol_A, symbol_B, symbol_C,
                                             cur_size_11, cur_size_23, cur_size_22,
                                             C_bestask_1, C_bestask_size_1, B_bestask_2, B_bestask_size_2,
                                             A_bestbid_2, A_bestbid_size_2,
                                             min_amt_C1, min_amt_B2, min_amt_A2, delay):
            return True
        
        return False
    
    def _execute_cross_triangular_112(self, surplus, X, Y, Z, symbol_A, symbol_B, symbol_C,
                                     cur_size_11, cur_size_12, cur_size_23,
                                     A_bestask_1, A_bestask_size_1, B_bestbid_1, B_bestbid_size_1,
                                     C_bestbid_2, C_bestbid_size_2,
                                     min_amt_A1, min_amt_B1, min_amt_C2, delay):
        """执行跨市三角套利模式112: A1买 B1卖 C2卖"""
        if surplus > 1 and surplus < 1.01:
            size_1 = min(cur_size_11, cur_size_12 * A_bestask_1, cur_size_23 * C_bestbid_2,
                        A_bestask_size_1 / A_bestask_1, B_bestbid_size_1 * A_bestask_1,
                        C_bestbid_size_2 * C_bestbid_2) * self.ratio
            size_2 = size_1 / A_bestask_1
            size_3 = size_2 * B_bestbid_1
            
            amt_A = float(self.exchange_1.amount_to_precision(symbol_A, size_2))
            amt_B = float(self.exchange_1.amount_to_precision(symbol_B, size_2))
            amt_C = float(self.exchange_2.amount_to_precision(symbol_C, size_3))
            price_A = float(self.exchange_1.price_to_precision(symbol_A, A_bestask_1))
            price_B = float(self.exchange_1.price_to_precision(symbol_B, B_bestbid_1))
            price_C = float(self.exchange_2.price_to_precision(symbol_C, C_bestbid_2))
            
            win = size_3 * C_bestbid_2 - size_1
            
            if (size_2 > min_amt_A1 and size_2 > min_amt_B1 and size_3 > min_amt_C2 and
                win > 0 and delay <= 95 and
                amt_A > 0 and amt_B > 0 and amt_C > 0 and
                price_A > 0 and price_B > 0 and price_C > 0):
                
                # 并发下单
                threads = []
                order_results = []
                
                threads.append(MyThread(self.create_order, args=(symbol_A, 1, 'limit', 'buy', size_2, A_bestask_1)))
                threads.append(MyThread(self.create_order, args=(symbol_B, 1, 'limit', 'sell', size_2, B_bestbid_1)))
                threads.append(MyThread(self.create_order, args=(symbol_C, 2, 'limit', 'sell', size_3, C_bestbid_2)))
                
                begin = time.time()
                for t in threads:
                    t.setDaemon(True)
                    t.start()
                
                for t in threads:
                    t.join()
                    order_results.append(t.get_result())
                
                end = time.time()
                delay_ = float((end - begin) // 0.001)
                
                self.log.info("发现跨市112正循环信号！")
                self.log.info(f"预估交易数量: {size_1:.6f} {size_2:.6f} {size_3:.6f}")
                self.log.info(f'投入: {size_1:.6f} 产出: {size_3 * C_bestbid_2:.6f} 盈利: {win:.6f}')
                self.log.info(f"价差比率: {surplus:.6f}")
                self.log.info(f'询价延迟: {delay:.1f}ms 下单延迟: {delay_:.1f}ms')
                
                # 检查订单结果
                signal = True
                for result in order_results:
                    if 'id' not in result:
                        signal = False
                        break
                
                if signal:
                    self.open_num += 1
                    self.win[X] += win
                    return True
        
        return False
    
    def _execute_cross_triangular_211(self, deficit, X, Y, Z, symbol_A, symbol_B, symbol_C,
                                     cur_size_21, cur_size_13, cur_size_12,
                                     C_bestask_2, C_bestask_size_2, B_bestask_1, B_bestask_size_1,
                                     A_bestbid_1, A_bestbid_size_1,
                                     min_amt_C2, min_amt_B1, min_amt_A1, delay):
        """执行跨市三角套利模式211: C2买 B1买 A1卖"""
        if deficit > 1 and deficit < 1.01:
            size_1 = min(cur_size_21, cur_size_13 * C_bestask_2, cur_size_12 * A_bestbid_1,
                        C_bestask_size_2 * C_bestask_2, B_bestask_size_1 * B_bestask_1 * C_bestask_2,
                        A_bestbid_1 * A_bestbid_size_1) * self.ratio
            size_2 = size_1 / C_bestask_2
            size_3 = size_2 / B_bestask_1
            
            amt_A = float(self.exchange_1.amount_to_precision(symbol_A, size_3))
            amt_B = float(self.exchange_1.amount_to_precision(symbol_B, size_3))
            amt_C = float(self.exchange_2.amount_to_precision(symbol_C, size_2))
            price_A = float(self.exchange_1.price_to_precision(symbol_A, A_bestbid_1))
            price_B = float(self.exchange_1.price_to_precision(symbol_B, B_bestask_1))
            price_C = float(self.exchange_2.price_to_precision(symbol_C, C_bestask_2))
            
            win = size_3 * A_bestbid_1 - size_1
            
            if (size_3 > min_amt_A1 and size_3 > min_amt_B1 and size_2 > min_amt_C2 and
                win > 0 and delay <= 95 and
                amt_A > 0 and amt_B > 0 and amt_C > 0 and
                price_A > 0 and price_B > 0 and price_C > 0):
                
                # 并发下单
                threads = []
                order_results = []
                
                threads.append(MyThread(self.create_order, args=(symbol_C, 2, 'limit', 'buy', size_2, C_bestask_2)))
                threads.append(MyThread(self.create_order, args=(symbol_B, 1, 'limit', 'buy', size_3, B_bestask_1)))
                threads.append(MyThread(self.create_order, args=(symbol_A, 1, 'limit', 'sell', size_3, A_bestbid_1)))
                
                begin = time.time()
                for t in threads:
                    t.setDaemon(True)
                    t.start()
                
                for t in threads:
                    t.join()
                    order_results.append(t.get_result())
                
                end = time.time()
                delay_ = float((end - begin) // 0.001)
                
                self.log.info("发现跨市211逆循环信号！")
                self.log.info(f"预估交易数量: {size_1:.6f} {size_2:.6f} {size_3:.6f}")
                self.log.info(f'投入: {size_1:.6f} 产出: {size_3 * A_bestbid_1:.6f} 盈利: {win:.6f}')
                self.log.info(f"价差比率: {deficit:.6f}")
                self.log.info(f'询价延迟: {delay:.1f}ms 下单延迟: {delay_:.1f}ms')
                
                # 检查订单结果
                signal = True
                for result in order_results:
                    if 'id' not in result:
                        signal = False
                        break
                
                if signal:
                    self.open_num += 1
                    self.win[X] += win
                    return True
        
        return False
    
    def _execute_cross_triangular_221(self, surplus, X, Y, Z, symbol_A, symbol_B, symbol_C,
                                     cur_size_21, cur_size_22, cur_size_13,
                                     A_bestask_2, A_bestask_size_2, B_bestbid_2, B_bestbid_size_2,
                                     C_bestbid_1, C_bestbid_size_1,
                                     min_amt_A2, min_amt_B2, min_amt_C1, delay):
        """执行跨市三角套利模式221: A2买 B2卖 C1卖"""
        if surplus > 1 and surplus < 1.01:
            size_1 = min(cur_size_21, cur_size_22 * A_bestask_2, cur_size_13 * C_bestbid_1,
                        A_bestask_size_2 / A_bestask_2, B_bestbid_size_2 * A_bestask_2,
                        C_bestbid_size_1 * C_bestbid_1) * self.ratio
            size_2 = size_1 / A_bestask_2
            size_3 = size_2 * B_bestbid_2
            
            amt_A = float(self.exchange_2.amount_to_precision(symbol_A, size_2))
            amt_B = float(self.exchange_2.amount_to_precision(symbol_B, size_2))
            amt_C = float(self.exchange_1.amount_to_precision(symbol_C, size_3))
            price_A = float(self.exchange_2.price_to_precision(symbol_A, A_bestask_2))
            price_B = float(self.exchange_2.price_to_precision(symbol_B, B_bestbid_2))
            price_C = float(self.exchange_1.price_to_precision(symbol_C, C_bestbid_1))
            
            win = size_3 * C_bestbid_1 - size_1
            
            if (size_2 > min_amt_A2 and size_2 > min_amt_B2 and size_3 > min_amt_C1 and
                win > 0 and delay <= 95 and
                amt_A > 0 and amt_B > 0 and amt_C > 0 and
                price_A > 0 and price_B > 0 and price_C > 0):
                
                # 并发下单
                threads = []
                order_results = []
                
                threads.append(MyThread(self.create_order, args=(symbol_A, 2, 'limit', 'buy', size_2, A_bestask_2)))
                threads.append(MyThread(self.create_order, args=(symbol_B, 2, 'limit', 'sell', size_2, B_bestbid_2)))
                threads.append(MyThread(self.create_order, args=(symbol_C, 1, 'limit', 'sell', size_3, C_bestbid_1)))
                
                begin = time.time()
                for t in threads:
                    t.setDaemon(True)
                    t.start()
                
                for t in threads:
                    t.join()
                    order_results.append(t.get_result())
                
                end = time.time()
                delay_ = float((end - begin) // 0.001)
                
                self.log.info("发现跨市221正循环信号！")
                self.log.info(f"预估交易数量: {size_1:.6f} {size_2:.6f} {size_3:.6f}")
                self.log.info(f'投入: {size_1:.6f} 产出: {size_3 * C_bestbid_1:.6f} 盈利: {win:.6f}')
                self.log.info(f"价差比率: {surplus:.6f}")
                self.log.info(f'询价延迟: {delay:.1f}ms 下单延迟: {delay_:.1f}ms')
                
                # 检查订单结果
                signal = True
                for result in order_results:
                    if 'id' not in result:
                        signal = False
                        break
                
                if signal:
                    self.open_num += 1
                    self.win[X] += win
                    return True
        
        return False
    
    def _execute_cross_triangular_122(self, deficit, X, Y, Z, symbol_A, symbol_B, symbol_C,
                                     cur_size_11, cur_size_23, cur_size_22,
                                     C_bestask_1, C_bestask_size_1, B_bestask_2, B_bestask_size_2,
                                     A_bestbid_2, A_bestbid_size_2,
                                     min_amt_C1, min_amt_B2, min_amt_A2, delay):
        """执行跨市三角套利模式122: C1买 B2买 A2卖"""
        if deficit > 1 and deficit < 1.01:
            size_1 = min(cur_size_11, cur_size_23 * C_bestask_1, cur_size_22 * A_bestbid_2,
                        C_bestask_size_1 * C_bestask_1, B_bestask_size_2 * B_bestask_2 * C_bestask_1,
                        A_bestbid_2 * A_bestbid_size_2) * self.ratio
            size_2 = size_1 / C_bestask_1
            size_3 = size_2 / B_bestask_2
            
            amt_A = float(self.exchange_2.amount_to_precision(symbol_A, size_3))
            amt_B = float(self.exchange_2.amount_to_precision(symbol_B, size_3))
            amt_C = float(self.exchange_1.amount_to_precision(symbol_C, size_2))
            price_A = float(self.exchange_2.price_to_precision(symbol_A, A_bestbid_2))
            price_B = float(self.exchange_2.price_to_precision(symbol_B, B_bestask_2))
            price_C = float(self.exchange_1.price_to_precision(symbol_C, C_bestask_1))
            
            win = size_3 * A_bestbid_2 - size_1
            
            if (size_3 > min_amt_A2 and size_3 > min_amt_B2 and size_2 > min_amt_C1 and
                win > 0 and delay <= 95 and
                amt_A > 0 and amt_B > 0 and amt_C > 0 and
                price_A > 0 and price_B > 0 and price_C > 0):
                
                # 并发下单
                threads = []
                order_results = []
                
                threads.append(MyThread(self.create_order, args=(symbol_C, 1, 'limit', 'buy', size_2, C_bestask_1)))
                threads.append(MyThread(self.create_order, args=(symbol_B, 2, 'limit', 'buy', size_3, B_bestask_2)))
                threads.append(MyThread(self.create_order, args=(symbol_A, 2, 'limit', 'sell', size_3, A_bestbid_2)))
                
                begin = time.time()
                for t in threads:
                    t.setDaemon(True)
                    t.start()
                
                for t in threads:
                    t.join()
                    order_results.append(t.get_result())
                
                end = time.time()
                delay_ = float((end - begin) // 0.001)
                
                self.log.info("发现跨市122逆循环信号！")
                self.log.info(f"预估交易数量: {size_1:.6f} {size_2:.6f} {size_3:.6f}")
                self.log.info(f'投入: {size_1:.6f} 产出: {size_3 * A_bestbid_2:.6f} 盈利: {win:.6f}')
                self.log.info(f"价差比率: {deficit:.6f}")
                self.log.info(f'询价延迟: {delay:.1f}ms 下单延迟: {delay_:.1f}ms')
                
                # 检查订单结果
                signal = True
                for result in order_results:
                    if 'id' not in result:
                        signal = False
                        break
                
                if signal:
                    self.open_num += 1
                    self.win[X] += win
                    return True
        
        return False

# 使用示例
def example_usage():
    """
    使用示例
    """
    # 配置交易所
    exchange1 = ccxt.binance({
        'apiKey': 'your_api_key_1',
        'secret': 'your_secret_1',
        'sandbox': True,  # 使用测试环境
        'enableRateLimit': True,
    })
    
    exchange2 = ccxt.okex({
        'apiKey': 'your_api_key_2',
        'secret': 'your_secret_2',
        'password': 'your_passphrase_2',
        'sandbox': True,  # 使用测试环境
        'enableRateLimit': True,
    })
    
    # 创建套利策略实例
    arbitrage = CrossExchangeTriangularArbitrage(
        exchange1=exchange1,
        exchange2=exchange2,
        ratio=0.8,  # 吃单比例
        fee_ratio_1=0.5,  # 交易所1手续费折扣
        fee_ratio_2=1.0   # 交易所2手续费折扣
    )
    
    # 检查套利机会
    # 常见的三角套利组合
    triangular_pairs = [
        ('USDT', 'TRX', 'BTC'),
        ('USDT', 'XLM', 'BTC'),
        ('USDT', 'EOS', 'BTC'),
        ('USDT', 'TRX', 'ETH'),
        ('USDT', 'XLM', 'ETH'),
        ('BTC', 'TRX', 'ETH'),
        ('BTC', 'XLM', 'ETH'),
    ]
    
    while True:
        try:
            for X, Y, Z in triangular_pairs:
                if arbitrage.check_triangular_arbitrage(X, Y, Z):
                    print(f"成功执行套利: {X}-{Y}-{Z}")
                    break
            
            time.sleep(1)  # 避免过于频繁的请求
            
        except Exception as e:
            print(f"错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    # 运行示例（需要配置真实的API密钥）
    # example_usage()
    print("跨所三角套利策略已准备就绪")
    print("请配置您的交易所API密钥后使用 example_usage() 函数")