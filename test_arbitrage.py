#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨所三角套利策略测试脚本
"""

import sys
import logging
from cross_exchange_triangular_arbitrage import CrossExchangeTriangularArbitrage

# 模拟交易所类（用于测试）
class MockExchange:
    def __init__(self, name, fees=None):
        self.id = name
        self.fees = fees or {
            'trading': {
                'maker': 0.001,  # 0.1% 手续费
                'taker': 0.001
            }
        }
    
    def fetch_markets(self):
        """模拟获取市场信息"""
        return [
            {
                'symbol': 'TRX/USDT',
                'limits': {
                    'amount': {'min': 1.0},
                    'price': {'min': 0.00001},
                    'cost': {'min': 1.0}
                }
            },
            {
                'symbol': 'TRX/BTC',
                'limits': {
                    'amount': {'min': 1.0},
                    'price': {'min': 0.00000001},
                    'cost': {'min': 0.0001}
                }
            },
            {
                'symbol': 'BTC/USDT',
                'limits': {
                    'amount': {'min': 0.0001},
                    'price': {'min': 0.01},
                    'cost': {'min': 1.0}
                }
            }
        ]
    
    def fetch_balance(self):
        """模拟获取余额"""
        return {
            'USDT': {'free': 1000.0, 'used': 0.0, 'total': 1000.0},
            'TRX': {'free': 10000.0, 'used': 0.0, 'total': 10000.0},
            'BTC': {'free': 0.1, 'used': 0.0, 'total': 0.1}
        }
    
    def fetch_order_book(self, symbol, limit=None):
        """模拟获取订单簿"""
        order_books = {
            'TRX/USDT': {
                'bids': [[0.08, 1000], [0.079, 2000]],
                'asks': [[0.081, 1000], [0.082, 2000]]
            },
            'TRX/BTC': {
                'bids': [[0.000002, 1000], [0.0000019, 2000]],
                'asks': [[0.0000021, 1000], [0.0000022, 2000]]
            },
            'BTC/USDT': {
                'bids': [[40000, 0.1], [39999, 0.2]],
                'asks': [[40001, 0.1], [40002, 0.2]]
            }
        }
        return order_books.get(symbol, {'bids': [[0, 0]], 'asks': [[0, 0]]})
    
    def create_order(self, symbol, type, side, amount, price):
        """模拟创建订单"""
        return {
            'id': f'mock_order_{symbol}_{side}_{amount}_{price}',
            'symbol': symbol,
            'type': type,
            'side': side,
            'amount': amount,
            'price': price,
            'status': 'open'
        }
    
    def amount_to_precision(self, symbol, amount):
        """模拟精度转换"""
        return round(amount, 6)
    
    def price_to_precision(self, symbol, price):
        """模拟精度转换"""
        return round(price, 8)

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试跨所三角套利策略基本功能 ===")
    
    # 创建模拟交易所
    exchange1 = MockExchange('binance')
    exchange2 = MockExchange('okex')
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('test')
    
    # 创建套利策略实例
    arbitrage = CrossExchangeTriangularArbitrage(
        exchange1=exchange1,
        exchange2=exchange2,
        logger=logger,
        ratio=0.8,
        fee_ratio_1=0.5,
        fee_ratio_2=1.0
    )
    
    print(f"✓ 策略实例创建成功")
    print(f"✓ 交易所1: {arbitrage.exchange_1.id}")
    print(f"✓ 交易所2: {arbitrage.exchange_2.id}")
    print(f"✓ 吃单比例: {arbitrage.ratio}")
    print(f"✓ 手续费比率: {arbitrage.fee_ratio_1}, {arbitrage.fee_ratio_2}")
    
    # 测试余额获取
    print("\n=== 测试余额获取 ===")
    balance = arbitrage.get_balance('USDT', 'TRX', 'BTC')
    print(f"✓ 余额获取成功: {balance}")
    
    # 测试交易限制获取
    print("\n=== 测试交易限制获取 ===")
    limit_A1 = arbitrage.get_limit('TRX/USDT', 1)
    limit_B1 = arbitrage.get_limit('TRX/BTC', 1)
    limit_C1 = arbitrage.get_limit('BTC/USDT', 1)
    print(f"✓ 交易所1限制: TRX/USDT={limit_A1}, TRX/BTC={limit_B1}, BTC/USDT={limit_C1}")
    
    # 测试订单簿获取
    print("\n=== 测试订单簿获取 ===")
    orderbook_A1 = arbitrage.get_order_book('TRX/USDT', 1)
    orderbook_B1 = arbitrage.get_order_book('TRX/BTC', 1)
    orderbook_C1 = arbitrage.get_order_book('BTC/USDT', 1)
    print(f"✓ 订单簿获取成功:")
    print(f"  TRX/USDT: {orderbook_A1}")
    print(f"  TRX/BTC: {orderbook_B1}")
    print(f"  BTC/USDT: {orderbook_C1}")
    
    # 测试套利检查
    print("\n=== 测试套利机会检查 ===")
    try:
        result = arbitrage.check_triangular_arbitrage('USDT', 'TRX', 'BTC')
        print(f"✓ 套利检查完成，结果: {result}")
    except Exception as e:
        print(f"✗ 套利检查出错: {e}")
    
    # 显示统计信息
    print("\n=== 统计信息 ===")
    print(f"信号数量: {arbitrage.signal_num}")
    print(f"开仓数量: {arbitrage.open_num}")
    print(f"失败数量: {arbitrage.open_fail}")
    print(f"盈利情况: {arbitrage.win}")
    
    print("\n=== 测试完成 ===")

def test_profit_calculation():
    """测试盈利计算逻辑"""
    print("\n=== 测试盈利计算逻辑 ===")
    
    # 模拟价格数据
    A_ask = 0.081    # TRX/USDT 卖一价
    B_bid = 0.000002 # TRX/BTC 买一价
    C_bid = 40000    # BTC/USDT 买一价
    
    # 计算理论套利比率（模式112）
    # 路径：USDT -> TRX -> BTC -> USDT
    theoretical_ratio = (C_bid * B_bid) / A_ask
    print(f"理论套利比率: {theoretical_ratio:.6f}")
    
    # 考虑手续费后的实际比率
    fee_cost = 3 * 0.001 * 0.5  # 3次交易，0.1%手续费，5折优惠
    actual_ratio = theoretical_ratio - fee_cost
    print(f"扣除手续费后比率: {actual_ratio:.6f}")
    
    if actual_ratio > 1:
        profit_rate = (actual_ratio - 1) * 100
        print(f"✓ 存在套利机会，预期收益率: {profit_rate:.4f}%")
    else:
        print("✗ 不存在套利机会")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_profit_calculation()
        print("\n🎉 所有测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
