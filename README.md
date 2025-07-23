# 跨所三角套利策略

从原始的 `BanZhuanKing.py` 文件中提取的跨所三角套利策略，重新组织为独立的类和函数。

## 功能特点

- **跨市三角套利**: 在两个不同交易所之间进行三角套利，包含4种模式
- **多线程并发**: 使用多线程进行订单簿查询和下单，提高执行效率
- **风险控制**: 内置延迟检测、最小交易量检查、盈利预估等风险控制机制
- **灵活配置**: 支持自定义吃单比例、手续费折扣等参数

## 套利模式

### 跨市三角套利的4种模式

假设有三个币种：X（基础币如USDT）、Y（中间币如TRX）、Z（目标币如BTC）
对应的交易对：A=Y/X, B=Y/Z, C=Z/X

1. **模式112**: 交易所1买A卖B，交易所2卖C
   - 路径：X → Y → Z → X
   - 操作：交易所1买入Y/X，卖出Y/Z；交易所2卖出Z/X

2. **模式211**: 交易所2买C，交易所1买B卖A  
   - 路径：X → Z → Y → X
   - 操作：交易所2买入Z/X；交易所1买入Y/Z，卖出Y/X

3. **模式221**: 交易所2买A卖B，交易所1卖C
   - 路径：X → Y → Z → X
   - 操作：交易所2买入Y/X，卖出Y/Z；交易所1卖出Z/X

4. **模式122**: 交易所1买C，交易所2买B卖A
   - 路径：X → Z → Y → X  
   - 操作：交易所1买入Z/X；交易所2买入Y/Z，卖出Y/X

## 使用方法

### 1. 安装依赖

```bash
pip install ccxt
```

### 2. 配置交易所

```python
import ccxt
from cross_exchange_triangular_arbitrage import CrossExchangeTriangularArbitrage

# 配置第一个交易所
exchange1 = ccxt.binance({
    'apiKey': 'your_api_key_1',
    'secret': 'your_secret_1',
    'sandbox': True,  # 测试环境
    'enableRateLimit': True,
})

# 配置第二个交易所
exchange2 = ccxt.okex({
    'apiKey': 'your_api_key_2',
    'secret': 'your_secret_2',
    'password': 'your_passphrase_2',
    'sandbox': True,  # 测试环境
    'enableRateLimit': True,
})
```

### 3. 创建套利策略实例

```python
arbitrage = CrossExchangeTriangularArbitrage(
    exchange1=exchange1,
    exchange2=exchange2,
    ratio=0.8,        # 吃单比例（0.5-0.8推荐）
    fee_ratio_1=0.5,  # 交易所1手续费折扣
    fee_ratio_2=1.0   # 交易所2手续费折扣
)
```

### 4. 执行套利检查

```python
# 单次检查
result = arbitrage.check_triangular_arbitrage('USDT', 'TRX', 'BTC')

# 持续监控
triangular_pairs = [
    ('USDT', 'TRX', 'BTC'),
    ('USDT', 'XLM', 'BTC'),
    ('USDT', 'EOS', 'BTC'),
    ('USDT', 'TRX', 'ETH'),
    ('BTC', 'TRX', 'ETH'),
]

while True:
    for X, Y, Z in triangular_pairs:
        if arbitrage.check_triangular_arbitrage(X, Y, Z):
            print(f"成功执行套利: {X}-{Y}-{Z}")
            break
    time.sleep(1)
```

## 参数说明

- `ratio`: 吃单比例，建议0.5-0.8，过高可能导致滑点，过低可能导致开仓率低
- `fee_ratio_1/2`: 手续费折扣比例，如果有点卡折扣填相应数值（如5折填0.5）
- `delay`: 询价延迟阈值，超过95ms的机会将被过滤

## 风险提示

1. **市场风险**: 套利机会稍纵即逝，价格波动可能导致损失
2. **技术风险**: 网络延迟、API限制等可能影响执行效果
3. **资金风险**: 需要在两个交易所都有足够的资金余额
4. **合规风险**: 请确保在合规的交易所进行交易

## 注意事项

1. 建议先在测试环境中运行和验证策略
2. 根据实际情况调整参数，特别是吃单比例和手续费折扣
3. 监控策略运行状态，及时处理异常情况
4. 定期检查和更新交易所的API配置

## 支持的交易所

理论上支持所有CCXT库支持的交易所，包括但不限于：
- Binance
- OKEx
- Huobi
- Gate.io
- FCoin
- 等等

## 技术支持

如有问题，请参考原始代码或联系开发者。

---

**免责声明**: 本策略仅供学习和研究使用，实际交易请谨慎评估风险。
