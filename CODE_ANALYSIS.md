# 全天候量化策略 — 逐层代码分析

> 项目：全天候策略（All-Weather Strategy）
> 平台：聚宽（JoinQuant）
> 入口文件：`all_weather.py`（634 行，单文件）
> 运行方式：粘贴到聚宽研究/回测平台编译运行，本地无法直接运行

---

## 第 0 层：仓库地图

### 一句话作用

单文件量化策略项目，运行于聚宽平台，通过 ETF 组合实现风险平价 + 动量增强的全天候资产配置。无实盘代码，回测与模拟交易共用同一套代码。

### 目录树

```
all-weather-strategy/
├── LICENSE              # MIT 开源协议
├── README.md            # 项目说明 + 实盘跟踪数据 + 联系方式
└── all_weather.py       # 策略唯一源码文件（634 行）
```

### 模块职责表

| 目录/文件          | 作用         | 关键内容                                             |
| ------------------ | ------------ | ---------------------------------------------------- |
| `all_weather.py` | 策略完整源码 | 入口、配置、数据、信号、组合、风控、执行全部在此文件 |
| `README.md`      | 项目说明     | 策略理念、实盘表现、使用方法、免责声明               |
| `LICENSE`        | 开源协议     | MIT                                                  |

### AI 自问自答

- **这个项目是回测框架、实盘框架，还是两者共用？**纯回测/模拟交易框架。代码基于聚宽 `jqdata` API，只能在聚宽平台运行。无独立实盘部署代码。
- **入口在哪？**
  `all_weather.py:7` — `initialize(context)` 函数，聚宽平台自动调用。

---

## 第 1 层：配置与参数

### 一句话作用

策略的"旋钮"全部以**硬编码类属性**或**函数内字面量**形式存在，无独立配置文件。

### 关键文件与函数

- [all_weather.py:313-332](file:///e:/Quant/all-weather-strategy/all_weather.py#L313-L332) — `All_Weather_Strategy.__init__()` 策略参数
- [all_weather.py:161-168](file:///e:/Quant/all-weather-strategy/all_weather.py#L161-L168) — `Strategy.__init__()` 基类参数
- [all_weather.py:442-449](file:///e:/Quant/all-weather-strategy/all_weather.py#L442-L449) — `calculate_asset_allocations()` 风险预算
- [all_weather.py:620-634](file:///e:/Quant/all-weather-strategy/all_weather.py#L620-L634) — `set_trading_costs()` 成本模型
- [all_weather.py:14-16](file:///e:/Quant/all-weather-strategy/all_weather.py#L14-L16) — `g.portfolio_value_proportion` 资金分配

### 输入

无外部配置输入，所有参数在代码中静态定义。

### 输出

策略运行时生效的参数集。

### 核心参数清单

#### 标的池（ETF Pool）

| 参数                               | 默认值                             | 位置                   | 说明             |
| ---------------------------------- | ---------------------------------- | ---------------------- | ---------------- |
| `etf_pool_all["equity"]`         | `['510300.XSHG', '159915.XSHE']` | `all_weather.py:326` | 沪深300 + 创业板 |
| `etf_pool_all["commodities"]`    | `['518880.XSHG']`                | `all_weather.py:327` | 黄金ETF          |
| `etf_pool_all["bonds"]`          | `['161716.XSHE']`                | `all_weather.py:328` | 招商双债         |
| `etf_pool_all["foreign_equity"]` | `['513100.XSHG']`                | `all_weather.py:329` | 纳指ETF          |

#### 调仓频率

| 参数                    | 默认值   | 位置                   | 说明                            |
| ----------------------- | -------- | ---------------------- | ------------------------------- |
| `cycle_period`        | `20`   | `all_weather.py:321` | 每 20 个交易日强制调仓          |
| `rebalance_threshold` | `0.15` | `all_weather.py:318` | 任一持仓价格偏离超 15% 触发调仓 |

#### 风险模型

| 参数                              | 默认值     | 位置                   | 说明              |
| --------------------------------- | ---------- | ---------------------- | ----------------- |
| `risk_budget["bonds"]`          | `0.4`    | `all_weather.py:444` | 债券风险预算最低  |
| `risk_budget["equity"]`         | `1.5`    | `all_weather.py:445` | 股票风险预算      |
| `risk_budget["foreign_equity"]` | `1.5`    | `all_weather.py:446` | 海外风险预算      |
| `risk_budget["commodities"]`    | `1.2`    | `all_weather.py:447` | 商品风险预算      |
| ES 置信水平                       | `0.90`   | `all_weather.py:338` | ES(90%)           |
| ES 回溯窗口                       | `120` 天 | `all_weather.py:334` | lag 参数          |
| 动量窗口                          | `25` 天  | `all_weather.py:389` | momentum_day 参数 |
| 波动率调整：短期窗口              | `10` 天  | `all_weather.py:347` | rolling(10)       |
| 波动率调整：长期窗口              | `250` 天 | `all_weather.py:348` | 全量 std          |

#### 动量调整 Sigmoid

| 参数          | 默认值   | 位置                   | 说明                                 |
| ------------- | -------- | ---------------------- | ------------------------------------ |
| `beta`      | `5`    | `all_weather.py:414` | Sigmoid 陡峭度                       |
| `min_ratio` | `0.75` | `all_weather.py:414` | 调整下限（强动量时 ES 打 75 折）     |
| `max_ratio` | `1.25` | `all_weather.py:414` | 调整上限（弱动量时 ES 放大 1.25 倍） |

#### 交易执行

| 参数                 | 默认值   | 位置                       | 说明                  |
| -------------------- | -------- | -------------------------- | --------------------- |
| `trade_threshold`  | `0.05` | `all_weather.py:322`     | 偏差 <5% 不交易       |
| `min_listing_days` | `125`  | `all_weather.py:323`     | ETF 最少上市天数      |
| `stop_loss_rate`   | `0`    | `all_weather.py:324`     | 全天候策略关闭止损    |
| 最小交易金额         | 分段函数 | `all_weather.py:186-194` | 2000 ~ 50000 动态计算 |

#### 成本模型（硬编码在 `set_trading_costs()`）

| 类型  | 印花税   | 佣金    | 最低佣金 | 滑点 |
| ----- | -------- | ------- | -------- | ---- |
| stock | 卖 0.05% | 0.85/万 | 5 元     | 0.2% |
| fund  | 0        | 0.5/万  | 5 元     | 0.1% |
| mmf   | 0        | 0       | 0        | —   |

### 风险与坑

1. **全部硬编码**：无配置文件，改参数需要直接改源码，容易遗漏。
2. **`risk_budget` 与 ES 单位不一致**：ES 是日收益率量级（~0.01），`1/ES` 量级约 100，乘以 risk_budget（0.4~1.5）后数值跨度大，但最终会归一化，所以不影响结果。不过裸眼看 `inv_ES` 值时容易误判。
3. **`stop_loss_rate = 0` 导致止损完全关闭**：全天候策略不设止损，完全依赖定期调仓控制风险。
4. **`min_listing_days = 125`**：如果 ETF 池中某只 ETF 上市不足 125 天会被跳过，但当前池中 ETF 均为老牌产品，实际无影响。
5. **`g.portfolio_value_proportion = [1.0]`**：数组设计暗示支持多策略组合，但当前只有全天候 100%，多策略框架未启用。

### [待确认]

- 无

---

## 第 2 层：数据层

### 一句话作用

通过聚宽 `jqdata` API 获取 ETF 历史日线行情数据（价格、涨跌停价）和实时行情数据，进行日收益率计算、波动率估算、动量评分。

### 关键文件与函数

- [all_weather.py:334-357](file:///e:/Quant/all-weather-strategy/all_weather.py#L334-L357) — `All_Weather_Strategy.get_ES()` 获取历史价格并计算 ES
- [all_weather.py:358-386](file:///e:/Quant/all-weather-strategy/all_weather.py#L358-L386) — `All_Weather_Strategy.get_portfolio_ES()` 组合级别 ES
- [all_weather.py:389-395](file:///e:/Quant/all-weather-strategy/all_weather.py#L389-L395) — `All_Weather_Strategy.get_ETF_momentum()` 动量计算
- [all_weather.py:418-440](file:///e:/Quant/all-weather-strategy/all_weather.py#L418-L440) — `All_Weather_Strategy.calculate_asset_class_ES()` 数据汇总入口
- [all_weather.py:548-558](file:///e:/Quant/all-weather-strategy/all_weather.py#L548-L558) — `All_Weather_Strategy.prepare_trade()` 调用入口
- [all_weather.py:280-308](file:///e:/Quant/all-weather-strategy/all_weather.py#L280-L308) — `Strategy.check_stop_loss()` 止损数据（全天候关闭）

### 输入

| 数据来源 | API                                                                       | 频率 | 字段                                                   |
| -------- | ------------------------------------------------------------------------- | ---- | ------------------------------------------------------ |
| 聚宽平台 | `history(lag, '1d', 'close', stock, df=True)`                           | 日频 | close（收盘价）                                        |
| 聚宽平台 | `attribute_history(etf, days, '1d', ['close'])`                         | 日频 | close                                                  |
| 聚宽平台 | `get_current_data()`                                                    | 实时 | last_price, paused, high_limit, low_limit, name, is_st |
| 聚宽平台 | `get_security_info(stock).start_date`                                   | 静态 | 上市日期                                               |
| 聚宽平台 | `get_price(stock_list, ..., fields=["close","high_limit"], count=days)` | 日频 | close, high_limit                                      |
| 聚宽平台 | `context.portfolio.positions`                                           | 实时 | price, avg_cost, value, closeable_amount               |

### 输出（进入策略的数据 schema）

最终进入 `calculate_asset_class_ES()` 的数据流：

```
etf_pool_all（硬编码 ETF 代码）
    ├── 过滤：上市天数 > min_listing_days (125)
    ├── get_ES(): 单只 ETF → float (ES 值)
    │     ├── history(120, '1d', 'close') → [120 天日收益率]
    │     ├── history(250, '1d', 'close') → [250 天日收益率 → 波动率调整因子]
    │     └── get_ETF_momentum() → attribute_history(25, '1d', 'close') → float (动量得分)
    └── get_portfolio_ES(): 资产组合 → float (组合 ES)
          ├── history(120, '1d', 'close', stock_list) → [组合日收益率]
          ├── history(250, '1d', 'close', stock_list) → [组合波动率调整]
          └── get_portfolio_momentum() → history(25, '1d', 'close', etf_list) → float
```

### 核心逻辑（伪代码）

```
def get_ES(stock, lag=120):
    # 1. 取最近 lag 天的日线收盘价
    prices = history(lag, '1d', 'close', stock)
    # 2. 计算日收益率
    daily_returns = prices.pct_change().dropna()
    # 3. 排序，取最差 10% 的均值，取负号
    sorted_returns = sorted(daily_returns)
    ES = -mean(sorted_returns[:lag * 0.10])
    # 4. 波动率调整：短期 vol / 长期 vol
    long_returns = history(250, '1d', 'close', stock).pct_change()
    short_vol = rolling_std(long_returns, 10) * sqrt(252)
    long_vol = std(long_returns) * sqrt(252)
    ES *= (short_vol / long_vol)
    # 5. 权益类额外做动量调整
    if stock in equity_etfs:
        score = get_ETF_momentum(stock)  # 年化收益 × R²
        ES = adjust_ES(ES, score)        # Sigmoid 映射到 [0.75, 1.25]
    return ES
```

### 风险与坑

1. **未来函数风险（Look-ahead bias）**：

   - `history()` 和 `attribute_history()` 是聚宽封装的 API，文档声称在回测模式下自动使用 `context.previous_date` 之前的数据，且 `set_option("avoid_future_data", True)` 已开启保护。
   - 但 `history(250, ...)` 取的是 250 个交易日的数据，在当前回测日期下，这些数据全部是历史数据，没有前视。[待确认] 需确认聚宽 `history()` 在回测模式下是否真的以 `end_date=context.previous_date` 为界。
   - `resample('D').last()` 操作存在风险：如果 `history()` 返回的 index 包含当天数据，`resample('D')` 不会自动过滤。但由于聚宽 API 在回测中以 `previous_date` 截止，风险较低。
2. **`fillna(value=0, method=None)` 的歧义**（第 336 行）：

   - `method=None` 和 `value=0` 同时指定。Pandas 新版中这会导致 FutureWarning，实际行为是 `value=0` 生效，缺失值填 0。填 0 意味着"缺失日无收益"，对 ES 计算有微弱影响（拉低尾部风险估计）。
3. **波动率调整的循环依赖**：

   - `get_ES()` 中用过去 250 天的全部收益率计算长期波动率，再从中取 `rolling(10).std().iloc[-1]` 作为短期波动率。两者共享同一段数据，且 `iloc[-1]` 取的是最近一天（即回测日的前一日），这是正确的。
   - 但 `history(250, ...)` 和 `history(120, ...)` 两次独立调用，产生两次网络请求，数据范围有重叠（最近 120 天被请求了两次）。
4. **动量得分 `r_squared` 计算**（第 394 行）：

   - 公式 `1 - (SS_res / ((n-1) * var(y)))` 等价于标准 R² = `1 - SS_res / SS_tot`，数学上正确。
   - 但当 `var(y) == 0`（价格完全不变）时会出现除以零，代码未处理。[待确认] ETF 价格不太可能 25 天完全不变，但理论上存在风险。
5. **`get_portfolio_momentum()` 中权重归一化**：

   - `weight_sum = float(sum(list(weights.values())))` 做了显式转换，但 `sum()` 在 Python 中已经返回数值，冗余但无害。

### [待确认]

- 聚宽 `history(lag, '1d', 'close', stock, df=True)` 在回测模式下，返回数据的截止日期是否确实为 `context.previous_date`？
- 聚宽的 `resample('D').last()` 在日频数据上是否等价于原数据（因为本身就是日频）？

---

## 第 3 层：信号/因子计算

### 一句话作用

计算每只 ETF 的 Expected Shortfall（ES，90% 置信水平），并通过波动率调整因子和动量调整因子动态修正，最终输出每只 ETF 的风险估计值。

### 关键文件与函数

| 函数                           | 位置                       | 作用                                          |
| ------------------------------ | -------------------------- | --------------------------------------------- |
| `get_ES()`                   | `all_weather.py:334-357` | 单 ETF 的 ES 计算（含波动率调整 + 动量调整）  |
| `get_portfolio_ES()`         | `all_weather.py:358-386` | 资产组合的 ES 计算（含波动率调整 + 动量调整） |
| `get_ETF_momentum()`         | `all_weather.py:389-395` | 单 ETF 动量得分                               |
| `get_portfolio_momentum()`   | `all_weather.py:397-413` | 组合动量得分                                  |
| `adjust_ES()`                | `all_weather.py:414-416` | Sigmoid 动量调整函数                          |
| `get_ma_signal()`            | `all_weather.py:418-419` | 均线信号（当前始终返回 1，未生效）            |
| `calculate_asset_class_ES()` | `all_weather.py:421-440` | 汇总所有 ETF 和资产大类的 ES                  |

### 输入

- ETF 代码列表（来自 `etf_pool_all` 硬编码）
- `history()` / `attribute_history()` 返回的历史价格 DataFrame

### 输出

- `asset_ES`: `{"equity": float, "commodities": float, "bonds": float, "foreign_equity": float}` — 每类资产的 ES 值
- `etf_ES`: `{"510300.XSHG": float, ...}` — 每只 ETF 的 ES 值
- `valid_etf_pools`: `{"equity": [...], ...}` — 通过上市天数过滤的有效 ETF 列表

### 核心逻辑（伪代码 + 数学定义）

#### 信号 1：Expected Shortfall（ES）

```
输入: stock, lag=120, confidence=0.90
输出: float (ES 值)

步骤:
1. 取 lag 天日线收盘价 → 计算日收益率 r₁, r₂, ..., rₙ
2. 排序: r₍₁₎ ≤ r₍₂₎ ≤ ... ≤ r₍ₙ₎
3. 取最差 α = 1-0.90 = 10% 的收益率的均值:
   ES_raw = -(1 / floor(n*α)) * Σᵢ₌₁^{floor(nα)} r₍ᵢ₎
4. 波动率调整:
   短期年化波动率 σ_short = rolling_std(r_250, 10)[-1] * √252
   长期年化波动率 σ_long  = std(r_250) * √252
   adj_vol = σ_short / σ_long  (若 σ_long=0 则为 1)
   ES_vol_adj = ES_raw * adj_vol
5. 动量调整 (仅权益类 ETF):
   score = get_ETF_momentum(stock)
   ES_final = ES_vol_adj * sigmoid(score, β=5, range=[0.75, 1.25])
```

**ES 的金融含义**：在 90% 置信水平下，当收益率突破 VaR 阈值后，预期的平均亏损幅度。ES 越大 = 风险越大。

#### 信号 2：得分

```
输入: ETF代码, momentum_day=25
输出: float (score = 年化收益率 × R²)

步骤:
1. 取最近 25 天收盘价
2. 对数线性回归: ln(price) = slope * t + intercept
3. 年化收益率 = exp(slope)^250 - 1
4. R² = 1 - SS_res / SS_tot
5. score = 年化收益率 × R²
```

**Score 的含义**：

- score > 0（正动量 + 趋势可靠）→ ES 被下调（风险估计更乐观）
- score < 0（负动量 + 趋势可靠）→ ES 被上调（风险估计更悲观）
- score ≈ 0（趋势不可靠或接近零收益）→ ES 几乎不调整

#### 信号 3：Sigmoid 调整函数

```
adjust_ES(ES, score, beta=5, min_ratio=0.75, max_ratio=1.25)

adjustment_factor = max_ratio - (max_ratio - min_ratio) / (1 + exp(-beta * score))
                  = 1.25 - 0.5 / (1 + exp(-5 * score))

映射关系:
  score → -∞ : factor → 1.25  (最大惩罚，ES 放大 25%)
  score = 0  : factor → 1.00  (不调整)
  score → +∞ : factor → 0.75  (最大奖励，ES 打 75 折)
```

### 信号时间尺度

- **频率**：日频（`'1d'`）
- **更新频率**：每 20 个交易日 或 价格偏离超 15% 时更新
- **回溯窗口**：ES 用 120 天 + 250 天，动量用 25 天

### 信号类型

所有信号都是**连续值**（float），不是离散信号。阈值体现在：

- `trade_threshold = 0.05`（调仓执行阶段，非信号阶段）
- `rebalance_threshold = 0.15`（触发条件，非信号阈值）

### 风险与坑

1. **`get_ma_signal()` 恒返回 1**（第 418 行）：（择时开关）

   - 函数名暗示 EMA 均线信号，但实现永远返回 1，意味着权益类的 EMA 过滤实际上**从未生效**。这是一个预留但未实现的信号接口。
2. **动量得分的 R² 稳定性**：

   - 25 天样本量较小，R² 估计不稳定。极端情况下（如 ETF 连续涨停或跌停），`np.var(y, ddof=1)` 可能接近 0，导致 R² 计算出现除零或极大值。
3. **`polyfit` 的数值稳定性**：

   - `x = np.arange(len(y))` 范围是 0~24，`y = np.log(close)` 范围约 ±0.1。斜率的量级很小（~0.001），年化后 `exp(slope)^250 - 1` 会将微小误差放大 250 倍。
4. **ES 与动量调整的相互作用**：

   - 权益类 ETF 的 ES 被调整了两次：波动率调整 + 动量调整。如果市场剧烈波动（高波动率 + 负动量），ES 可能被双倍放大，导致权益配置被过度压低。
5. **债券和商品不享受动量调整**：

   - `get_ES()` 中只有 `equity_etfs` 才做动量调整（第 353 行），债券和商品的 ES 仅做波动率调整。这是有意设计（债券动量信号不稳定），但未在代码注释中说明。

### [待确认]

- `get_ma_signal()` 是预留接口还是忘记实现？README 未提及 EMA 过滤。
- `get_portfolio_momentum()` 中的 `momentum_day` 参数（第 397 行）在 `get_portfolio_ES()` 调用时未显式传入，使用的是默认值 25，与 `get_ETF_momentum()` 一致。这是有意为之还是疏漏？

---

## 第 4 层：组合与仓位

### 一句话作用

从资产大类 ES → 风险预算加权 → 大类权重，再分配至具体 ETF（按 1/ES 加权），产出目标持仓向量（ETF 代码 → 权重）。

### 关键文件与函数

| 函数                              | 位置                       | 作用                           |
| --------------------------------- | -------------------------- | ------------------------------ |
| `calculate_asset_allocations()` | `all_weather.py:442-455` | ES → 大类权重                 |
| `calculate_etf_allocations()`   | `all_weather.py:457-530` | 大类权重 → ETF 权重           |
| `prepare_trade()`               | `all_weather.py:548-558` | 汇总上述两步，产出最终目标仓位 |
| `prepare()`                     | `all_weather.py:560-566` | 调仓触发逻辑                   |

### 输入

- `asset_ES_results`: `{"equity": 0.023, "bonds": 0.008, ...}` — 来自第 3 层
- `etf_ES_results`: `{"510300.XSHG": 0.025, ...}` — 来自第 3 层
- `valid_etf_pools`: `{"equity": ["510300.XSHG", "159915.XSHE"], ...}` — 来自第 3 层

### 输出

- **大类权重** `asset_allocations`: `{"equity": 0.15, "bonds": 0.55, "commodities": 0.20, "foreign_equity": 0.10}`
- **ETF 权重** `etf_allocations`: `{"510300.XSHG": 0.10, "159915.XSHE": 0.05, "518880.XSHG": 0.20, ...}`
- 所有权重之和 = 1.0

### 核心逻辑（伪代码 + 数学定义）

#### 步骤 1：大类层面的风险平价

```
输入: asset_ES = {"equity": ES_e, "bonds": ES_b, "commodities": ES_c, "foreign_equity": ES_f}
      risk_budget = {"bonds": 0.4, "equity": 1.5, "foreign_equity": 1.5, "commodities": 1.2}

对每类资产:
  inv_ES[class] = (1 / ES[class]) * risk_budget[class]  (ES ≤ 0 时取 0)

归一化:
  total = sum(inv_ES.values())
  weight[class] = inv_ES[class] / total
```

**数学含义**：
权重 ∝ (1/ES) × risk_budget，即"风险越大配得越少"（风险平价核心思想），同时通过 risk_budget 调节各类资产的相对风险敞口。债券 risk_budget 最低（0.4），体现了全天候"债券打底"的理念。

#### 步骤 2：闲置资金再分配（从未触发）

```
对权益类 ETF（equity + foreign_equity）:
  如果某大类的所有 ETF 都没有 EMA 信号 (get_ma_signal() ≠ 1):
    该大类的权重 → unused_funds (闲置资金池)

如果存在活跃大类:
  scale_factor = 1 + unused_funds / sum(活跃大类权重)
  所有活跃大类权重 ×= scale_factor  (闲置资金按比例重新分配)
```

**注意**：由于 `get_ma_signal()` 始终返回 1，权益类永远不会被标记为无信号，所以 `unused_funds` 实际上永远为 0，闲置资金再分配逻辑当前**从未触发**。

#### 步骤 3：大类 → ETF 分配

```
对每类资产下的 ETF 列表:
  如果只有 1 只 ETF:
    该 ETF 权重 = 大类权重
  如果有多只 ETF:
    对每只 ETF: inv_ES_etf = 1 / ES_etf
    归一化: etf_weight_within_class = inv_ES_etf / sum(inv_ES_etf)
    最终权重 = etf_weight_within_class * 大类权重
```

**数学含义**：同一大类内按 1/ES 加权，风险越低的 ETF 配得越多。在当前 ETF 池中：

- equity 有 2 只（510300 + 159915），会做内部分配
- 其他大类各只有 1 只 ETF，无内部分配

#### 步骤 4：调仓触发条件

```
prepare():
  if period_count == cycle_period (20):     ← 周期触发
    或 need_rebalance() == True:             ← 价格偏离触发
    period_count = 0
    prepare_trade()  ← 重新计算目标权重
  else:
    period_count += 1

need_rebalance():
  遍历当前持仓:
    if |现价 - 上次调仓时价格| / 上次价格 > 0.15:
      return True
  return False
```

### 权重特性

- **目标值还是增量？**：输出的是**目标权重**（绝对比例），非增量。`adjust()` 中计算 `target_value = allocation * strategy_total_value`，然后与当前持仓比较产生买卖订单。
- **再平衡频率**：至少 20 个交易日一次，或任意持仓价格偏离超 15% 时触发。
- **约束**：代码中无单票上限、无行业中性、无杠杆上限、无换手约束。依赖风险平价自然分散。

### 风险与坑

1. **ETF 池极度集中**：

   - 4 大类共 5 只 ETF，债券和商品各只有 1 只可选。如果某只 ETF 因上市天数不足被过滤，该大类将完全无配置。
   - 同时持有沪深 300（510300）和创业板（159915）存在风格重叠问题（都是 A 股），并非真正的"全天候"分散。
2. **权重输出的精度**：

   - `round(v / total_inv_ES, 3)` 保留 3 位小数，4 类权重之和可能因舍入而不等于 1.000（例如 0.333 + 0.333 + 0.333 + 0.001 = 1.000 但实际是 0.999 或 1.001）。代码未做尾差处理。
3. **风险预算的不对称性**：

   - `risk_budget` 设计上偏向债券（0.4 最低预算），但乘以 `1/ES` 后债券的实际权重通常会非常大（因为债券 ES 天然远小于股票），这符合全天候理念但也意味着策略收益主要由债券驱动。
4. **未使用现金的处理**：

   - 策略始终以 `target_value = strategy_total_value` 为目标（满仓），无预留现金。调仓时先卖后买，如果卖出不足以覆盖买入资金，买入订单会被跳过（第 607 行：`if cash >= target_value * self.trade_threshold`）。
5. **`need_rebalance()` 的参考价格基准**：

   - `rebalance_price` 在 `prepare_trade()` 中记录，取的是"上次计算目标仓位时"的持仓价格。如果 20 个交易日中价格先涨 20% 再跌回原位，该检查不会触发（因为与初始价格对比偏差已回落），但实际上期间已经历了大幅波动。

### [待确认]

- 无

---

## 第 5 层：风控与交易执行

### 一句话作用

将目标持仓向量转换为实际订单，在执行过程中施加涨跌停/停牌保护、最小交易金额过滤、偏差阈值过滤，并通过成本模型模拟交易摩擦。

### 关键文件与函数

| 函数                                 | 位置                       | 作用                            |
| ------------------------------------ | -------------------------- | ------------------------------- |
| `adjust()`                         | `all_weather.py:568-608` | All_Weather_Strategy 的执行入口 |
| `order_target_value_()`            | `all_weather.py:217-248` | 实际下单（基类方法）            |
| `_adjust()`                        | `all_weather.py:196-215` | 基类通用调仓（全天候未使用）    |
| `get_min_trade_value()`            | `all_weather.py:186-194` | 动态最小交易金额                |
| `check_stop_loss()`                | `all_weather.py:280-308` | 回撤止损（全天候关闭）          |
| `check_stop_loss()` (全局)         | `all_weather.py:56-62`   | 全局止损调度                    |
| `end_trade()`                      | `all_weather.py:64-70`   | 尾盘清仓                        |
| `set_trading_costs()`              | `all_weather.py:620-634` | 成本模型配置                    |
| `filter_basic_stock()`             | `all_weather.py:250-260` | 基础过滤（全天候未用）          |
| `filter_limitup_limitdown_stock()` | `all_weather.py:262-268` | 涨跌停过滤（全天候未用）        |
| `filter_limitup_stock()`           | `all_weather.py:270-277` | N日涨停过滤（全天候未用）       |

### 输入

- `etf_allocations`: 目标 ETF 权重字典（来自第 4 层）
- `context.portfolio.positions`: 当前持仓（实时）
- `context.portfolio.total_value` / `available_cash`: 账户状态（实时）
- `get_current_data()`: 实时行情（last_price, paused, high_limit, low_limit）

### 输出

- 实际发出的订单列表（通过聚宽 `order()` 函数）
- 更新后的 `g.positions[index]` 持仓记录

### 核心逻辑（伪代码）

#### All_Weather_Strategy.adjust()

```
adjust():
  if etf_allocations 为空:
    return

  strategy_total_value = 总市值 × 该策略资金占比

  # 第一步：构建卖出订单
  sell_orders = {}
  for 当前持仓中每只 ETF:
    target_value = 目标权重 × strategy_total_value
    if target_value < current_value:
      sell_orders[etf] = target_value

  # 第二步：构建买入订单
  buy_orders = {}
  for 目标中的每只 ETF:
    target_value = 目标权重 × strategy_total_value
    if target_value > current_value (无持仓按 0):
      buy_orders[etf] = target_value

  # 第三步：执行卖出
  for 每笔卖出:
    deviation = (current_value - target_value) / target_value
    if deviation > 5% 且 差值金额 > 1000:
      下单卖出到 target_value

  # 第四步：执行买入
  for 每笔买入:
    deviation = (target_value - current_value) / target_value
    if deviation > 5% 且 差值金额 > 1000 且 可用现金 > target_value × 5%:
      下单买入到 target_value
```

**注意**：`adjust()` 是全天候策略**自己重写的版本**，没有调用基类的 `_adjust()`。两者逻辑类似但细节不同。

#### order_target_value_() 下单保护

```
order_target_value_(security, target_value):
  # 保护 1：停牌不交易
  if security 停牌: return False

  # 保护 2：涨跌停不交易
  if 现价 == 涨停价 or 现价 == 跌停价: return False

  # 保护 3：手数取整（A 股 100 股整数倍）
  target_position = floor(target_value / price / 100) * 100
  if target_position == 0 且 target_value > 0: return False

  # 保护 4：可用持仓检查（卖出时）
  adjustment = target_position - current_position
  if adjustment < 0 (卖出) 且 可卖数量 == 0: return False

  # 下单
  if adjustment != 0:
    o = order(security, adjustment)
    if o 成交:
      if adjustment > 0 (买入):
        记录止损跟踪价格 (如果该标的尚未记录)
      更新 g.positions[index][security]
      if 持仓归零: 从 g.positions 中删除该标的
      return True
  return False
```

### 风控规则清单

| 风控措施           | 类型 | 是否生效      | 位置                                    |
| ------------------ | ---- | ------------- | --------------------------------------- |
| 涨跌停过滤         | 事前 | ✅ 生效       | `all_weather.py:224-227`              |
| 停牌过滤           | 事前 | ✅ 生效       | `all_weather.py:222-223`              |
| 可卖数量检查       | 事前 | ✅ 生效       | `all_weather.py:239-240`              |
| 手数取整（100 股） | 事前 | ✅ 生效       | `all_weather.py:231`                  |
| 最小交易金额       | 事前 | ✅ 生效       | `all_weather.py:186-194`              |
| 偏差阈值 5%        | 事前 | ✅ 生效       | `all_weather.py:596,607`              |
| 现金充足检查       | 事前 | ✅ 生效       | `all_weather.py:607`                  |
| 回撤止损 12%       | 事后 | ❌ 关闭       | `all_weather.py:324` stop_loss_rate=0 |
| 尾盘清仓未记录持仓 | 事后 | ✅ 生效       | `all_weather.py:64-70`                |
| ST/退市/次新股过滤 | 事前 | ❌ 未用于 ETF | `all_weather.py:250-260`              |

### 成本模型

| 成本类型       | stock 类 | fund 类 | 说明         |
| -------------- | -------- | ------- | ------------ |
| 印花税（买入） | 0        | 0       |              |
| 印花税（卖出） | 0.05%    | 0       |              |
| 佣金           | 0.85/万  | 0.5/万  |              |
| 最低佣金       | 5 元     | 5 元    |              |
| 滑点           | 0.2%     | 0.1%    | 固定比例滑点 |

**注意**：ETF 在聚宽平台通常归类为 `stock` 类型（代码后缀 `.XSHG`/`.XSHE`），适用 stock 类费率（印花税 0.05% 卖出 + 佣金 0.85/万）。

### 执行假设

- **市价单**：`order(security, adjustment)` 聚宽默认为市价单，回测中假设立即以当前 bar 的价格成交。
- **部分成交**：代码中检查 `o.filled`（实际成交数量）并据此更新持仓（第 245 行），但未处理部分成交后的重试逻辑。
- **无冲击成本**：成本模型中只有固定比例滑点，无成交量相关的冲击成本。

### 风险与坑

1. **卖出-买入顺序可能导致资金不足**：

   - `adjust()` 先执行全部卖出，再执行全部买入。卖出释放的现金在聚宽回测框架中可能不会立即可用于买入（取决于框架设计），代码中直接使用 `context.portfolio.available_cash` 做买入检查（第 607 行）。[待确认] 聚宽回测中卖出是否立即释放现金？
2. **`buy_orders` 中 `current_value = 1` 的处理**（第 602 行）：

   - 对于无持仓的 ETF，`current_value` 被硬编码为 `1` 而非 `0`。这是一个**有意的小技巧**：如果 `current_value = 0`，则 `(target_value - 0) / target_value = 1`，偏差 100%，必定触发买入。但设为 `1` 后，对于极小目标值（如 target_value=100），偏差计算 `(100-1)/100 = 0.99 > 0.05` 仍然触发，效果几乎相同。区别在于 `target_value = 0` 时 `(0-0)/0` 会除零错误，但 `target_value > current_value` 条件下 `target_value` 不可能为 0，所以这个处理是安全的。
3. **`sell_orders` 中 `target_value == 0` 的特殊处理**（第 598 行）：

   - 只有当 `target_value == 0` 且 `current_value > 1000` 时才执行清仓。如果 `target_value > 0` 但偏差不足 5%，则不会卖出。这意味着如果某 ETF 应减持但减幅很小，会被跳过，导致持仓与目标存在系统性偏差。
4. **`buy_orders` 中未在目标列表的 ETF 不会被卖出**（第 576-580 行）：

   - `sell_orders` 只遍历 `context.portfolio.positions`（账户实际持仓），而非 `g.positions[index]`（策略记录的持仓）。如果策略记录中有某 ETF 但实际持仓中没有（例如被外部卖出），该 ETF 不会出现在 sell_orders 中——它本来也不需要卖出。但如果反过来——策略记录中没有但实际持仓中有——会被 `end_trade()` 在 14:59 清仓。
5. **尾盘 `end_trade()` 使用全局 `order_target_value()`**（第 68 行）：

   - `end_trade()` 调用的是聚宽内置的 `order_target_value()`，不是策略自己的 `order_target_value_()`。这意味着尾盘清仓**绕过了策略的涨跌停/停牌保护和 `g.positions` 持仓更新**。被清仓的标的只从实际持仓中移除，`g.positions[index]` 中可能仍然保留记录。
6. **止损跟踪价格只在买入时初始化**（第 243-244 行）：

   - `stop_loss_tracking[security] = price` 仅在 `adjustment > 0`（买入）时记录。如果策略初始持仓是从回测初始资金买入的（首次调仓），止损跟踪价格会被正确设定。但如果初始持仓是回测框架自动建立的（非策略下单），则止损跟踪价格可能缺失。

### [待确认]

- 聚宽回测中 `order()` 卖出后，`available_cash` 是否立即更新为包含卖出所得？
- 聚宽回测中 `order()` 是市价单还是限价单？部分成交的概率模型是什么？

---

## 第 6 层：回测/评估循环

### 一句话作用

聚宽平台内置回测引擎负责主循环、账户更新和绩效计算；本策略代码只负责信号计算和订单生成，不包含独立的回测循环代码。

### 关键文件与函数

| 功能         | 位置                                                     | 说明                      |
| ------------ | -------------------------------------------------------- | ------------------------- |
| 日终持仓记录 | `all_weather.py:78-157` — `record_daily_holdings()` | 每日 15:00 记录持仓和盈亏 |
| 日志输出     | 策略各函数中的 `log.info()` / `log.error()`          | 调试/监控信息             |
| 图表记录     | `all_weather.py:155-157` — `record()`               | 聚宽图表绑定              |
| 基准设置     | `all_weather.py:8` — `set_benchmark("000300.XSHG")` | 回测对比基准              |

### 输入

- 聚宽回测引擎自动提供：初始资金、回测起止日期、日频 bar 数据

### 输出

- `g.daily_holdings`: 每日持仓记录列表（内存中，回测结束后可导出）
- 聚宽平台自动生成的绩效报告（收益曲线、回撤、Sharpe、换手率等）
- `record()` 绑定的自定义图表指标

### 核心逻辑

#### 主循环结构

```
聚宽回测引擎（黑盒，本代码不可见）:
  for each 交易日 in 回测区间:
    按时间顺序触发注册的 run_daily() 回调:
      9:10 → all_weather_prepare() → 计算目标仓位
      9:35 → all_weather_adjust()  → 执行调仓
      14:50 → check_stop_loss()    → 止损检查
      14:59 → end_trade()          → 清仓未记录持仓
      15:00 → record_daily_holdings() → 记录持仓
```

**特性**：本策略是**逐 bar（逐日）**回调模式，不是向量化回测。聚宽引擎按交易日推进，每天触发注册的函数。

#### 账户更新

- 聚宽引擎自动管理 `context.portfolio`：现金、持仓市值、净值、盈亏、可用现金
- 策略代码通过 `g.positions[index]` 额外维护一份持仓记录（金额单位而非股数），用于策略内部的状态跟踪
- 策略自行计算部分绩效指标（`record_daily_holdings()` 中的当日收益、累计收益）

#### 绩效指标

| 指标来源                          | 说明                                                                                        |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| 聚宽内置                          | 年化收益、最大回撤、Sharpe Ratio、Calmar Ratio、胜率、盈亏比、Alpha、Beta、信息比率、换手率 |
| 策略自定义 (`record()`)         | 总市值、现金、持仓市值、总收益率、各策略持仓占比、各策略收益率                              |
| 策略自定义 (`g.daily_holdings`) | 每日完整持仓明细（可事后分析）                                                              |

#### 样本划分

- 无训练/验证/测试集划分。这是一个**纯配置型策略**（规则驱动，非 ML），参数通过经验设定，无 walk-forward 优化。
- README 提到"实盘留存率 50%"的验证方式：回测得出的年化收益与实盘年化收益的比值，作为回测可信度的参考指标。

### 风险与坑

1. **回测-实盘一致性**：

   - 回测与实盘**共用同一套代码**（都粘贴到聚宽运行），这是最大的一致性保障。
   - 但回测中的成交假设（立即市价成交、无冲击成本、固定滑点）在实盘中不成立。
   - README 自述"实盘留存率约 50%"，说明回测结果存在显著高估。
2. **`g.daily_holdings` 内存累积**：

   - `g.daily_holdings.append(holding_record)` 每天追加一条记录。如果回测 10 年（约 2500 个交易日），列表中将有 2500 个字典对象。对于聚宽平台的内存限制，这可能不是问题，但如果在本地分析需要注意。
3. **策略收益的累计方式**（第 119 行）：

   - `g.strategy_returns[strategy_name] += strategy_profit` 是**简单加法累加**，不是复利计算。`strategy_profit = strategy_hold_value - strategy_allocated_value` 每天重新计算，累加后相当于"每日分配本金不受前一日盈亏影响"的假设。这与实际账户的复利增长不一致。
4. **无回测-实盘差异的代码层面处理**：

   - 代码中没有针对实盘的额外逻辑（如重连、状态持久化），说明回测和实盘共用完全相同的代码。README 提到"模拟交易先观察"是聚宽平台的功能，并非本代码的独立实现。

### [待确认]

- 聚宽回测引擎是否支持分钟级 bar？本策略全部使用日频数据，但 `run_daily()` 回调时间精确到分钟（9:10、9:35），这意味着回测引擎至少在日内的分钟粒度推进。
- 聚宽的 `record()` 函数是否在回测结束后自动生成图表？还是需要手动配置？

---

## 第 7 层：实盘/部署

### 一句话作用

本策略**没有独立的实盘代码**。实盘运行依赖聚宽平台的"模拟交易"功能，代码与回测完全一致，无交易所接口、无状态持久化、无重连机制。

### 关键文件与函数

| 功能         | 位置                        | 说明                                               |
| ------------ | --------------------------- | -------------------------------------------------- |
| 策略全部代码 | `all_weather.py` (全文件) | 回测和模拟交易共用                                 |
| 实盘部署说明 | `README.md:35-40`         | "粘贴 all_weather.py 全部代码 → 用模拟交易先观察" |

### 输入

- 聚宽模拟交易平台自动提供：实时行情、账户信息、订单执行

### 输出

- 聚宽模拟交易平台自动显示：持仓、委托、成交、盈亏

### 核心逻辑

```
部署流程（全手动）:
1. 注册聚宽账号
2. 新建策略
3. 粘贴 all_weather.py 全部代码
4. 点击"模拟交易"启动

运行逻辑:
  与回测完全相同。每天 9:10/9:35/14:50/14:59/15:00
  触发相同的 run_daily() 回调函数。
```

### 回测与实盘的代码复用

| 模块       | 回测           | 实盘（模拟交易）                   | 复用程度           |
| ---------- | -------------- | ---------------------------------- | ------------------ |
| 信号计算   | ✅             | ✅                                 | 100% 相同          |
| 组合权重   | ✅             | ✅                                 | 100% 相同          |
| 订单生成   | ✅             | ✅                                 | 100% 相同          |
| 风控规则   | ✅             | ✅                                 | 100% 相同          |
| 成本模型   | ✅（固定滑点） | ✅（相同代码，但实盘实际成本不同） | 代码相同，效果不同 |
| 账户管理   | 聚宽引擎       | 聚宽引擎                           | 100% 相同          |
| 状态持久化 | ❌ 无          | ❌ 无                              | 均缺失             |

### 回测-实盘一致性风险点

| 风险                                       | 严重程度 | 说明                                                                                                                                         |
| ------------------------------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **成交假设差异**                     | ⚠️ 高  | 回测假设市价立即成交；实盘存在排队、部分成交、滑点                                                                                           |
| **固定滑点模型**                     | ⚠️ 中  | 代码中滑点是固定比例（0.2%/0.1%），实盘滑点随市场流动性和订单大小变化                                                                        |
| **无冲击成本**                       | ⚠️ 中  | 策略资金规模小（README 提到 8 万），ETF 流动性好，冲击成本可能不大                                                                           |
| **停牌/涨跌停处理**                  | ⚠️ 低  | 代码已有停牌和涨跌停保护（`order_target_value_()` 中），回测和实盘行为一致                                                                 |
| **无状态持久化**                     | ⚠️ 中  | `g` 对象在聚宽平台中由平台管理，模拟交易重启后状态可能丢失（`g.positions`、`g.daily_holdings` 等）。策略没有从持久化存储恢复状态的逻辑 |
| **`end_trade()` 绕过策略内部记录** | ⚠️ 低  | 使用全局 `order_target_value()` 而非策略自己的 `order_target_value_()`，可能导致 `g.positions` 与实际持仓不一致                        |
| **`get_ma_signal()` 恒为 1**       | ⚠️ 低  | 回测与实盘行为一致，但信号本身无区分能力                                                                                                     |

### 风险与坑

1. **无独立的实盘工程结构**：整个策略是"单文件粘贴到平台运行"的模式，没有本地可运行的实盘系统。如需迁移到其他平台（如 QMT、PTrade），需要完全重写数据层和交易执行层。
2. **README 实盘数据的手动性质**：README 中的实盘收益数据（"首月 1.6%，近 3 个月 3.1%"）来自手动记录，非代码自动统计。回测与实盘的差值验证也是人工进行。
3. **无法本地回测**：依赖聚宽 `jqdata` API，离开聚宽平台无法运行。本地开发只能做静态代码分析，无法实际运行和调试。
4. **模拟交易 vs 实盘的区别**：聚宽的"模拟交易"仍然使用模拟环境（模拟撮合、模拟行情延迟），与真实券商实盘环境有差距。README 提到的"实盘"更可能指模拟交易而非真实券商账户。[待确认]

### [待确认]

- README 中提到的"实盘资金 8 万"是指聚宽模拟交易还是真实券商账户？
- 策略是否有计划迁移到独立实盘框架（如 QMT、PTrade、vnpy）？
- 聚宽模拟交易在策略重启后是否会保留 `g` 全局变量的状态？

---

## 附录：完整调用链

```
initialize(context)
  └── process_initialize(context)
        └── All_Weather_Strategy.__init__()
              └── Strategy.__init__()

==================== 每个交易日 ====================

9:10  run_daily(all_weather_prepare, "9:10")
  └── All_Weather_Strategy.prepare()
        ├── period_count == cycle_period? (20 天周期触发)
        ├── need_rebalance()? (价格偏离 15% 触发)
        └── prepare_trade()
              ├── calculate_asset_class_ES()
              │     ├── get_ES() × N  (每只 ETF)
              │     │     ├── history(120, '1d', 'close')
              │     │     ├── history(250, '1d', 'close')
              │     │     ├── get_ETF_momentum()
              │     │     └── adjust_ES()
              │     └── get_portfolio_ES() × M  (每类资产，如果 >1 ETF)
              ├── calculate_asset_allocations()
              │     └── risk_budget 风险平价
              └── calculate_etf_allocations()
                    ├── get_ma_signal()  (恒为 1)
                    └── 1/ES ETF 内部分配

9:35  run_daily(all_weather_adjust, "9:35")
  └── All_Weather_Strategy.adjust()
        ├── 构建 sell_orders / buy_orders
        ├── 卖出 (偏差 > 5% 且 > 1000 元)
        ├── 买入 (偏差 > 5% 且 > 1000 元 且 现金充足)
        └── order_target_value_()
              ├── 停牌检查
              ├── 涨跌停检查
              ├── 手数取整
              └── order() 下单

14:50 run_daily(check_stop_loss, "14:50")
  └── Strategy.check_stop_loss()
        (stop_loss_rate=0 → 跳过)

14:59 run_daily(end_trade, "14:59")
  └── 清仓 g.positions 中未记录的持仓

15:00 run_daily(record_daily_holdings, "15:00")
  └── 记录持仓明细、盈亏、日志、图表
```

---

> 分析完成时间：2026-10-27
> 分析范围：`all-weather-strategy` 仓库全部源码（634 行）
> 分析工具：手动逐行阅读 + 聚宽 API 文档参考
