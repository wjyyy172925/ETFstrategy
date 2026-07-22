# A股ETF模拟全天候策略（带约束风险平价） Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用A股ETF实现带约束的等风险贡献（ERC）风险平价策略，模拟桥水全天候策略，含完整的回测、风控和绩效分析。

**Architecture:** 分层模块设计：config（参数配置）→ data（数据加载）→ risk_parity（ERC权重计算）→ backtest_engine（回测执行+风控）→ performance（绩效分析+基准对比）。入口脚本编排全流程。

**Tech Stack:** Python, pandas, numpy, scipy.optimize, akshare, matplotlib

---

### Task 1: 配置模块 — config.py

**Files:**
- Create: `2A股ETF模拟全天候策略/config.py`

- [ ] **Step 1: 创建 config.py 定义ETF池和策略参数**

```python
ETF_POOL = [
    {"code": "510300", "name": "沪深300ETF",  "asset_class": "equity"},
    {"code": "510500", "name": "中证500ETF",  "asset_class": "equity"},
    {"code": "511260", "name": "十年国债ETF", "asset_class": "long_bond"},
    {"code": "511010", "name": "国债ETF",     "asset_class": "mid_bond"},
    {"code": "159649", "name": "国开债ETF",   "asset_class": "mid_bond"},
    {"code": "518880", "name": "黄金ETF",     "asset_class": "gold"},
    {"code": "159985", "name": "豆粕ETF",     "asset_class": "commodity"},
]

BENCHMARK_CODES = ["510300"]  # 沪深300 作为对比基准

ASSET_CLASS_MAP = {
    "equity":    {"risk_budget": 0.25, "label": "权益"},
    "long_bond": {"risk_budget": 0.25, "label": "长久期利率债"},
    "mid_bond":  {"risk_budget": 0.25, "label": "中短债+信用债"},
    "gold":      {"risk_budget": 0.25, "label": "黄金"},
    "commodity": {"risk_budget": 0.25, "label": "商品"},
}

# ERC 约束
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.40
SINGLE_ETF_MAX = 0.30
VOL_WINDOW = 60  # 滚动窗口（交易日）

# 再平衡
REBALANCE_FREQ = "ME"  # 月度
THRESHOLD_PCT = 0.08   # 偏离阈值
VOL_CHANGE_THRESHOLD = 0.30  # 波动率变化阈值
EXECUTION_RATIO = 0.80  # 每次调整到目标权重的80%

# 风控
TARGET_VOL = 0.10     # 目标年化波动率 10%
VOL_BAND = (0.08, 0.12)
CVAR_THRESHOLD = 0.15
DRAWDOWN_WARN = 0.10
DRAWDOWN_MAX = 0.20
LIQUIDITY_MIN_DAILY_VOL = 50_000_000  # 5000万
DURATION_LIMIT = 7.0  # 十年国债久期上限

# 交易成本
COMMISSION_RATE = 0.00003
SLIPPAGE_RATE = 0.0005
TOTAL_COST_PER_SIDE = COMMISSION_RATE + SLIPPAGE_RATE

# 无风险利率
RISK_FREE_RATE = 0.025

# 初始资金
INITIAL_CAPITAL = 100_000_000  # 1亿

# 回测日期
BACKTEST_START = "20150101"
BACKTEST_END = "20260722"

# 基准组合权重
CLASSIC_AA_WEIGHTS = [0.30, 0.55, 0.075, 0.075]  # 权益/债券/黄金/商品
EQUAL_WEIGHT = [0.25, 0.25, 0.25, 0.25]
```

- [ ] **Step 2: 提交**

```bash
git add 2A股ETF模拟全天候策略/config.py
git commit -m "feat: add config module for All Weather risk parity strategy"
```

---

### Task 2: 数据加载模块 — data.py

**Files:**
- Create: `2A股ETF模拟全天候策略/data.py`

- [ ] **Step 1: 实现 get_etf_data_paths 和 load_single_etf**

```python
import os
import pandas as pd

DATA_DIR = "data/etf"


def get_etf_data_path(code: str) -> str:
    return os.path.join(DATA_DIR, f"{code}.csv")


def load_single_etf(code: str) -> pd.DataFrame:
    path = get_etf_data_path(code)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").set_index("日期")
    return df


def load_multi_etf(codes: list[str]) -> pd.DataFrame:
    close_df = pd.DataFrame()
    for code in codes:
        df = load_single_etf(code)
        close_df[code] = df["收盘"]
    close_df = close_df.dropna()
    return close_df
```

- [ ] **Step 2: 实现 download_data 函数**

```python
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from download_etf import download_etf_history, ensure_dir
from config import BACKTEST_START, BACKTEST_END, ETF_POOL, BENCHMARK_CODES


def download_all_data():
    ensure_dir(DATA_DIR)
    all_codes = set()
    for etf in ETF_POOL:
        all_codes.add(etf["code"])
    for code in BENCHMARK_CODES:
        all_codes.add(code)
    for code in sorted(all_codes):
        try:
            download_etf_history(code, "", BACKTEST_START, BACKTEST_END, force=False)
        except Exception as e:
            print(f"下载 {code} 失败: {e}")
```

- [ ] **Step 3: 提交**

```bash
git add 2A股ETF模拟全天候策略/data.py
git commit -m "feat: add data module for ETF loading and download"
```

---

### Task 3: ERC风险平价权重计算 — risk_parity.py

**Files:**
- Create: `2A股ETF模拟全天候策略/risk_parity.py`

- [ ] **Step 1: 实现波动率和协方差计算**

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calc_annual_vol(returns: pd.Series, window: int = 60) -> float:
    return returns.tail(window).std() * np.sqrt(252)


def calc_cov_matrix(returns: pd.DataFrame, window: int = 60) -> np.ndarray:
    return returns.tail(window).cov().values * 252
```

- [ ] **Step 2: 实现ERC优化器**

```python
def _risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    portfolio_var = weights @ cov @ weights
    marginal_contrib = cov @ weights
    risk_contrib = weights * marginal_contrib / np.sqrt(portfolio_var)
    return risk_contrib


def _erc_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    rc = _risk_contribution(weights, cov)
    target = rc.mean()
    return np.sum((rc - target) ** 2)


def solve_erc_weights(returns: pd.DataFrame,
                      w_min: float = 0.05,
                      w_max: float = 0.40) -> np.ndarray:
    n = returns.shape[1]
    cov = calc_cov_matrix(returns)
    bounds = [(w_min, w_max)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    x0 = np.array([1.0 / n] * n)
    result = minimize(_erc_objective, x0, args=(cov,),
                      method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        bounds_loose = [(0.02, 0.60)] * n
        result = minimize(_erc_objective, x0, args=(cov,),
                          method="SLSQP", bounds=bounds_loose, constraints=constraints)
    return result.x / result.x.sum()
```

- [ ] **Step 3: 实现组合权重到ETF权重的映射**

```python
def calc_etf_weights(return_df: pd.DataFrame, etf_pool: list[dict],
                     w_min: float = 0.05, w_max: float = 0.40) -> pd.Series:
    asset_classes = list(dict.fromkeys(e["asset_class"] for e in etf_pool))
    class_returns = pd.DataFrame()
    for ac in asset_classes:
        codes = [e["code"] for e in etf_pool if e["asset_class"] == ac]
        cols = [c for c in codes if c in return_df.columns]
        class_returns[ac] = return_df[cols].mean(axis=1)
    class_weights = solve_erc_weights(class_returns, w_min, w_max)
    etf_weights = pd.Series(0.0, index=return_df.columns)
    for ac, cw in zip(asset_classes, class_weights):
        codes = [e["code"] for e in etf_pool if e["asset_class"] == ac]
        cols = [c for c in codes if c in return_df.columns]
        for col in cols:
            etf_weights[col] = cw / len(cols)
    return etf_weights
```

- [ ] **Step 4: 提交**

```bash
git add 2A股ETF模拟全天候策略/risk_parity.py
git commit -m "feat: add ERC risk parity weight calculator"
```

---

### Task 4: 回测引擎 — backtest_engine.py

**Files:**
- Create: `2A股ETF模拟全天候策略/backtest_engine.py`

- [ ] **Step 1: 实现风控模块**

```python
import numpy as np
import pandas as pd


def check_risk_controls(portfolio_value: float,
                        peak_value: float,
                        current_drawdown: float,
                        daily_returns: pd.Series) -> dict:
    actions = {}
    vol = daily_returns.tail(20).std() * np.sqrt(252)
    if vol > 0.12:
        actions["vol_scale"] = 0.10 / vol
    elif vol < 0.08:
        pass
    if current_drawdown > 0.20:
        actions["emergency_exit"] = True
    elif current_drawdown > 0.10:
        actions["reduce_equity"] = True
    if len(daily_returns) >= 252:
        cvar = daily_returns.tail(252).quantile(0.05)
        if abs(cvar) > 0.15:
            actions["cvar_trigger"] = True
    return actions
```

- [ ] **Step 2: 实现单次再平衡执行**

```python
from config import TOTAL_COST_PER_SIDE, SINGLE_ETF_MAX


def execute_rebalance(current_weights: pd.Series,
                      target_weights: pd.Series,
                      portfolio_value: float,
                      prices: pd.Series) -> list[dict]:
    target_weights_clipped = target_weights.clip(upper=SINGLE_ETF_MAX)
    target_weights_clipped /= target_weights_clipped.sum()
    diff = target_weights_clipped - current_weights
    trades = []
    for code in diff.index:
        delta = diff[code] * portfolio_value * EXECUTION_RATIO
        if abs(delta) < 5000:
            continue
        price = prices.get(code, np.nan)
        if pd.isna(price):
            continue
        direction = "buy" if delta > 0 else "sell"
        trade_amount = abs(delta)
        cost = trade_amount * TOTAL_COST_PER_SIDE
        trades.append({
            "日期": prices.name,
            "代码": code,
            "方向": direction,
            "金额": round(trade_amount, 2),
            "价格": round(price, 4),
            "成本": round(cost, 2),
        })
    return trades
```

- [ ] **Step 3: 实现回测主循环**

```python
def run_backtest(close_df: pd.DataFrame,
                 etf_pool: list[dict]) -> dict:
    from config import INITIAL_CAPITAL, REBALANCE_FREQ
    return_df = close_df.pct_change().dropna()
    cash = INITIAL_CAPITAL
    holdings = pd.Series(0.0, index=close_df.columns)
    portfolio_values = []
    trade_records = []
    weight_history = []
    peak_value = INITIAL_CAPITAL
    last_rebalance_date = None
    prev_vol = None

    for date in close_df.index:
        prices = close_df.loc[date]
        position_value = (holdings * prices).sum()
        portfolio_value = cash + position_value
        portfolio_values.append(portfolio_value)
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        current_drawdown = (peak_value - portfolio_value) / peak_value

        current_weights = holdings * prices / portfolio_value if portfolio_value > 0 else pd.Series(0.0, index=close_df.columns)

        # 风控检查
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        risk_actions = check_risk_controls(portfolio_value, peak_value, current_drawdown, daily_returns)

        if risk_actions.get("emergency_exit"):
            for code in holdings.index:
                if holdings[code] > 0:
                    trade_records.append({
                        "日期": date, "代码": code, "方向": "sell",
                        "金额": round(holdings[code] * prices[code], 2),
                        "价格": round(prices[code], 4), "成本": 0,
                    })
            cash = portfolio_value
            holdings[:] = 0
            continue

        # 检查是否需要再平衡
        should_rebalance = False
        is_month_start = (
            last_rebalance_date is None or
            (date.month != last_rebalance_date.month)
        )
        target_weights = weight_history[-1] if weight_history else pd.Series(1.0 / len(close_df.columns), index=close_df.columns)
        if is_month_start:
            from risk_parity import calc_etf_weights
            returns_up_to_date = return_df.loc[:date]
            if len(returns_up_to_date) >= 60:
                target_weights = calc_etf_weights(returns_up_to_date, etf_pool)
                should_rebalance = True
                last_rebalance_date = date
        if not should_rebalance and date in return_df.index:
            current_vol = return_df.loc[:date].tail(60).std().mean() * np.sqrt(252)
            if prev_vol is not None and prev_vol > 0:
                if abs(current_vol - prev_vol) / prev_vol > VOL_CHANGE_THRESHOLD:
                    should_rebalance = True
            prev_vol = current_vol

        if should_rebalance:
            trades = execute_rebalance(current_weights, target_weights, portfolio_value, prices)
            trade_records.extend(trades)
            for t in trades:
                delta = -t["金额"] if t["方向"] == "buy" else t["金额"]
                cash += delta - t["成本"]
                idx = close_df.columns.get_loc(t["代码"])
                shares_delta = t["金额"] / t["价格"] if t["价格"] > 0 else 0
                if t["方向"] == "buy":
                    holdings.iloc[idx] += shares_delta
                else:
                    holdings.iloc[idx] -= shares_delta

        weight_history.append(target_weights if should_rebalance else weight_history[-1] if weight_history else current_weights)

    return {
        "portfolio_values": pd.Series(portfolio_values, index=close_df.index),
        "trade_records": pd.DataFrame(trade_records) if trade_records else pd.DataFrame(),
        "weight_history": pd.DataFrame(weight_history, index=close_df.index) if weight_history else pd.DataFrame(),
    }
```

**注意：** 上面使用`EXECUTION_RATIO`需要从config导入，在backtest_engine.py顶部补充：

```python
from config import (
    EXECUTION_RATIO, SINGLE_ETF_MAX, TOTAL_COST_PER_SIDE,
    VOL_CHANGE_THRESHOLD, INITIAL_CAPITAL,
)
```

- [ ] **Step 4: 提交**

```bash
git add 2A股ETF模拟全天候策略/backtest_engine.py
git commit -m "feat: add backtest engine with rebalancing and risk controls"
```

---

### Task 5: 绩效分析 — performance.py

**Files:**
- Create: `2A股ETF模拟全天候策略/performance.py`

- [ ] **Step 1: 实现核心绩效指标计算**

```python
import numpy as np
import pandas as pd
from config import RISK_FREE_RATE, INITIAL_CAPITAL


def calc_metrics(equity_curve: pd.Series) -> dict:
    total_return = (equity_curve.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    years = max(days / 365.25, 1 / 365.25)
    annual_return = (1 + total_return) ** (1 / years) - 1
    daily_returns = equity_curve.pct_change().dropna()
    annual_vol = daily_returns.std() * np.sqrt(252)
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol if annual_vol > 0 else 0
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = drawdown.min()
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
    return {
        "总收益率": round(total_return * 100, 2),
        "年化收益率": round(annual_return * 100, 2),
        "年化波动率": round(annual_vol * 100, 2),
        "夏普比率": round(sharpe, 3),
        "最大回撤": round(max_dd * 100, 2),
        "卡玛比率": round(calmar, 3),
    }
```

- [ ] **Step 2: 实现基准组合模拟**

```python
def calc_benchmark_metrics(close_df: pd.DataFrame, weights: list[float],
                           codes: list[str], label: str) -> dict:
    port_returns = close_df[codes].pct_change().dropna()
    weighted_returns = port_returns.dot(weights)
    cum_returns = (1 + weighted_returns).cumprod()
    equity = cum_returns * INITIAL_CAPITAL
    return calc_metrics(equity)


def classic_aa_return(close_df: pd.DataFrame) -> pd.Series:
    codes = ["510300", "511260", "518880", "159985"]
    weights = [0.30, 0.55, 0.075, 0.075]
    return close_df[codes].pct_change().dropna().dot(weights)


def equal_weight_return(close_df: pd.DataFrame) -> pd.Series:
    return close_df.pct_change().dropna().mean(axis=1)
```

- [ ] **Step 3: 实现报告生成和可视化**

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_equity_curve(results: dict, benchmark_results: dict, save_path: str):
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    eq = results["portfolio_values"] / INITIAL_CAPITAL
    axes[0].plot(eq.index, eq, label="风险平价策略", linewidth=2)
    for label, data in benchmark_results.items():
        axes[0].plot(data.index, data / INITIAL_CAPITAL, label=label, alpha=0.7)
    axes[0].set_title("净值曲线对比")
    axes[0].legend()
    axes[0].grid(True)

    cummax = results["portfolio_values"].cummax()
    drawdown = (results["portfolio_values"] - cummax) / cummax * 100
    axes[1].fill_between(drawdown.index, 0, drawdown, color="red", alpha=0.3)
    axes[1].set_title("回撤曲线")
    axes[1].grid(True)

    if "weight_history" in results and not results["weight_history"].empty:
        results["weight_history"].plot(ax=axes[2])
        axes[2].set_title("权重变化")
        axes[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_report(strategy_metrics: dict, benchmark_metrics: dict, trades_df: pd.DataFrame):
    print("=" * 70)
    print("         A股ETF风险平价策略回测报告")
    print("=" * 70)
    print(f"\n【策略绩效】")
    for k, v in strategy_metrics.items():
        print(f"  {k}: {v}")
    print(f"\n【基准对比】")
    for label, metrics in benchmark_metrics.items():
        print(f"  {label}:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")
    if not trades_df.empty:
        total_cost = trades_df["成本"].sum()
        total_trades = len(trades_df)
        print(f"\n【交易统计】")
        print(f"  总交易次数: {total_trades}")
        print(f"  总交易成本: {round(total_cost, 2)}")
    print("=" * 70)
```

- [ ] **Step 4: 提交**

```bash
git add 2A股ETF模拟全天候策略/performance.py
git commit -m "feat: add performance analysis and reporting module"
```

---

### Task 6: 运行入口 — run_backtest.py

**Files:**
- Create: `2A股ETF模拟全天候策略/run_backtest.py`

- [ ] **Step 1: 实现主入口，编排全流程**

```python
#!/usr/bin/env python3
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from config import ETF_POOL, BENCHMARK_CODES, INITIAL_CAPITAL
from data import load_multi_etf, download_all_data
from backtest_engine import run_backtest
from performance import calc_metrics, print_report, plot_equity_curve
from risk_parity import calc_etf_weights

OUTPUT_DIR = "2A股ETF模拟全天候策略/output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Step 1: 下载数据...")
    download_all_data()
    all_codes = [e["code"] for e in ETF_POOL]
    print("Step 2: 加载数据...")
    close_df = load_multi_etf(all_codes)
    print(f"  数据范围: {close_df.index[0]} ~ {close_df.index[-1]}, {len(close_df)}天")
    print("Step 3: 运行回测...")
    results = run_backtest(close_df, ETF_POOL)
    strategy_metrics = calc_metrics(results["portfolio_values"])
    print("Step 4: 计算基准...")
    benchmark_results = {}
    hs300_close = load_multi_etf(BENCHMARK_CODES)
    hs300_eq = hs300_close.iloc[:, 0] / hs300_close.iloc[0, 0] * INITIAL_CAPITAL
    benchmark_results["沪深300"] = hs300_eq
    benchmark_metrics = {}
    benchmark_metrics["沪深300"] = calc_metrics(hs300_eq)
    print("Step 5: 生成报告...")
    print_report(strategy_metrics, benchmark_metrics, results["trade_records"])
    plot_equity_curve(
        results, benchmark_results,
        os.path.join(OUTPUT_DIR, "equity_curve.png")
    )
    results["portfolio_values"].to_csv(
        os.path.join(OUTPUT_DIR, "portfolio_values.csv")
    )
    if not results["trade_records"].empty:
        results["trade_records"].to_csv(
            os.path.join(OUTPUT_DIR, "trades.csv"), index=False
        )
    if not results["weight_history"].empty:
        results["weight_history"].to_csv(
            os.path.join(OUTPUT_DIR, "weight_history.csv")
        )
    print(f"结果已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行测试**

Run: `cd /home/jinqh/ETF && python -c "from config import *; print('config OK')"`
Expected: `config OK`

Run: `cd /home/jinqh/ETF && python -c "from risk_parity import *; print('risk_parity OK')"`
Expected: `risk_parity OK`

Run: `cd /home/jinqh/ETF && python 2A股ETF模拟全天候策略/run_backtest.py --dry-run`
说明：首次仅测试导入和配置，不完整运行。

- [ ] **Step 3: 提交**

```bash
git add 2A股ETF模拟全天候策略/run_backtest.py
git commit -m "feat: add run_backtest entry point"
```

---

### Task 7: 完善对比基准

**Files:**
- Modify: `2A股ETF模拟全天候策略/run_backtest.py`
- Modify: `2A股ETF模拟全天候策略/performance.py`

- [ ] **Step 1: 在回测入口中增加经典全天候和等权重基准**

在 `run_backtest.py` 的 main() 中增加：

```python
from performance import calc_classic_all_weather, calc_equal_weight

# 经典全天候 (30/55/7.5/7.5)
aw_equity = calc_classic_all_weather(close_df)
benchmark_results["经典全天候"] = aw_equity
benchmark_metrics["经典全天候"] = calc_metrics(aw_equity)

# 等权重
ew_equity = calc_equal_weight(close_df)
benchmark_results["等权重"] = ew_equity
benchmark_metrics["等权重"] = calc_metrics(ew_equity)
```

在 `performance.py` 中增加：

```python
def calc_classic_all_weather(close_df: pd.DataFrame) -> pd.Series:
    codes = ["510300", "511260", "518880", "159985"]
    overlap = [c for c in codes if c in close_df.columns]
    if not overlap:
        return pd.Series(dtype=float)
    weights = [0.30, 0.55, 0.075, 0.075]
    sub_weights = [weights[codes.index(c)] for c in overlap]
    sub_weights = np.array(sub_weights) / sum(sub_weights)
    port_returns = close_df[overlap].pct_change().dropna().dot(sub_weights)
    return (1 + port_returns).cumprod() * INITIAL_CAPITAL


def calc_equal_weight(close_df: pd.DataFrame) -> pd.Series:
    port_returns = close_df.pct_change().dropna().mean(axis=1)
    return (1 + port_returns).cumprod() * INITIAL_CAPITAL
```

- [ ] **Step 2: 验证完整回测**

Run: `cd /home/jinqh/ETF && python 2A股ETF模拟全天候策略/run_backtest.py`

Expected: 策略完成、输出报告、生成图表

- [ ] **Step 3: 提交**

```bash
git add 2A股ETF模拟全天候策略/
git commit -m "feat: add benchmark comparison for All Weather strategy"
```

---

### Task 8: 最终自审与清理

**Files:**
- All files in `2A股ETF模拟全天候策略/`

- [ ] **Step 1: 检查所有文件无硬编码路径、无 TODO、无占位符**

Run: `cd /home/jinqh/ETF && grep -n "TODO\|FIXME\|hardcoded\|/home/jintz\|/Users" 2A股ETF模拟全天候策略/*.py`

Expected: 无匹配

- [ ] **Step 2: 检查import引用一致性**

Run: `cd /home/jinqh/ETF && python -c "exec(open('2A股ETF模拟全天候策略/run_backtest.py').read().replace('if __name__', 'if False and __name__'))" 2>&1 | head -20`

Expected: 无ImportError

- [ ] **Step 3: 最终提交**

```bash
git add 2A股ETF模拟全天候策略/ docs/superpowers/
git commit -m "feat: complete All Weather risk parity strategy implementation"
```
