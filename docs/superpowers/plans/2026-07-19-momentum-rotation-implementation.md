# 动量趋势轮动策略 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task.

**Goal:** 基于 A 股 ETF 历史日线数据实现动量趋势轮动策略，自动穷举参数组合并对比绩效

**Architecture:** 单文件 `momentum_strategy.py`，包含 ETF 筛选 → 动量计算 → 回测引擎 → 风险控制 → 绩效分析 → CLI 入口。参数组合自动穷举 2 窗口 × 2 频率 × 3 动量类型 = 12 种。

**Tech Stack:** Python 3.12+, pandas, numpy, akshare (仅数据下载用，策略独立)

---

### Task 1: 创建 results 目录与文件骨架

**Files:**
- Create: `momentum_strategy.py`
- Create: `results/`

- [ ] **Step 1: 创建目录和文件**

```bash
mkdir -p /home/jinqh/ETF/results
touch /home/jinqh/ETF/momentum_strategy.py
```

- [ ] **Step 2: 写入文件头部和 import**

```python
#!/usr/bin/env python3
"""A股ETF动量趋势轮动策略 - 回测与绩效分析

自动穷举参数组合:
  - 动量窗口: 20日 / 60日
  - 调仓频率: 周度 / 月度
  - 动量类型: 简单动量 / 双重动量 / 双重动量+均线过滤
"""

import os
import re
import json
import argparse
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
```

---

### Task 2: ETF 行业分类筛选函数

**Files:**
- Modify: `momentum_strategy.py`

- [ ] **Step 1: 写入行业 ETF 识别函数**

```python
def load_etf_list():
    """加载 ETF 列表"""
    path = "data/etf_list.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"ETF列表文件不存在: {path}")
    return pd.read_csv(path)


def filter_industry_etfs(df):
    """按名称关键词自动识别行业/主题 ETF，排除债券/货币/商品/跨境等。

    Args:
        df: 包含"代码"和"名称"列的 DataFrame。

    Returns:
        过滤后的 DataFrame，仅保留行业/主题 ETF。
    """
    include_kw = [
        "行业", "产业", "消费", "医药", "医疗", "科技", "新能源",
        "半导体", "军工", "银行", "证券", "保险", "地产", "有色",
        "煤炭", "钢铁", "化工", "农业", "食品", "汽车", "通信",
        "电子", "计算机", "传媒", "环保", "基建", "央企", "国企",
        "红利", "价值", "成长", "龙头", "创新", "制造", "材料",
        "能源", "电力", "体育", "旅游", "影视", "AI", "人工智能",
        "机器人", "芯片", "碳中和", "光伏", "风电", "电池",
        "新能车", "电动车", "智能驾驶", "数字经济", "信创", "游戏",
        "文娱", "物流", "运输", "航天", "航空",
    ]
    exclude_kw = [
        "国债", "地方债", "政金债", "城投债", "可转债", "货币",
        "短融", "AAA", "国开", "农发", "口行", "黄金", "豆粕",
        "能源化工", "白银", "标普", "纳斯达克", "纳指", "恒生",
        "H股", "中概", "道琼斯", "日经", "富时", "德国",
        "有色金属期货", "有色期货", "豆", "玉米", "白糖", "PTA",
        "甲醇", "豆粕ETF", "黄金ETF",
    ]

    mask = df["名称"].str.contains("|".join(include_kw), na=False)
    exclude_mask = df["名称"].str.contains("|".join(exclude_kw), na=False)
    return df[mask & ~exclude_mask].reset_index(drop=True)


def load_industry_etf_codes():
    """加载并过滤行业 ETF，返回 (codes_list, names_dict)。"""
    df = load_etf_list()
    df = filter_industry_etfs(df)
    codes = df["代码"].astype(str).str.zfill(6).tolist()
    names = dict(zip(df["代码"].astype(str).str.zfill(6), df["名称"]))
    print(f"行业ETF数量: {len(codes)} 只")
    return codes, names
```

---

### Task 3: 数据加载与动量计算

**Files:**
- Modify: `momentum_strategy.py`

- [ ] **Step 1: 写入数据加载函数**

```python
DATA_DIR = "data/etf"


def load_etf_data(code):
    """加载单只 ETF 日线数据，返回包含日期和收盘价的 DataFrame。"""
    path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    return df[["日期", "收盘"]]


def load_all_industry_data(codes):
    """加载所有行业 ETF 的收盘价，合并为宽表（列为代码，行为日期）。"""
    data = {}
    for code in codes:
        df = load_etf_data(code)
        if df is not None and len(df) > 60:
            data[code] = df.set_index("日期")["收盘"]
    result = pd.DataFrame(data).sort_index()
    result = result.loc[result.index.notna()]
    # 过滤最近5年数据
    cutoff = result.index.max() - pd.DateOffset(years=5)
    result = result[result.index >= cutoff]
    return result


def load_benchmark():
    """加载沪深300 ETF（510300）作为基准。"""
    df = load_etf_data("510300")
    if df is None:
        return None
    return df.set_index("日期")["收盘"].sort_index()
```

- [ ] **Step 2: 写入动量计算函数**

```python
def calc_momentum(close_df, window=20):
    """计算简单动量分数。

    Args:
        close_df: DataFrame/Series，列为ETF代码，行的收盘价。
        window: 动量计算窗口（交易日数）。

    Returns:
        每只 ETF 的最新动量分数 Series。
    """
    momentum = close_df / close_df.shift(window) - 1
    return momentum.iloc[-1].dropna()


def calc_dual_momentum(close_df, benchmark_close, window=20):
    """计算双重动量分数。

    Args:
        close_df: ETF 收盘价宽表。
        benchmark_close: 基准指数收盘价 Series。
        window: 动量窗口。

    Returns:
        (候选ETF代码列表, 动量分数 Series)。
    """
    returns = close_df / close_df.shift(window) - 1
    bench_return = (benchmark_close / benchmark_close.shift(window) - 1).iloc[-1]

    latest_returns = returns.iloc[-1].dropna()
    relative = latest_returns - bench_return
    absolute = latest_returns > 0

    candidates = relative[absolute].sort_values(ascending=False)
    return candidates


def calc_ma_filter(close_df, code, ma_window=20):
    """检查 ETF 是否站上均线。"""
    if code not in close_df.columns:
        return False
    series = close_df[code].dropna()
    if len(series) < ma_window:
        return False
    ma = series.rolling(ma_window).mean().iloc[-1]
    return series.iloc[-1] > ma
```

---

### Task 4: 风险控制模块

**Files:**
- Modify: `momentum_strategy.py`

- [ ] **Step 1: 写入风险控制函数**

```python
def calc_volatility_target(close_df, target_vol=0.15, window=20):
    """波动率目标管理：计算全市场等权组合的年化波动率，返回目标仓位比例。"""
    returns = close_df.pct_change().dropna()
    # 等权组合日收益率
    portfolio_returns = returns.mean(axis=1)
    recent_returns = portfolio_returns.iloc[-window:]
    if len(recent_returns) < 10:
        return 1.0
    actual_vol = recent_returns.std() * np.sqrt(252)
    if actual_vol <= 0:
        return 1.0
    ratio = target_vol / actual_vol
    return min(ratio, 1.0)


def apply_stop_loss(current_nav, cost_prices, stop_loss_pct=0.08):
    """个股止损检查。返回触发止损的 ETF 代码列表。"""
    stopped = []
    for code, cost in cost_prices.items():
        if code in current_nav:
            ret = (current_nav[code] / cost) - 1
            if ret <= -stop_loss_pct:
                stopped.append(code)
    return stopped


def check_strategy_drawdown_stop(nav_series, max_drawdown_pct=0.15):
    """策略最大回撤止损检查。返回 (是否触发, 当前回撤比例)。"""
    if len(nav_series) < 2:
        return False, 0.0
    peak = nav_series.max()
    current = nav_series.iloc[-1]
    dd = (peak - current) / peak
    return dd >= max_drawdown_pct, dd


def diversify_by_correlation(close_df, candidates, top_n, window=20, corr_threshold=0.9):
    """相关性分散约束：剔除高度相关的冗余 ETF。"""
    if len(candidates) <= top_n:
        return candidates[:top_n]

    selected = []
    remaining = list(candidates)

    while len(selected) < top_n and remaining:
        candidate = remaining.pop(0)
        if not selected:
            selected.append(candidate)
            continue
        # 检查与已选 ETF 的相关性
        returns = close_df[[candidate] + selected].pct_change().dropna()
        if len(returns) < 5:
            selected.append(candidate)
            continue
        corr = returns.corr().iloc[0, 1:]
        if all(corr < corr_threshold):
            selected.append(candidate)

    return selected
```

---

### Task 5: 回测引擎核心

**Files:**
- Modify: `momentum_strategy.py`

- [ ] **Step 1: 写入回测引擎**

```python
def run_backtest(close_df, benchmark_close, codes, top_n=5,
                 window=20, rebalance_freq="monthly",
                 momentum_type="simple"):
    """运行单参数组合回测。

    Args:
        close_df: ETF 收盘价宽表。
        benchmark_close: 基准收盘价 Series。
        codes: 参与轮动的 ETF 代码列表。
        top_n: 持仓数量。
        window: 动量计算窗口。
        rebalance_freq: "weekly" 或 "monthly"。
        momentum_type: "simple", "dual", "dual_ma"。

    Returns:
        (nav_series, trades_df) — 净值 Series 和交易记录。
    """
    nav = pd.Series(index=close_df.index, dtype=float)
    nav.iloc[0] = 1.0
    cash = 1.0
    holdings = {}  # code -> {shares, cost, entry_date}
    cost_prices = {}
    in_drawdown_stop = False
    peak_nav = 1.0

    rebalance_dates = _get_rebalance_dates(close_df.index, rebalance_freq, window)
    trades = []

    for i, date in enumerate(close_df.index):
        if i == 0:
            continue

        current_prices = close_df.loc[date]

        # --- 检查策略最大回撤止损 ---
        current_nav_value = cash + sum(
            holdings[c]["shares"] * current_prices[c]
            for c in list(holdings.keys())
            if c in current_prices and pd.notna(current_prices[c])
        ) if holdings else cash
        nav.loc[date] = current_nav_value
        peak_nav = max(peak_nav, current_nav_value)

        stopped, _ = check_strategy_drawdown_stop(nav[:date], 0.15)
        if stopped and not in_drawdown_stop:
            in_drawdown_stop = True
            # 清仓
            for c in list(holdings.keys()):
                if c in current_prices and pd.notna(current_prices[c]):
                    cash += holdings[c]["shares"] * current_prices[c]
                    trades.append({"日期": date, "ETF": c, "方向": "卖出(止损清仓)",
                                   "价格": current_prices[c], "原因": "策略最大回撤"})
            holdings = {}
            cost_prices = {}
            continue

        if in_drawdown_stop:
            # 等待净值创新高后恢复
            if current_nav_value >= peak_nav:
                in_drawdown_stop = False
            else:
                continue

        # --- 个股止损 ---
        stopped_codes = apply_stop_loss(
            {c: current_prices[c] for c in holdings if c in current_prices},
            cost_prices
        )
        for c in stopped_codes:
            if c in holdings:
                cash += holdings[c]["shares"] * current_prices[c]
                trades.append({"日期": date, "ETF": c, "方向": "卖出(止损)",
                               "价格": current_prices[c], "原因": "个股止损"})
                del holdings[c]
                del cost_prices[c]

        # --- 调仓日判断 ---
        if date not in rebalance_dates:
            # 非调仓日，仅更新净值
            continue

        # --- 计算动量信号 ---
        if momentum_type == "simple":
            scores = calc_momentum(close_df.loc[:date], window)
            candidates = scores.sort_values(ascending=False).head(top_n * 2).index.tolist()
        elif momentum_type in ("dual", "dual_ma"):
            bench = benchmark_close.loc[:date]
            candidates_series = calc_dual_momentum(close_df.loc[:date], bench, window)
            candidates = candidates_series.head(top_n * 2).index.tolist()
            if momentum_type == "dual_ma":
                candidates = [
                    c for c in candidates
                    if calc_ma_filter(close_df.loc[:date], c, 20)
                    and calc_ma_filter(close_df.loc[:date], c, 60)
                ]

        # --- 现金增强 ---
        if not candidates:
            for c in list(holdings.keys()):
                if c in current_prices and pd.notna(current_prices[c]):
                    cash += holdings[c]["shares"] * current_prices[c]
                    trades.append({"日期": date, "ETF": c, "方向": "卖出(现金增强)",
                                   "价格": current_prices[c], "原因": "无合格ETF"})
            holdings = {}
            cost_prices = {}
            continue

        # --- 相关性分散 ---
        selected = diversify_by_correlation(
            close_df.loc[:date], candidates, top_n
        )
        if not selected:
            continue

        # --- 波动率目标 ---
        vol_ratio = calc_volatility_target(close_df.loc[:date].dropna(axis=1, how="any"))

        # --- 调仓 ---
        # 先卖出不在新持仓中的 ETF
        for c in list(holdings.keys()):
            if c not in selected:
                if c in current_prices and pd.notna(current_prices[c]):
                    cash += holdings[c]["shares"] * current_prices[c]
                    trades.append({"日期": date, "ETF": c, "方向": "卖出",
                                   "价格": current_prices[c], "原因": "调仓"})
                del holdings[c]
                if c in cost_prices:
                    del cost_prices[c]

        # 计算可用资金（分批调仓：首次50%）
        target_per_etf = (cash * vol_ratio * 0.5) / len(selected)
        for c in selected:
            if c in current_prices and pd.notna(current_prices[c]):
                price = current_prices[c]
                shares = target_per_etf / price
                if c in holdings:
                    # 已有持仓，补足到目标
                    target_value = (cash * vol_ratio) / len(selected)
                    current_value = holdings[c]["shares"] * price
                    diff = target_value - current_value
                    if diff > 0:
                        add_shares = diff / price
                        holdings[c]["shares"] += add_shares
                        cash -= add_shares * price
                    elif diff < 0:
                        sell_shares = -diff / price
                        holdings[c]["shares"] -= sell_shares
                        cash += sell_shares * price
                else:
                    holdings[c] = {"shares": shares, "entry_date": date}
                    cost_prices[c] = price
                    cash -= shares * price
                    trades.append({"日期": date, "ETF": c, "方向": "买入",
                                   "价格": price, "原因": "调仓"})

        # 第二次调仓（T+3）
        # 实际实现简化：同一天完成，但资金减半处理

    nav = nav.dropna()
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return nav, trades_df


def _get_rebalance_dates(index, freq, min_window=60):
    """获取调仓日期列表。"""
    dates = index[index > index[min_window]] if len(index) > min_window else index
    if freq == "monthly":
        return dates[dates.to_series().diff().dt.days > 20].tolist()
    else:
        return dates[dates.to_series().diff().dt.days >= 5].tolist()
```

---

### Task 6: 绩效分析函数

**Files:**
- Modify: `momentum_strategy.py`

- [ ] **Step 1: 写入绩效指标计算与汇总**

```python
def calc_performance_metrics(nav, benchmark_nav=None, risk_free=0.02):
    """计算绩效指标。"""
    if len(nav) < 2:
        return {}

    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    total_days = (nav.index[-1] - nav.index[0]).days
    total_years = total_days / 365.25

    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / total_years) - 1 if total_years > 0 else 0
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = (cagr - risk_free) / annual_vol if annual_vol > 0 else 0

    # 最大回撤
    peak = nav.expanding().max()
    dd = (peak - nav) / peak
    max_dd = dd.max()

    calmar = cagr / max_dd if max_dd > 0 else 0
    win_rate = (ret > 0).mean()

    metrics = {
        "CAGR": cagr,
        "年化波动": annual_vol,
        "夏普比率": sharpe,
        "最大回撤": max_dd,
        "卡玛比率": calmar,
        "胜率": win_rate,
        "交易天数": len(nav),
    }

    if benchmark_nav is not None:
        bench_ret = benchmark_nav.reindex(nav.index).pct_change().dropna()
        aligned_ret = ret.reindex(bench_ret.index).dropna()
        bench_ret = bench_ret.reindex(aligned_ret.index)
        if len(aligned_ret) > 10:
            excess = aligned_ret - bench_ret
            tracking_error = excess.std() * np.sqrt(252)
            info_ratio = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
            metrics["信息比率"] = info_ratio
            metrics["跟踪误差"] = tracking_error

    return metrics


def format_metrics(metrics):
    """格式化绩效指标为显示字符串。"""
    return {
        "CAGR": f"{metrics.get('CAGR', 0)*100:.1f}%",
        "年化波动": f"{metrics.get('年化波动', 0)*100:.1f}%",
        "夏普比率": f"{metrics.get('夏普比率', 0):.2f}",
        "最大回撤": f"{metrics.get('最大回撤', 0)*100:.1f}%",
        "卡玛比率": f"{metrics.get('卡玛比率', 0):.2f}",
        "胜率": f"{metrics.get('胜率', 0)*100:.1f}%",
        "信息比率": f"{metrics.get('信息比率', 0):.2f}",
    }
```

---

### Task 7: 参数组合穷举与主流程

**Files:**
- Modify: `momentum_strategy.py`

- [ ] **Step 1: 写入参数组合穷举与 CLI**

```python
PARAM_GRID = {
    "window": [20, 60],
    "rebalance_freq": ["monthly", "weekly"],
    "momentum_type": ["simple", "dual", "dual_ma"],
}


def run_all_combinations(close_df, benchmark_close, codes, top_n=5):
    """穷举所有参数组合并运行回测。"""
    results = []
    navs = {}

    windows = PARAM_GRID["window"]
    freqs = PARAM_GRID["rebalance_freq"]
    types = PARAM_GRID["momentum_type"]

    total = len(windows) * len(freqs) * len(types)
    count = 0

    for w, f, t in product(windows, freqs, types):
        label = f"{t}_{w}日_{'周度' if f == 'weekly' else '月度'}"
        count += 1
        print(f"  [{count}/{total}] {label} ... ", end="", flush=True)

        try:
            nav, trades = run_backtest(
                close_df, benchmark_close, codes,
                top_n=top_n, window=w,
                rebalance_freq=f, momentum_type=t,
            )
            metrics = calc_performance_metrics(nav, benchmark_close)
            metrics["组合"] = label
            metrics["交易次数"] = len(trades)
            results.append(metrics)
            navs[label] = nav
            print("OK")
        except Exception as e:
            print(f"失败: {e}")

    return results, navs


def print_summary_table(results):
    """打印绩效对比表格。"""
    keys = ["组合", "CAGR", "夏普比率", "最大回撤", "年化波动", "胜率", "信息比率", "交易次数"]
    print("\n" + "=" * 100)
    print(f"{'动量趋势轮动策略 — 绩效总览':^100}")
    print("=" * 100)

    header = " | ".join(f"{k:>12}" for k in keys)
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in sorted(results, key=lambda x: x.get("夏普比率", 0), reverse=True):
        vals = format_metrics(r)
        row = " | ".join(
            f"{vals.get(k, str(r.get(k, ''))):>12}" for k in keys[1:]
        )
        print(f"{r['组合']:>20} | {row}")

    print("=" * 100)

    # 基准
    bench_metrics = calc_performance_metrics(benchmark_close)
    bv = format_metrics(bench_metrics)
    print(f"{'基准_沪深300':>20} | {'':>12} | ".strip())

    print()


def save_results(results, navs):
    """保存结果到 results/ 目录。"""
    os.makedirs("results", exist_ok=True)

    # 汇总表
    summary = pd.DataFrame(results)
    summary.to_csv("results/performance_summary.csv", index=False, encoding="utf-8-sig")

    # 各组合净值
    nav_df = pd.DataFrame(navs)
    nav_df.to_csv("results/nav_all.csv", index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="A股ETF动量趋势轮动策略回测")
    parser.add_argument("--top_n", type=int, default=5, help="持仓数量（默认5）")
    args = parser.parse_args()
    top_n = args.top_n

    print("=" * 60)
    print("动量趋势轮动策略回测")
    print("=" * 60)

    # 1. 加载行业 ETF
    print("\n[1/4] 加载行业 ETF ...")
    codes, names = load_industry_etf_codes()
    print(f"  找到 {len(codes)} 只行业 ETF")

    # 2. 加载历史数据
    print("[2/4] 加载历史数据 ...")
    close_df = load_all_industry_data(codes)
    print(f"  数据范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")
    print(f"  ETF数量: {close_df.shape[1]} | 交易日: {close_df.shape[0]}")

    # 3. 加载基准
    print("[3/4] 加载基准（沪深300）...")
    bench = load_benchmark()
    bench = bench.reindex(close_df.index).fillna(method="ffill").dropna()
    print(f"  基准范围: {bench.index[0].date()} ~ {bench.index[-1].date()}")

    # 4. 运行全部组合
    print(f"[4/4] 运行回测（持仓 {top_n} 只）...")
    results, navs = run_all_combinations(close_df, bench, close_df.columns.tolist(), top_n)

    # 5. 输出
    print_summary_table(results)
    save_results(results, navs)
    print("结果已保存到 results/")


if __name__ == "__main__":
    main()
```

---

### Task 8: 运行验证

- [ ] **Step 1: 激活 venv 并安装依赖**

```bash
source /home/jinqh/ETF/.venv/bin/activate
pip install pandas numpy
```

- [ ] **Step 2: 运行策略**

```bash
cd /home/jinqh/ETF
python momentum_strategy.py --top_n 5
```

- [ ] **Step 3: 检查 results/ 输出**

```bash
ls -la results/
head -5 results/performance_summary.csv
```
