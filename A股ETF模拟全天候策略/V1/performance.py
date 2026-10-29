"""
绩效分析模块
============
计算组合回测的全面绩效指标，并与多个基准策略进行对比。

核心功能：
  1. 计算6大类绩效指标：年化收益、年化波动、夏普比率、最大回撤、卡玛比率、胜率
  2. 与3个基准对比：沪深300、经典全天候（30-55-7.5-7.5）、等权重
  3. 生成3面板可视化：净值曲线、权重演变、绩效指标对比表

复利计算说明：
  所有年化指标均基于对数收益率的复利计算，以准确反映长期复合增长。
  日收益率 = ln(P_t / P_{t-1})
  年化收益率 = exp(mean(日收益率)*252) - 1
  年化波动率 = std(日收益率) * sqrt(252)
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASSET_CLASS_MAP,
    BENCHMARK_CODES,
    CLASSIC_AA_WEIGHTS,
    EQUAL_WEIGHT,
    ETF_POOL,
    INITIAL_CAPITAL,
    OUTPUT_DIR,
    RISK_FREE_RATE,
    TOTAL_COST_PER_SIDE,
)

# CLASSIC_AA_WEIGHTS 和 EQUAL_WEIGHT 按四大类定义：[权益, 债券, 黄金, 商品]
# 需映射到 ETF_POOL 中实际的 5 个 asset_class
_AA_GROUP_MAP = {
    0: ["equity"],               # 权益
    1: ["long_bond", "mid_bond"], # 债券（长久期+中短债）
    2: ["gold"],                  # 黄金
    3: ["commodity"],            # 商品
}

# ============================================================================
# 中文显示设置
# ============================================================================
# 尝试多种途径加载中文字体，确保图表标签正常显示
_CJK_FONTS = [
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "DejaVu Sans",
]
_loaded_font = None
for _f in _CJK_FONTS:
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        _loaded_font = _f
        break
    except Exception:
        continue

if _loaded_font:
    matplotlib.rcParams["font.family"] = _loaded_font
else:
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]

matplotlib.rcParams["axes.unicode_minus"] = False


# ============================================================================
# 绩效指标计算
# ============================================================================
def compute_all_metrics(
    equity_series: pd.Series,
    risk_free_rate: float = RISK_FREE_RATE,
) -> Dict[str, float]:
    """
    计算组合的全面绩效指标。

    共6个核心指标：
      1. 年化收益率（CAGR）：基于复利，几何平均
      2. 年化波动率：对数日收益率标准差年化
      3. 夏普比率：超额收益/波动率（年化）
      4. 最大回撤：峰值到谷底的最大跌幅
      5. 卡玛比率：年化收益/最大回撤
      6. 胜率：正收益日占比

    Args:
        equity_series: 组合净值序列（index=日期，values=净值）
        risk_free_rate: 无风险年化利率（默认2.5%）

    Returns:
        dict: {指标名: 数值}
    """
    # 日收益率（对数）
    returns = np.log(equity_series).diff().dropna()
    total_days = len(returns)

    if total_days < 2:
        return {
            "年化收益率": 0, "年化波动率": 0, "夏普比率": 0, "最大回撤": 0,
            "卡玛比率": 0, "胜率": 0,
            "最大回撤区间": "N/A",
        }

    # 1. 年化收益率
    total_return = equity_series.iloc[-1] / equity_series.iloc[0]
    years = total_days / 252
    cagr = total_return ** (1 / years) - 1 if years > 0 else 0

    # 2. 年化波动率
    ann_vol = returns.std() * np.sqrt(252)

    # 3. 夏普比率
    daily_rf = (1 + risk_free_rate) ** (1 / 252) - 1
    excess_returns = returns - daily_rf
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 1e-10 else 0

    # 4. 最大回撤
    peak = equity_series.expanding().max()
    drawdown = (equity_series - peak) / peak
    max_dd = drawdown.min()

    # 最大回撤区间：峰值日期 → 谷底日期
    max_dd_end_date = drawdown.idxmin()
    max_dd_start_date = equity_series[:max_dd_end_date].idxmax()
    max_dd_range = f"{max_dd_start_date.strftime('%Y-%m-%d')} ~ {max_dd_end_date.strftime('%Y-%m-%d')}"

    # 5. 卡玛比率
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-10 else 0

    # 6. 胜率
    win_rate = (returns > 0).sum() / total_days if total_days > 0 else 0

    return {
        "年化收益率": round(cagr, 4),
        "年化波动率": round(ann_vol, 4),
        "夏普比率": round(sharpe, 4),
        "最大回撤": round(float(max_dd), 4),
        "卡玛比率": round(calmar, 4),
        "胜率": round(win_rate, 4),
        "最大回撤区间": max_dd_range,
    }


# ============================================================================
# 基准策略构建
# ============================================================================
def build_benchmark_portfolios(
    price_df: pd.DataFrame,
    codes: list,
    initial_capital: float = INITIAL_CAPITAL,
    rf: float = RISK_FREE_RATE,
) -> Dict[str, Dict]:
    """
    构建3个对比基准策略的净值曲线和绩效指标。

    基准策略：
      1. 沪深300：仅持有沪深300ETF（单资产）
      2. 经典全天候：30%权益/55%债券/7.5%黄金/7.5%商品（固定权重）
      3. 等权重：所有ETF等权配置（1/N）

    Args:
        price_df: ETF价格矩阵
        codes: ETF代码列表
        initial_capital: 初始资金（默认1亿）
        rf: 无风险利率

    Returns:
        dict: {基准名称: {"equity": Series净值, "metrics": dict指标}}
    """
    n = len(codes)
    benchmarks = {}

    # ----- 1. 沪深300 -----
    # 如果沪深300在ETF池中，直接取对应列
    if "510300" in codes:
        idx = codes.index("510300")
        hs300_eq = price_df.iloc[:, idx] / price_df.iloc[0, idx] * initial_capital
    else:
        hs300_eq = pd.Series(index=price_df.index, data=initial_capital)

    benchmarks["沪深300"] = {
        "equity": hs300_eq,
        "metrics": compute_all_metrics(hs300_eq, rf),
    }

    # ----- 2. 经典全天候：30%权益/55%债券/7.5%黄金/7.5%商品 -----
    code_to_etf = {e["code"]: e for e in ETF_POOL}
    active_etfs = [code_to_etf[c] for c in codes if c in code_to_etf]

    classic_w = _aa_weights_to_etf_weights(CLASSIC_AA_WEIGHTS, active_etfs, n)
    classic_eq = _build_rebalanced_portfolio(price_df, classic_w, initial_capital)
    benchmarks["经典全天候"] = {
        "equity": classic_eq,
        "metrics": compute_all_metrics(classic_eq, rf),
    }

    # ----- 3. 等权重：四大类各25% -----
    equal_w = _aa_weights_to_etf_weights(EQUAL_WEIGHT, active_etfs, n)
    equal_eq = _build_rebalanced_portfolio(price_df, equal_w, initial_capital)
    benchmarks["等权重"] = {
        "equity": equal_eq,
        "metrics": compute_all_metrics(equal_eq, rf),
    }

    return benchmarks


def _aa_weights_to_etf_weights(
    aa_weights: list,
    active_etfs: list,
    n_codes: int,
) -> np.ndarray:
    """
    将四大类权重转换为每只 ETF 的权重。

    CLASSIC_AA_WEIGHTS / EQUAL_WEIGHT 格式：[权益, 债券, 黄金, 商品]
    债券大类下包含 long_bond + mid_bond，内部等分。
    """
    w = np.zeros(n_codes)
    for group_idx, asset_classes in _AA_GROUP_MAP.items():
        group_weight = aa_weights[group_idx]
        members = [i for i, etf in enumerate(active_etfs) if etf["asset_class"] in asset_classes]
        if members:
            per_member = group_weight / len(members)
            for i in members:
                w[i] = per_member
    return w


def _build_rebalanced_portfolio(
    price_df: pd.DataFrame,
    target_weights: np.ndarray,
    initial_capital: float,
) -> pd.Series:
    """
    构建月度再平衡的组合净值序列（含交易成本、整手100股取整）。

    每月第一个交易日按 target_weights 调仓。
    """
    n_assets = len(target_weights)
    n_dates = len(price_df)
    prices = price_df.values

    equity = np.full(n_dates, np.nan)
    cash = initial_capital
    shares = np.zeros(n_assets)

    # 首日建仓（全量买入）
    first_price = prices[0, :]
    for i in range(n_assets):
        target_value = initial_capital * target_weights[i]
        raw_shares = target_value / first_price[i]
        shares[i] = int(raw_shares / 100) * 100
        cash -= shares[i] * first_price[i]
    cash -= initial_capital * TOTAL_COST_PER_SIDE
    equity[0] = np.sum(shares * first_price) + cash

    prev_month = price_df.index[0].month

    for t in range(1, n_dates):
        prices_today = prices[t, :]
        current_month = price_df.index[t].month

        # 每日更新净值
        portfolio_value = np.sum(shares * prices_today) + cash
        equity[t] = portfolio_value

        # 月初再平衡
        if current_month != prev_month:
            old_values = shares * prices_today
            new_target_values = portfolio_value * target_weights
            turnover = np.sum(np.abs(new_target_values - old_values))
            trade_cost = turnover * TOTAL_COST_PER_SIDE
            available_capital = portfolio_value - trade_cost

            new_shares = np.zeros(n_assets)
            remaining_cash = available_capital
            for i in range(n_assets):
                target_value = available_capital * target_weights[i]
                raw_shares = target_value / prices_today[i]
                new_shares[i] = int(raw_shares / 100) * 100
                remaining_cash -= new_shares[i] * prices_today[i]

            shares = new_shares
            cash = remaining_cash

        prev_month = current_month

    return pd.Series(equity, index=price_df.index, name="benchmark")


# ============================================================================
# 绩效对比摘要表
# ============================================================================
def build_performance_summary(
    portfolio_metrics: Dict[str, float],
    benchmarks: Dict[str, Dict],
) -> pd.DataFrame:
    """
    将主策略和所有基准的绩效指标汇总为对比表格。

    Args:
        portfolio_metrics: 主策略绩效指标 dict
        benchmarks: 基准策略 dict {名称: {"metrics": dict}}

    Returns:
        DataFrame: 行=指标，列=策略，格式化为百分比
    """
    # 基础指标列表
    metric_names = ["年化收益率", "年化波动率", "夏普比率", "最大回撤", "卡玛比率", "胜率", "最大回撤区间"]
    # 格式函数：比率转百分比，夏普保留两位小数，区间直接透传
    fmt_funcs = {
        metric_names[0]: lambda x: f"{x:.2%}",
        metric_names[1]: lambda x: f"{x:.2%}",
        metric_names[2]: lambda x: f"{x:.2f}",
        metric_names[3]: lambda x: f"{x:.2%}",
        metric_names[4]: lambda x: f"{x:.2f}",
        metric_names[5]: lambda x: f"{x:.2%}",
        metric_names[6]: lambda x: str(x),
    }

    summary = {}
    summary["全天候(ERC)"] = {m: fmt_funcs[m](portfolio_metrics.get(m, 0)) for m in metric_names}

    for name, bm in benchmarks.items():
        summary[name] = {m: fmt_funcs[m](bm["metrics"].get(m, 0)) for m in metric_names}

    df = pd.DataFrame(summary)
    df.index.name = "指标"
    return df


# ============================================================================
# 可视化
# ============================================================================
def plot_results(
    portfolio_equity: pd.Series,
    portfolio_weights: pd.DataFrame,
    benchmarks: Dict[str, Dict],
    portfolio_metrics: Dict[str, float],
    output_dir: str = OUTPUT_DIR,
):
    """
    生成3面板可视化图表。

    面板1：净值曲线对比
        - 主策略 vs 沪深300 vs 经典全天候 vs 等权重
        - 对数坐标（便于观察长期复利增长）

    面板2：权重演变堆叠图
        - 按资产大类合并展示权重随时间变化
        - 每种颜色代表一个资产大类

    面板3：绩效指标对比表
        - 主策略与3个基准的指标数值对比

    Args:
        portfolio_equity: 主策略净值Series
        portfolio_weights: 主策略每日权重DataFrame
        benchmarks: 基准策略dict {名称: {"equity": Series, "metrics": dict}}
        portfolio_metrics: 主策略绩效指标
        output_dir: 图片保存目录
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))

    # ========== 面板1：净值曲线 ==========
    ax1 = axes[0]

    # 归一化到同一基准日（首日=1）
    norm_equity = portfolio_equity / portfolio_equity.iloc[0]

    ax1.plot(norm_equity.index, norm_equity.values, label="全天候(ERC)", linewidth=2.0, color="#1f77b4")

    colors = {"沪深300": "#d62728", "经典全天候": "#2ca02c", "等权重": "#ff7f0e"}
    for name, bm in benchmarks.items():
        bm_equity = bm["equity"]
        bm_norm = bm_equity / bm_equity.iloc[0]
        ax1.plot(
            bm_norm.index,
            bm_norm.values,
            label=name,
            linewidth=1.5,
            alpha=0.85,
            color=colors.get(name, "#9467bd"),
        )

    # 标注最终倍数
    y_pos = norm_equity.values[-1]
    ax1.annotate(
        f"{y_pos:.2f}x",
        xy=(norm_equity.index[-1], y_pos),
        xytext=(10, 0),
        textcoords="offset points",
        fontsize=9,
        color="#1f77b4",
    )

    ax1.set_yscale("log")
    ax1.set_title("净值曲线对比（对数坐标）", fontsize=13, fontweight="bold")
    ax1.set_ylabel("净值倍数")
    ax1.legend(loc="best", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    # 格式化x轴为年份
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # 在图上添加绩效指标文本
    metrics_text = (
        f"年化收益率: {portfolio_metrics.get('年化收益率', 0):.2%} | "
        f"年化波动: {portfolio_metrics.get('年化波动率', 0):.2%} | "
        f"夏普: {portfolio_metrics.get('夏普比率', 0):.2f}\n"
        f"最大回撤: {portfolio_metrics.get('最大回撤', 0):.2%} | "
        f"最大回撤区间: {portfolio_metrics.get('最大回撤区间', 'N/A')}\n"
        f"卡玛: {portfolio_metrics.get('卡玛比率', 0):.2f} | "
        f"胜率: {portfolio_metrics.get('胜率', 0):.2%}"
    )
    ax1.text(
        0.02, 0.02, metrics_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.2),
    )

    # ========== 面板2：权重演变 ==========
    ax2 = axes[1]

    # 按资产大类合并权重
    class_colors = {
        "equity": "#1f77b4",
        "long_bond": "#2ca02c",
        "mid_bond": "#98df8a",
        "gold": "#ffbb78",
        "commodity": "#ff7f0e",
    }
    class_labels = {ct: cfg["label"] for ct, cfg in ASSET_CLASS_MAP.items()}

    # 计算大类权重
    class_weight_series = {}
    for ct in ASSET_CLASS_MAP:
        ct_codes = [etf["code"] for etf in ETF_POOL if etf["asset_class"] == ct]
        ct_columns = [c for c in portfolio_weights.columns if c in ct_codes]
        if ct_columns:
            class_weight_series[ct] = portfolio_weights[ct_columns].sum(axis=1)

    class_df = pd.DataFrame(class_weight_series, index=portfolio_weights.index)
    class_df = class_df.fillna(0)

    class_df.plot(
        ax=ax2,
        kind="area",
        stacked=True,
        color=[class_colors.get(c, "#9467bd") for c in class_df.columns],
        alpha=0.85,
        legend=False,  # 自定义图例
    )

    ax2.set_title("组合权重演变（按资产大类）", fontsize=13, fontweight="bold")
    ax2.set_ylabel("权重")
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    # 自定义图例
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=class_colors.get(ct, "#9467bd"), alpha=0.85)
        for ct in class_df.columns
    ]
    labels = [class_labels.get(ct, ct) for ct in class_df.columns]
    ax2.legend(handles, labels, loc="best", framealpha=0.9)

    # ========== 面板3：指标对比表 ==========
    ax3 = axes[2]
    ax3.axis("tight")
    ax3.axis("off")

    summary = build_performance_summary(portfolio_metrics, benchmarks)
    table_data = summary.values
    col_labels = summary.columns.tolist()
    row_labels = summary.index.tolist()

    table = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc="center",
        rowLoc="center",
        loc="center",
        colColours=["#e6e6e6"] * len(col_labels),
        rowColours=["#f0f0f0"] * len(row_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    ax3.set_title("绩效指标对比", fontsize=13, fontweight="bold", pad=20)

    # ========== 保存 ==========
    plt.tight_layout(pad=3.0)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "performance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[performance] 图表已保存: {save_path}")