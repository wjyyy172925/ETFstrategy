import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RISK_FREE_RATE

plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei",
                                   "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def calc_metrics(equity_curve: pd.Series,
                 initial_capital: float) -> dict:
    if len(equity_curve) < 2:
        return {
            "总收益率": 0.0,
            "年化收益率": 0.0,
            "年化波动率": 0.0,
            "夏普比率": 0.0,
            "最大回撤": 0.0,
            "卡玛比率": 0.0,
        }
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
    years = days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1
    daily_returns = equity_curve.pct_change().dropna()
    annual_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
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


def calc_classic_all_weather(close_df: pd.DataFrame,
                              initial_capital: float) -> pd.Series:
    codes = ["510300", "511260", "518880", "159985"]
    weights = [0.30, 0.55, 0.075, 0.075]
    overlap = [c for c in codes if c in close_df.columns]
    if not overlap:
        return pd.Series(dtype=float)
    sub_weights = [weights[codes.index(c)] for c in overlap]
    sub_weights = np.array(sub_weights) / sum(sub_weights)
    port_returns = close_df[overlap].pct_change().dropna().dot(sub_weights)
    return (1 + port_returns).cumprod() * initial_capital


def calc_equal_weight(close_df: pd.DataFrame,
                       initial_capital: float) -> pd.Series:
    if close_df.empty or len(close_df.columns) == 0:
        return pd.Series(dtype=float)
    port_returns = close_df.pct_change().dropna().mean(axis=1)
    return (1 + port_returns).cumprod() * initial_capital


def plot_equity_curve(results: dict,
                      benchmark_results: dict[str, pd.Series],
                      save_path: str):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    eq = results["portfolio_values"]
    norm = eq / eq.iloc[0]
    axes[0].plot(norm.index, norm, label="风险平价策略", linewidth=2)
    for label, data in benchmark_results.items():
        if len(data) > 0:
            norm_b = data / data.iloc[0]
            axes[0].plot(norm_b.index, norm_b, label=label, alpha=0.7)
    axes[0].set_title("净值曲线对比（归一化）")
    axes[0].legend()
    axes[0].grid(True)

    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax * 100
    axes[1].fill_between(drawdown.index, 0, drawdown.values, color="red", alpha=0.3)
    axes[1].set_title("策略回撤曲线 (%)")
    axes[1].grid(True)

    if "weight_history" in results and not results["weight_history"].empty:
        wh = results["weight_history"]
        wh.plot(ax=axes[2])
        axes[2].set_title("权重变化")
        axes[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_report(strategy_metrics: dict,
                 benchmark_metrics: dict[str, dict],
                 trades_df: pd.DataFrame):
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
    if not trades_df.empty and {"成本", "方向"}.issubset(trades_df.columns):
        total_cost = trades_df["成本"].sum()
        total_trades = len(trades_df)
        buy_trades = len(trades_df[trades_df["方向"] == "buy"])
        sell_trades = len(trades_df[trades_df["方向"] == "sell"])
        print(f"\n【交易统计】")
        print(f"  总交易次数: {total_trades} (买入: {buy_trades}, 卖出: {sell_trades})")
        print(f"  总交易成本: {round(total_cost, 2)}")
    print("=" * 70)
