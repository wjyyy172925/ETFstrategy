"""
ETF标的描述性统计
==================
对 V1 策略中每只 ETF 标的计算完整的描述性统计指标：
  - 年化收益率（对数复利）
  - 年化波动率
  - 最大回撤（含回撤区间）
  - 夏普比率（年化）
  - 胜率（正收益日占比）
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ETF_POOL, OUTPUT_DIR, RISK_FREE_RATE


def compute_max_drawdown(prices: pd.Series) -> tuple:
    """计算最大回撤及回撤区间。"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    max_dd = drawdown.min()
    end_date = drawdown.idxmin()
    start_date = prices[:end_date].idxmax()
    return float(max_dd), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")


def compute_etf_stats(price_df: pd.DataFrame, rf: float = RISK_FREE_RATE) -> pd.DataFrame:
    """对每只 ETF 计算完整描述性统计。"""
    codes = list(price_df.columns)
    name_map = {etf["code"]: etf["name"] for etf in ETF_POOL}

    rows = []
    for code in codes:
        prices = price_df[code].dropna()
        returns = np.log(prices).diff().dropna()
        n = len(returns)

        if n < 2:
            continue

        # 年化收益率（对数复利 CAGR）
        total_return = prices.iloc[-1] / prices.iloc[0]
        years = n / 252
        cagr = total_return ** (1 / years) - 1 if years > 0 else 0

        # 年化波动率
        ann_vol = float(returns.std() * np.sqrt(252))

        # 最大回撤
        max_dd, dd_start, dd_end = compute_max_drawdown(prices)

        # 夏普比率（年化）
        daily_rf = (1 + rf) ** (1 / 252) - 1
        excess = returns - daily_rf
        sharpe = float(np.sqrt(252) * excess.mean() / returns.std()) if returns.std() > 1e-10 else 0.0

        # 胜率（日收益率 > 0 的交易日占比）
        win_rate = float((returns > 0).sum() / n)

        # 累计总收益率
        cum_return = total_return - 1

        rows.append({
            "标的": name_map.get(code, code),
            "代码": code,
            "起始日期": prices.index[0].strftime("%Y-%m-%d"),
            "结束日期": prices.index[-1].strftime("%Y-%m-%d"),
            "有效交易日": n,
            "累计总收益率(%)": round(cum_return * 100, 2),
            "年化收益率(%)": round(cagr * 100, 2),
            "年化波动率(%)": round(ann_vol * 100, 2),
            "最大回撤(%)": round(max_dd * 100, 2),
            "最大回撤起始": dd_start,
            "最大回撤结束": dd_end,
            "夏普比率": round(sharpe, 4),
            "胜率(%)": round(win_rate * 100, 2),
        })

    return pd.DataFrame(rows)


def main():
    price_csv = os.path.join(OUTPUT_DIR, "daily_prices.csv")
    if not os.path.exists(price_csv):
        print(f"[错误] 价格文件不存在: {price_csv}")
        print("请先运行 generate_daily_returns.py 或 run_backtest.py 生成数据。")
        return

    price_df = pd.read_csv(price_csv, encoding="utf-8-sig", index_col=0, parse_dates=True)
    print(f"价格矩阵形状: {price_df.shape}")
    print(f"日期范围: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")
    print()

    stats_df = compute_etf_stats(price_df, rf=RISK_FREE_RATE)

    print("=" * 80)
    print("ETF 标的描述性统计")
    print("=" * 80)
    print(stats_df.to_string(index=False))
    print()

    # 保存为 CSV
    output_csv = os.path.join(OUTPUT_DIR, "etf_descriptive_stats.csv")
    stats_df.to_csv(output_csv, encoding="utf-8-sig", index=False)
    print(f"结果已保存: {output_csv}")


if __name__ == "__main__":
    main()