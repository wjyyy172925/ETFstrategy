"""
生成ETF资产日收益率DataFrame并保存为CSV
==========================================
根据 config.py 中定义的 ETF_POOL 标的，
从本地CSV或akshare下载净值数据，
计算日收益率（pct_change），并以CSV格式保存。
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ETF_POOL, BACKTEST_START, BACKTEST_END, OUTPUT_DIR
from data import load_and_align


def compute_daily_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    由价格矩阵计算日收益率矩阵（百分比形式）。

    日收益率 = (当日价格 - 前一日价格) / 前一日价格

    Args:
        price_df: DataFrame, index=date, columns=ETF代码, values=收盘价

    Returns:
        DataFrame, index=date, columns=ETF代码, values=日收益率（小数形式）
    """
    returns_df = price_df.pct_change()
    returns_df = returns_df.dropna(how="all")
    return returns_df


def main():
    print("=" * 60)
    print("ETF日收益率生成")
    print("=" * 60)

    codes = [etf["code"] for etf in ETF_POOL]
    names = {etf["code"]: etf["name"] for etf in ETF_POOL}

    print(f"\nETF标的池 ({len(codes)}只):")
    for etf in ETF_POOL:
        print(f"  {etf['code']} - {etf['name']} ({etf['asset_class']})")

    print(f"\n回测时间范围: {BACKTEST_START} ~ {BACKTEST_END}")

    # 1. 加载并对齐价格数据
    print("\n[1/3] 加载ETF净值数据...")
    price_df, codes = load_and_align(codes=codes, start=BACKTEST_START, end=BACKTEST_END)
    print(f"  价格矩阵形状: {price_df.shape}")
    print(f"  日期范围: {price_df.index[0].strftime('%Y-%m-%d')} ~ {price_df.index[-1].strftime('%Y-%m-%d')}")

    # 2. 计算日收益率
    print("\n[2/3] 计算日收益率...")
    returns_df = compute_daily_returns(price_df)
    print(f"  收益率矩阵形状: {returns_df.shape}")
    print(f"  日期范围: {returns_df.index[0].strftime('%Y-%m-%d')} ~ {returns_df.index[-1].strftime('%Y-%m-%d')}")

    # 3. 保存为CSV
    print("\n[3/3] 保存CSV文件...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3a. 保存完整的收益率矩阵（宽表，一列一个ETF）
    returns_csv = os.path.join(OUTPUT_DIR, "daily_returns.csv")
    returns_df.to_csv(returns_csv, encoding="utf-8-sig", float_format="%.6f")
    print(f"  收益率矩阵: {returns_csv}")

    # 3b. 保存每个ETF的单独收益率序列（方便单个查看）
    single_dir = os.path.join(OUTPUT_DIR, "daily_returns_single")
    os.makedirs(single_dir, exist_ok=True)
    for code in codes:
        name = names.get(code, code)
        single_df = returns_df[[code]].copy()
        single_df.columns = ["daily_return"]
        single_csv = os.path.join(single_dir, f"{code}_{name}_daily_return.csv")
        single_df.to_csv(single_csv, encoding="utf-8-sig", float_format="%.6f")
        print(f"  {code} ({name}): {single_csv}")

    # 3c. 保存价格数据（用于参考）
    price_csv = os.path.join(OUTPUT_DIR, "daily_prices.csv")
    price_df.to_csv(price_csv, encoding="utf-8-sig", float_format="%.4f")
    print(f"  价格矩阵（参考）: {price_csv}")

    # 4. 输出统计摘要
    print("\n" + "=" * 60)
    print("收益率统计摘要")
    print("=" * 60)
    stats = pd.DataFrame({
        "标的": [names.get(c, c) for c in codes],
        "代码": codes,
        "日均收益率(%)": [returns_df[c].mean() * 100 for c in codes],
        "年化收益率(%)": [returns_df[c].mean() * 252 * 100 for c in codes],
        "日波动率(%)": [returns_df[c].std() * 100 for c in codes],
        "年化波动率(%)": [returns_df[c].std() * np.sqrt(252) * 100 for c in codes],
        "夏普比率": [(returns_df[c].mean() / returns_df[c].std()) * np.sqrt(252) if returns_df[c].std() > 0 else 0 for c in codes],
        "最大日涨幅(%)": [returns_df[c].max() * 100 for c in codes],
        "最大日跌幅(%)": [returns_df[c].min() * 100 for c in codes],
        "有效交易日": [returns_df[c].notna().sum() for c in codes],
    })
    stats = stats.round(4)
    print(stats.to_string(index=False))

    stats_csv = os.path.join(OUTPUT_DIR, "daily_returns_stats.csv")
    stats.to_csv(stats_csv, encoding="utf-8-sig", index=False)
    print(f"\n统计摘要已保存: {stats_csv}")

    print("\n完成!")


if __name__ == "__main__":
    main()