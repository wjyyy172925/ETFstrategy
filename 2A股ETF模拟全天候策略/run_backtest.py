#!/usr/bin/env python3
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ETF_POOL, BENCHMARK_CODES, INITIAL_CAPITAL
from data import load_multi_etf, download_all_data
from backtest_engine import run_backtest
from performance import calc_metrics, print_report, plot_equity_curve, calc_classic_all_weather, calc_equal_weight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Step 1: 下载/更新数据...")
    try:
        download_all_data()
    except Exception as e:
        print(f"数据下载失败: {e}")
        sys.exit(1)

    all_codes = [e["code"] for e in ETF_POOL]

    print("Step 2: 加载数据...")
    try:
        close_df = load_multi_etf(all_codes)
    except Exception as e:
        print(f"数据加载失败: {e}")
        sys.exit(1)
    if close_df.empty:
        print("错误: 无有效数据，请检查ETF数据文件")
        sys.exit(1)
    print(f"  数据范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}, {len(close_df)}个交易日")

    print("Step 3: 运行回测...")
    results = run_backtest(close_df, ETF_POOL)
    strategy_metrics = calc_metrics(results["portfolio_values"], INITIAL_CAPITAL)

    print("Step 4: 计算对比基准...")
    benchmark_results = {}
    benchmark_metrics = {}

    hs300_close = load_multi_etf(BENCHMARK_CODES)
    hs300_eq = hs300_close.iloc[:, 0] / hs300_close.iloc[0, 0] * INITIAL_CAPITAL
    benchmark_results["沪深300"] = hs300_eq
    benchmark_metrics["沪深300"] = calc_metrics(hs300_eq, INITIAL_CAPITAL)

    aw_eq = calc_classic_all_weather(close_df, INITIAL_CAPITAL)
    if len(aw_eq) > 0:
        benchmark_results["经典全天候(30/55/7.5/7.5)"] = aw_eq
        benchmark_metrics["经典全天候(30/55/7.5/7.5)"] = calc_metrics(aw_eq, INITIAL_CAPITAL)

    ew_eq = calc_equal_weight(close_df, INITIAL_CAPITAL)
    if len(ew_eq) > 0:
        benchmark_results["等权重"] = ew_eq
        benchmark_metrics["等权重"] = calc_metrics(ew_eq, INITIAL_CAPITAL)

    print("\nStep 5: 生成报告...")
    print_report(strategy_metrics, benchmark_metrics, results["trade_records"])

    plot_path = os.path.join(OUTPUT_DIR, "equity_curve.png")
    plot_equity_curve(results, benchmark_results, plot_path)
    print(f"净值曲线已保存: {plot_path}")

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
