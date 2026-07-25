"""
回测运行入口
============
全天候风险平价策略回测的全流程编排脚本。

执行流程（5步）：
  1. 加载并对齐ETF数据 → data.load_and_align()
  2. 运行回测引擎 → BacktestEngine().run()
  3. 计算绩效指标 → compute_all_metrics()
  4. 构建对比基准 → build_benchmark_portfolios()
  5. 生成报告和图表 → plot_results() + 打印绩效摘要

使用方法：
    python run_backtest.py

预期输出：
  - 控制台：回测流程日志 + 绩效指标表格
  - output/logs/backtest.log：完整回测日志
  - output/performance.png：3面板可视化图表
  - output/weights.csv：每日权重变化（可选）

依赖：
  - numpy, pandas, matplotlib, scipy
  - akshare（如需从网络下载数据）
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 确保能在项目根目录下正确导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ETF_POOL, INITIAL_CAPITAL, OUTPUT_DIR, RISK_FREE_RATE
from data import load_and_align
from backtest_engine import BacktestEngine
from performance import (
    build_benchmark_portfolios,
    build_performance_summary,
    compute_all_metrics,
    plot_results,
)
from logger import setup_logger


# ============================================================================
# 主函数
# ============================================================================
def main():
    """
    回测主流程。

    执行步骤：
      1. 数据加载：从CSV读取ETF净值，多ETF日期对齐
      2. 回测运行：逐日模拟，三触发再平衡，三层风控
      3. 绩效计算：6大指标
      4. 基准构建：沪深300/经典全天候/等权重
      5. 结果展示：打印绩效表 + 生成图表
    """
    logger = setup_logger("run_backtest")

    logger.info("=" * 60)
    logger.info("全天候风险平价策略回测 - 启动")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"初始资金: {INITIAL_CAPITAL:,.0f}")
    logger.info("=" * 60)

    # ================================================================
    # 步骤1：加载并对齐ETF数据
    # ================================================================
    logger.info("[Step 1/5] 加载ETF数据...")
    codes = [etf["code"] for etf in ETF_POOL]
    price_df, codes = load_and_align(codes=codes)

    if price_df.empty:
        logger.error("数据加载为空，请检查CSV文件或网络连接")
        return

    logger.info(f"数据加载完成: {price_df.shape[0]}天 × {len(codes)}只ETF")
    logger.info(f"时间范围: {price_df.index[0].date()} ~ {price_df.index[-1].date()}")

    # ================================================================
    # 步骤2：运行回测
    # ================================================================
    logger.info("[Step 2/5] 运行回测引擎...")
    engine = BacktestEngine(price_df, codes, initial_capital=INITIAL_CAPITAL)
    result = engine.run()

    equity_curve = result["equity_curve"]
    weights_df = result["weights"]
    n_trades = result["n_trades"]

    logger.info(f"回测完成: {n_trades}笔交易 | 最终净值: {equity_curve.iloc[-1]:,.0f}")

    # ================================================================
    # 步骤3：计算绩效指标
    # ================================================================
    logger.info("[Step 3/5] 计算绩效指标...")
    portfolio_metrics = compute_all_metrics(equity_curve, RISK_FREE_RATE)

    logger.info("主策略绩效:")
    for name, val in portfolio_metrics.items():
        if "比率" in name:
            logger.info(f"  {name}: {val:.2f}")
        else:
            logger.info(f"  {name}: {val:.2%}")

    # ================================================================
    # 步骤4：构建对比基准
    # ================================================================
    logger.info("[Step 4/5] 构建对比基准组合...")
    benchmarks = build_benchmark_portfolios(price_df, codes, INITIAL_CAPITAL, RISK_FREE_RATE)

    for bm_name, bm_data in benchmarks.items():
        bm_metrics = bm_data["metrics"]
        logger.info(f"  {bm_name}: 年化{bm_metrics['年化收益率']:.2%} | "
                    f"夏普{bm_metrics['夏普比率']:.2f} | "
                    f"最大回撤{bm_metrics['最大回撤']:.2%}")

    # ================================================================
    # 步骤5：展示结果
    # ================================================================
    logger.info("[Step 5/5] 生成绩效报告...")

    # 打印绩效对比表
    summary_df = build_performance_summary(portfolio_metrics, benchmarks)
    print("\n" + "=" * 65)
    print("                     绩效指标对比表")
    print("=" * 65)
    print(summary_df.to_string())
    print("=" * 65 + "\n")

    # 生成图表
    plot_results(equity_curve, weights_df, benchmarks, portfolio_metrics, OUTPUT_DIR)

    # 保存权重变化到CSV（可选）
    weights_path = os.path.join(OUTPUT_DIR, "weights.csv")
    weights_df.to_csv(weights_path)
    logger.info(f"权重历史已保存: {weights_path}")

    # 保存净值曲线到CSV（可选）
    nav_path = os.path.join(OUTPUT_DIR, "nav.csv")
    equity_curve.to_csv(nav_path)
    logger.info(f"净值曲线已保存: {nav_path}")

    # ================================================================
    # 结束
    # ================================================================
    logger.info("=" * 60)
    logger.info("回测完成")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"图表: {os.path.join(OUTPUT_DIR, 'performance.png')}")
    logger.info("=" * 60)


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    main()
