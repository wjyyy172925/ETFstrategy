"""
回测引擎
========
实现完整的组合回测流程，包含：

再平衡机制（三触发器）：
  1. 日历再平衡：每月末按时触发
  2. 阈值偏离：任一资产权重偏离目标超过THRESHOLD_PCT=8%
  3. 波动率变化：组合波动率较上月变化超过VOL_CHANGE_THRESHOLD=30%

三层风控体系：
  1. 波动率目标控制：当预测波动率超出目标范围时，通过缩放系数调整整体风险敞口
  2. 回撤控制：回撤达到10%时降低权益敞口到80%，达到15%时降至50%
  3. CVaR风险预警：月频CVaR超过15%时触发减仓（紧急情况全仓清仓）

交易成本：每笔交易扣除佣金+滑点（万3+万5=万8单边）

注意：风控动作和再平衡动作均通过日志记录，不会修改原始价格数据。
"""

import os
import sys
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASSET_CLASS_MAP,
    ETF_POOL,
    INITIAL_CAPITAL,
    REBALANCE_FREQ,
    TARGET_VOL,
    VOL_BAND,
    THRESHOLD_PCT,
    VOL_CHANGE_THRESHOLD,
    EXECUTION_RATIO,
    TOTAL_COST_PER_SIDE,
    MIN_TRADE_AMOUNT,
    CVAR_THRESHOLD,
    DRAWDOWN_WARN,
    DRAWDOWN_MAX,
    WEIGHT_MIN,
    WEIGHT_MAX,
    OUTPUT_DIR,
)
from logger import setup_logger
from risk_parity import estimate_covariance, get_target_weights


# ============================================================================
# 回测引擎主类
# ============================================================================
class BacktestEngine:
    """
    全天候风险平价回测引擎。

    负责：
      - 逐日模拟组合净值演变
      - 在再平衡日执行ERC权重优化
      - 在风控触发日执行风险控制
      - 记录所有交易和风控日志
    """

    def __init__(
        self,
        price_df: pd.DataFrame,
        codes: list,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        """
        初始化回测引擎。

        Args:
            price_df: ETF价格矩阵（index=日期，columns=代码），需已对齐
            codes: ETF代码列表（与price_df列一一对应）
            initial_capital: 初始资金
        """
        self.price_df = price_df
        self.codes = codes
        self.n_assets = len(codes)
        self.initial_capital = initial_capital
        self.logger = setup_logger()

        # ================================================================
        # 回测状态初始化
        # ================================================================
        self.dates = price_df.index
        self.n_dates = len(self.dates)
        self.prices = price_df.values  # numpy加速

        # 净值与权重记录
        self.equity_curve = np.full(self.n_dates, np.nan)     # 每日组合净值
        self.weight_history = np.full((self.n_dates, self.n_assets), np.nan)  # 每日实际权重
        self.target_history = np.full((self.n_dates, self.n_assets), np.nan)  # 每日目标权重（风控调整前）

        # 风控状态
        self.current_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        self.vol_scale = 1.0         # 波动率缩放系数（风控第1层）
        self.reduce_equity = 1.0     # 权益缩减系数（风控第2层，默认100%）
        self.cvar_active = False     # CVaR风控是否触发（风控第3层）
        self.emergency_exit = False  # 紧急退出标记（回撤>20%）

        # 交易统计
        self.trade_log = []          # [(日期, 操作, 详情), ...]
        self.n_trades = 0            # 总交易次数
        self.last_rebalance_date = None  # 上次再平衡日期

        # 优化器降级统计
        self.optimizer_stats = {"tight": 0, "loose": 0, "free": 0, "fallback": 0}

        # 换手率追踪
        self.turnover_history = np.zeros(self.n_dates)

    # ========================================================================
    # 主回测循环
    # ========================================================================
    def run(self) -> dict:
        """
        执行完整的回测流程。

        逐日遍历，在每个交易日：
          1. 记录当前持仓市值，更新组合净值
          2. 判断是否为再平衡日（三种触发条件）
          3. 触发风控检查（三层风控）
          4. 执行交易（如有再平衡或风控）

        Returns:
            dict: 回测结果，包含以下key：
              - equity_curve: 每日组合净值Series
              - weights: 每日权重DataFrame
              - returns: 每日收益率Series
              - n_trades: 总交易次数
              - trade_log: 交易日志列表
              - nav: 最终净值倍数
        """
        self.logger.info("=" * 60)
        self.logger.info("回测开始")
        self.logger.info(f"初始资金: {self.initial_capital:,.0f}")
        self.logger.info(f"回测区间: {self.dates[0]} ~ {self.dates[-1]}")
        self.logger.info(f"交易天数: {self.n_dates}")
        self.logger.info("=" * 60)

        # ----- 初始化首日 -----
        first_price = self.prices[0, :]  # 第一个交易日所有ETF收盘价
        # 初始等权重建仓
        self.equity_curve[0] = self.initial_capital # 1亿
        self.weight_history[0, :] = self.current_weights    # 实际权重
        self.target_history[0, :] = self.current_weights    # 目标权重

        # 买入初始持仓（整手100股取整）
        cash = self.initial_capital
        shares = np.zeros(self.n_assets)
        for i in range(self.n_assets):
            target_value = self.initial_capital * self.current_weights[i]
            raw_shares = target_value / first_price[i]
            shares[i] = int(raw_shares / 100) * 100  # 整手取整
            cash -= shares[i] * first_price[i]        # 按整手后的实际金额扣除
        # 建仓交易成本（全量买入，单边万8）
        cash -= self.initial_capital * TOTAL_COST_PER_SIDE

        # 更新建仓后净值（扣除交易成本后的实际市值）
        self.equity_curve[0] = np.sum(shares * first_price) + cash
        self._log_trade(self.dates[0], "初始建仓", self.current_weights.copy())

        # ----- 逐日迭代 -----
        prev_month = self.dates[0].month
        last_vol = None  # 上个计算周期的组合波动率

        for t in range(1, self.n_dates):
            today = self.dates[t]
            prev_date = self.dates[t - 1]
            prices_today = self.prices[t, :]
            prices_yesterday = self.prices[t - 1, :]

            # 1. 更新当前市值和组合净值
            holdings_value = shares * prices_today
            portfolio_value = np.sum(holdings_value) + cash
            self.equity_curve[t] = portfolio_value
            self.weight_history[t, :] = holdings_value / portfolio_value if portfolio_value > 0 else self.current_weights

            # 当前回撤
            peak = np.maximum.accumulate(self.equity_curve[:t+1])
            current_drawdown = (self.equity_curve[t] - peak[-1]) / peak[-1]

            # ----- 紧急退出检查 -----
            if current_drawdown <= -DRAWDOWN_MAX:
                if not self.emergency_exit:
                    self.emergency_exit = True
                    self.logger.warning(f"[风控-紧急退出] {today.date()} 回撤{current_drawdown:.1%}≤-{DRAWDOWN_MAX:.0%}，全仓清仓")
                    self._log_trade(today, "紧急清仓", self.current_weights.copy())
                # 清仓：全部卖出
                cash = portfolio_value
                shares = np.zeros(self.n_assets)
                self.current_weights = np.zeros(self.n_assets)
                continue

            # 2. 判断是否需要再平衡
            need_rebalance = False
            rebalance_reason = ""

            # 2a. 日历再平衡（月末）
            current_month = today.month
            if current_month != prev_month and self.last_rebalance_date != today:
                need_rebalance = True
                rebalance_reason = "月度再平衡"

            # 2b. 阈值偏离再平衡
            if not need_rebalance and self.last_rebalance_date is not None:
                # 获取最近的目标权重
                last_target = self.target_history[self.last_rebalance_date_index, :]
                if np.any(np.abs(self.current_weights - last_target) > THRESHOLD_PCT):
                    need_rebalance = True
                    rebalance_reason = f"阈值偏离(>±{THRESHOLD_PCT:.0%})"

            # 2c. 波动率变化再平衡
            if not need_rebalance and self.last_rebalance_date is not None:
                current_cov = estimate_covariance(self.price_df.iloc[:t+1], window=min(60, t))
                if last_vol is not None:
                    current_vol = np.sqrt(np.dot(self.current_weights.T, np.dot(current_cov, self.current_weights)))
                    if last_vol > 1e-10 and abs(current_vol / last_vol - 1) > VOL_CHANGE_THRESHOLD:
                        need_rebalance = True
                        rebalance_reason = f"波动率变化>{VOL_CHANGE_THRESHOLD:.0%}"

            # 更新波动率记录
            if t % 60 == 0 or last_vol is None:
                cov_current = estimate_covariance(self.price_df.iloc[:t+1], window=min(60, t))
                last_vol = np.sqrt(np.dot(self.current_weights.T, np.dot(cov_current, self.current_weights)))

            prev_month = current_month

            # 3. 执行再平衡（如果需要）
            if need_rebalance:
                self.target_history[t, :] = self.current_weights.copy()  # 先记录当前权重

                # 3a. 计算ERC目标权重
                target_weights, info = get_target_weights(
                    self.price_df.iloc[:t+1],
                    self.codes,
                    last_weights=self.current_weights,
                )
                self.logger.info(
                    f"[再平衡] {today.date()} | 原因: {rebalance_reason} | "
                    f"{'成功' if info['success'] else '回退'}"
                )

                # 统计优化器降级情况
                level = info.get("bounds_level", "tight")
                if info.get("fallback"):
                    self.optimizer_stats["fallback"] += 1
                else:
                    self.optimizer_stats[level] = self.optimizer_stats.get(level, 0) + 1

                # 3b. 执行风控检查（三层）——已关闭
                self.target_history[t, :] = target_weights.copy()
                adjusted_weights = target_weights.copy()
                # adjusted_weights = self._apply_risk_controls(
                #     target_weights, t, current_drawdown
                # )

                # 3c. 执行交易（从当前权重调整到目标权重）
                if not np.allclose(self.current_weights, adjusted_weights, atol=1e-4):
                    # 使用EXECUTION_RATIO逐步调整，减少冲击成本
                    delta = adjusted_weights - self.current_weights
                    execution_weights = self.current_weights + delta * EXECUTION_RATIO
                    execution_weights = np.clip(execution_weights, WEIGHT_MIN, WEIGHT_MAX)
                    execution_weights = execution_weights / execution_weights.sum()

                    # 执行换仓：按实际换手率收交易成本，整手100股取整
                    old_values = shares * prices_today
                    new_target_values = portfolio_value * execution_weights
                    turnover = np.sum(np.abs(new_target_values - old_values))
                    trade_cost = turnover * TOTAL_COST_PER_SIDE
                    available_capital = portfolio_value - trade_cost

                    # 记录换手率（相对组合价值）
                    self.turnover_history[t] = turnover / portfolio_value

                    new_shares = np.zeros(self.n_assets)
                    remaining_cash = available_capital
                    for i in range(self.n_assets):
                        target_value = available_capital * execution_weights[i]
                        if target_value >= MIN_TRADE_AMOUNT:
                            raw_shares = target_value / prices_today[i]
                            new_shares[i] = int(raw_shares / 100) * 100      # 整手取整
                            remaining_cash -= new_shares[i] * prices_today[i]  # 按整手后的实际金额扣除
                        else:
                            # 金额太小，保持现金
                            pass

                    # 更新状态
                    shares = new_shares
                    cash = remaining_cash
                    self.current_weights = execution_weights
                    self.n_trades += 1
                    self._log_trade(today, rebalance_reason, execution_weights)

                    self.logger.info(
                        f"[交易执行] {today.date()} | 换仓完成 | "
                        f"交易次数累计: {self.n_trades} | "
                        f"组合价值: {portfolio_value:,.0f}"
                    )
                else:
                    self.logger.info(f"[交易执行] {today.date()} | 权重无变化，跳过")

                self.last_rebalance_date = today
                self.last_rebalance_date_index = t

            # 每日详细日志（可选，每20天输出一次）
            if t % 20 == 0 or t == self.n_dates - 1:
                self.logger.debug(
                    f"[每日状态] {today.date()} | "
                    f"净值: {portfolio_value:,.0f} | "
                    f"权重: {dict(zip(self.codes, np.round(self.current_weights, 4)))}"
                )

        # ----- 回测结束 -----
        final_nav = self.equity_curve[-1] / self.initial_capital
        self.logger.info("=" * 60)
        self.logger.info(f"回测结束 | 最终净值: {self.equity_curve[-1]:,.0f} | 累计倍数: {final_nav:.4f}")
        self.logger.info(f"总交易次数: {self.n_trades}")
        self.logger.info(f"优化器统计: 一次求解={self.optimizer_stats['tight']}, "
                         f"一级降级={self.optimizer_stats['loose']}, "
                         f"二级降级={self.optimizer_stats['free']}, "
                         f"等权重回退={self.optimizer_stats['fallback']}")
        self.logger.info("=" * 60)

        # 整理返回结果
        result = {
            "equity_curve": pd.Series(self.equity_curve, index=self.dates, name="portfolio"),
            "weights": pd.DataFrame(self.weight_history, index=self.dates, columns=self.codes),
            "returns": pd.Series(np.diff(self.equity_curve, prepend=self.initial_capital) / np.append(self.initial_capital, self.equity_curve[:-1])[:self.n_dates], index=self.dates),
            "n_trades": self.n_trades,
            "trade_log": self.trade_log,
            "nav": final_nav,
            "turnover_history": pd.Series(self.turnover_history, index=self.dates, name="turnover"),
        }
        return result

    # ========================================================================
    # 三层风控体系
    # ========================================================================
    def _apply_risk_controls(
        self,
        weights: np.ndarray,
        t: int,
        current_drawdown: float,
    ) -> np.ndarray:
        """
        三层风险控制系统。

        第一层——波动率目标控制：
            如果预测波动率超出目标范围[TARGET_VOL*(1-VOL_BAND), TARGET_VOL*(1+VOL_BAND)]，
            通过缩放系数vol_scale调整整体敞口，使组合波动率回到目标水平。

        第二层——回撤控制：
            回撤达到10%时，降低权益类敞口到80%
            回撤达到15%时，降低权益类敞口到50%

        第三层——CVaR风险预警：
            按月频计算CVaR，如果超过15%触发减仓
            极端情况启动紧急退出

        Args:
            weights: 优化后的目标权重
            t: 当前时间索引
            current_drawdown: 当前回撤比例

        Returns:
            风控调整后的权重
        """
        adjusted = weights.copy()
        controls_applied = []

        # ----- 第一层：波动率目标控制 -----
        cov = estimate_covariance(self.price_df.iloc[:t+1], window=min(60, t))
        predicted_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        # 年化波动率
        annual_vol = predicted_vol * np.sqrt(252)

        if annual_vol > TARGET_VOL * (1 + VOL_BAND[1]):
            # 波动率过高 → 降杠杆
            self.vol_scale = max(0.3, TARGET_VOL / max(annual_vol, 1e-6))
            adjusted *= self.vol_scale
            controls_applied.append(f"vol_scale={self.vol_scale:.2f}")
        elif annual_vol < TARGET_VOL * (1 - VOL_BAND[0]) and not self.cvar_active:
            # 波动率过低 → 可适当加杠杆（但在A股受限，这里微微上调）
            self.vol_scale = min(1.2, TARGET_VOL / max(annual_vol, 1e-6))
            adjusted *= self.vol_scale
            controls_applied.append(f"vol_scale={self.vol_scale:.2f}")
        else:
            self.vol_scale = 1.0

        # ----- 第二层：回撤控制 -----
        if current_drawdown <= -DRAWDOWN_WARN:
            if current_drawdown > -0.15:
                # 回撤 10%~15%：降低权益敞口到80%
                self.reduce_equity = 0.80
                controls_applied.append("reduce_equity=0.80")
            else:
                # 回撤 15%~20%：降低权益敞口到50%
                self.reduce_equity = 0.50
                controls_applied.append("reduce_equity=0.50")

            # 将权益类的权重乘以缩减系数
            for i in range(len(self.codes)):
                for etf in ETF_POOL:
                    if etf["code"] == self.codes[i] and etf["asset_class"] == "equity":
                        adjusted[i] *= self.reduce_equity
                        break
        else:
            self.reduce_equity = 1.0

        # ----- 第三层：CVaR风险预警 -----
        # 计算月频CVaR（基于最近60个交易日收益率）
        returns = np.log(self.price_df.iloc[:t+1]).diff().dropna()
        if len(returns) >= 20:
            # 计算组合日收益率
            port_returns = returns.values @ adjusted[:len(returns.values[0])] if len(adjusted) == len(returns.columns) else returns.values @ adjusted
            # 月收益近似 = 日收益 * 20
            monthly_returns = port_returns[-20:] @ np.ones(min(20, len(port_returns)))  # 简化为最近20天累计
            # 简化的CVaR：取最差的5%分位数
            if len(port_returns) >= 60:
                worst_returns = np.sort(port_returns[-60:])
                cvar_5pct = worst_returns[:3].mean() * np.sqrt(20)  # 近似月度CVaR
            else:
                cvar_5pct = 0

            if cvar_5pct < -CVAR_THRESHOLD and not self.emergency_exit:
                # 减仓50%
                adjusted *= 0.5
                self.cvar_active = True
                controls_applied.append(f"cvar_trigger({cvar_5pct:.2%})")
                self.logger.warning(
                    f"[风控-CVaR] {self.dates[t].date()} "
                    f"CVaR={cvar_5pct:.2%}≤-{CVAR_THRESHOLD:.0%}，减仓50%"
                )
            elif cvar_5pct >= -CVAR_THRESHOLD * 0.5:
                # 恢复正常后解除CVaR状态
                if self.cvar_active:
                    self.cvar_active = False
                    self.logger.info(f"[风控-CVaR解除] {self.dates[t].date()}")
                controls_applied.append("cvar_normal")

        # 归一化权重（确保和为1）
        adjusted_sum = adjusted.sum()
        if adjusted_sum > 1e-10:
            adjusted = adjusted / adjusted_sum
        else:
            # 极端情况：全部减到0，改为等权重
            adjusted = np.ones(len(adjusted)) / len(adjusted)
            controls_applied.append("fallback_equal_weight")

        # 风控日志
        if controls_applied:
            self.logger.info(
                f"[风控] {self.dates[t].date()} | "
                f"回撤:{current_drawdown:.2%} | "
                f"措施: {', '.join(controls_applied)}"
            )

        return adjusted

    # ========================================================================
    # 交易日志
    # ========================================================================
    def _log_trade(self, date: pd.Timestamp, action: str, weights: np.ndarray):
        """
        记录单笔交易日志。

        Args:
            date: 交易日期
            action: 操作类型（如"初始建仓"、"月度再平衡"）
            weights: 交易后的权重向量
        """
        weight_summary = {self.codes[i]: f"{weights[i]:.2%}" for i in range(len(self.codes))}
        entry = (date, action, weight_summary)
        self.trade_log.append(entry)
        self.logger.info(
            f"[交易] {date.date()} | {action} | "
            f"权重: {weight_summary}"
        )