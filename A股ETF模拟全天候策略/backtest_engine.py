import numpy as np
import pandas as pd

from config import (
    INITIAL_CAPITAL, THRESHOLD_PCT,
    VOL_CHANGE_THRESHOLD, EXECUTION_RATIO, SINGLE_ETF_MAX,
    TOTAL_COST_PER_SIDE, VOL_WINDOW, MIN_TRADE_AMOUNT,
)
from risk_parity import calc_etf_weights
from logger import logger


RISK_LABELS = {
    "vol_scale": "波动率超标",
    "emergency_exit": "最大回撤止损",
    "reduce_equity": "回撤降权益",
    "cvar_trigger": "CVaR风险预警",
}


def check_risk_controls(portfolio_value: float,
                        peak_value: float,
                        daily_returns: pd.Series) -> dict:
    current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
    actions = {}
    vol = daily_returns.tail(20).std() * np.sqrt(252) if len(daily_returns) >= 20 else 0
    if vol > 0.12:
        actions["vol_scale"] = 0.10 / vol
    if current_drawdown > 0.20:
        actions["emergency_exit"] = True
    elif current_drawdown > 0.10:
        actions["reduce_equity"] = True
    if len(daily_returns) >= 252:
        tail = daily_returns.tail(252)
        var_95 = tail.quantile(0.05)
        cvar = tail[tail <= var_95].mean() if (tail <= var_95).any() else var_95
        if abs(cvar) > 0.15:
            actions["cvar_trigger"] = True
    return actions


def execute_rebalance(current_weights: pd.Series,
                      target_weights: pd.Series,
                      portfolio_value: float,
                      prices: pd.Series,
                      date) -> list[dict]:
    target_weights_clipped = target_weights.clip(upper=SINGLE_ETF_MAX)
    target_weights_clipped /= target_weights_clipped.sum()
    diff = target_weights_clipped - current_weights
    trades = []
    for code in diff.index:
        delta = diff[code] * portfolio_value * EXECUTION_RATIO
        if abs(delta) < MIN_TRADE_AMOUNT:
            continue
        price = prices.get(code, np.nan)
        if pd.isna(price) or price <= 0:
            continue
        direction = "buy" if delta > 0 else "sell"
        trade_amount = abs(delta)
        cost = trade_amount * TOTAL_COST_PER_SIDE
        trades.append({
            "日期": date,
            "代码": code,
            "方向": direction,
            "金额": round(trade_amount, 2),
            "价格": round(price, 4),
            "成本": round(cost, 6),
        })
    return trades


def run_backtest(close_df: pd.DataFrame,
                 etf_pool: list[dict]) -> dict:
    return_df = close_df.pct_change().dropna()
    n_assets = len(close_df.columns)
    cash = float(INITIAL_CAPITAL)
    holdings = pd.Series(0.0, index=close_df.columns, dtype=float)
    equal_weight = pd.Series(1.0 / n_assets, index=close_df.columns)

    portfolio_values = []
    trade_records = []
    weight_history = []
    peak_value = float(INITIAL_CAPITAL)
    last_rebalance_date = None
    prev_vol = None
    target_weights = equal_weight.copy()

    daily_returns_buffer = []

    for date in close_df.index:
        prices = close_df.loc[date]
        position_value = (holdings * prices).sum()
        portfolio_value = cash + position_value
        portfolio_values.append(portfolio_value)

        if portfolio_value > peak_value:
            peak_value = portfolio_value

        daily_returns_buffer.append(portfolio_value)
        daily_returns_series = pd.Series(daily_returns_buffer).pct_change().dropna()

        risk_actions = check_risk_controls(portfolio_value, peak_value, daily_returns_series)
        for key in risk_actions:
            if key != "emergency_exit":
                logger.warning("风控触发 | %s | 净值: %.2f | 回撤: %.2f%%",
                               RISK_LABELS.get(key, key), portfolio_value,
                               (peak_value - portfolio_value) / peak_value * 100)

        current_weights = pd.Series(0.0, index=close_df.columns)
        if portfolio_value > 0:
            current_weights = (holdings * prices) / portfolio_value

        equity_codes = [e["code"] for e in etf_pool if e["asset_class"] == "equity"]
        mid_bond_codes = [e["code"] for e in etf_pool if e["asset_class"] == "mid_bond"]
        bond_code = "511010"

        if "vol_scale" in risk_actions:
            scale = risk_actions["vol_scale"]
            target_weights = target_weights * scale
            target_weights /= target_weights.sum()

        if "reduce_equity" in risk_actions:
            for code in equity_codes:
                if code in target_weights.index:
                    target_weights[code] *= 0.4
            if bond_code in target_weights.index:
                target_weights[bond_code] += target_weights.sum() * 0.15
            target_weights /= target_weights.sum()

        if "cvar_trigger" in risk_actions:
            for code in target_weights.index:
                target_weights[code] *= 0.8
            if bond_code in target_weights.index:
                target_weights[bond_code] += target_weights.sum() * 0.2
            target_weights /= target_weights.sum()

        if risk_actions.get("emergency_exit"):
            logger.critical("风控触发 | 紧急清仓 | 日期: %s | 净值: %.2f | 回撤: %.2f%%",
                            date.date(), portfolio_value,
                            (peak_value - portfolio_value) / peak_value * 100)
            for code in holdings.index:
                if holdings[code] > 0:
                    trade_amount = holdings[code] * prices[code]
                    cost = trade_amount * TOTAL_COST_PER_SIDE
                    trade_records.append({
                        "日期": date, "代码": code, "方向": "sell",
                        "金额": round(trade_amount, 2),
                        "价格": round(prices[code], 4),
                        "成本": round(cost, 6),
                    })
                    logger.info("清仓 | %s | 金额: %.2f | 价格: %.4f", code, trade_amount, prices[code])
                    cash += trade_amount - cost
                    holdings[code] = 0
            weight_history.append(current_weights.copy())
            continue

        should_rebalance = False
        strategy_weights = target_weights

        is_month_start = (
            last_rebalance_date is None
            or date.month != last_rebalance_date.month
            or date.year != last_rebalance_date.year
        )
        if is_month_start and date in return_df.index:
            returns_up_to = return_df.loc[:date]
            if len(returns_up_to) >= VOL_WINDOW:
                try:
                    strategy_weights = calc_etf_weights(returns_up_to, etf_pool)
                    should_rebalance = True
                    last_rebalance_date = date
                except (ValueError, RuntimeError):
                    pass

        if not should_rebalance:
            max_deviation = (current_weights - target_weights).abs().max()
            if max_deviation > THRESHOLD_PCT:
                if date in return_df.index:
                    returns_up_to = return_df.loc[:date]
                    if len(returns_up_to) >= VOL_WINDOW:
                        try:
                            strategy_weights = calc_etf_weights(returns_up_to, etf_pool)
                            should_rebalance = True
                            last_rebalance_date = date
                        except (ValueError, RuntimeError):
                            pass

        if not should_rebalance and prev_vol is not None:
            if date in return_df.index:
                returns_up_to = return_df.loc[:date]
                if len(returns_up_to) >= VOL_WINDOW:
                    current_vol = returns_up_to.tail(VOL_WINDOW).std().mean() * np.sqrt(252)
                    if prev_vol > 0 and abs(float(current_vol) - float(prev_vol)) / float(prev_vol) > VOL_CHANGE_THRESHOLD:
                        try:
                            strategy_weights = calc_etf_weights(returns_up_to, etf_pool)
                            should_rebalance = True
                            last_rebalance_date = date
                        except (ValueError, RuntimeError):
                            pass

        target_weights = strategy_weights

        if should_rebalance:
            trades = execute_rebalance(current_weights, target_weights,
                                       portfolio_value, prices, date)
            if trades:
                logger.info("再平衡 | 日期: %s | 交易笔数: %d | 组合净值: %.2f",
                            date.date(), len(trades), portfolio_value)
            trade_records.extend(trades)
            for t in trades:
                index = close_df.columns.get_loc(t["代码"])
                if t["方向"] == "buy":
                    share_delta = t["金额"] / t["价格"]
                    cash -= t["金额"] + t["成本"]
                    holdings.iloc[index] += share_delta
                    logger.debug("买入 | %s | 金额: %.2f | 价格: %.4f | 股数: %.0f",
                                 t["代码"], t["金额"], t["价格"], share_delta)
                else:
                    share_delta = t["金额"] / t["价格"]
                    cash += t["金额"] - t["成本"]
                    holdings.iloc[index] -= share_delta
                    logger.debug("卖出 | %s | 金额: %.2f | 价格: %.4f | 股数: %.0f",
                                 t["代码"], t["金额"], t["价格"], share_delta)

        if date in return_df.index:
            returns_up_to = return_df.loc[:date]
            if len(returns_up_to) >= VOL_WINDOW:
                prev_vol = returns_up_to.tail(VOL_WINDOW).std().mean() * np.sqrt(252)

        if last_rebalance_date is not None and date.month != last_rebalance_date.month and date.day <= 5:
            logger.info("持仓月报 | %s | 净值: %.2f | 持仓: %s",
                        date.date(), portfolio_value,
                        {k: f"{v:.1%}" for k, v in current_weights.items() if v > 0.01})

        weight_history.append(current_weights.copy())

    return {
        "portfolio_values": pd.Series(portfolio_values, index=close_df.index),
        "trade_records": pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
            columns=["日期", "代码", "方向", "金额", "价格", "成本"]
        ),
        "weight_history": pd.DataFrame(weight_history, index=close_df.index) if weight_history else pd.DataFrame(),
    }
