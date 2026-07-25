#!/usr/bin/env python3
"""
ETF双均线滚动策略回测 V2 — 风险控制增强版
==========================================

策略原理：
    基于双均线交叉信号的趋势跟踪策略（V1基础），增加以下风险控制：
    1. 大盘择时过滤：沪深300均线过滤，熊市不开仓
    2. 单笔止损：持仓亏损达3%强制平仓
    3. ATR波动率适应：根据波动率动态调整仓位大小
    4. 最大回撤止损：净值回撤达15%全仓清仓并暂停交易
    5. 总量风控：单只ETF最大持仓不超过总资金30%

功能说明：
    - 支持9只宽基ETF的回测
    - 测试15种均线周期组合
    - 包含完整的风控模块
    - 生成绩效矩阵、热力图、调仓记录

作者：ETF策略研究
日期：2026-07-19
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 配置参数区域
# =============================================================================

# ETF标的池：覆盖A股主要宽基指数
ETFS = {
    '512150': '富时A50',
    '510310': '沪深300',
    '510500': '中证500',
    '512100': '中证1000',
    '563300': '中证2000',
    '510210': '上证指数',
    '159943': '深证成指',
    '159915': '创业板指',
    '588000': '科创50',
}

# 均线周期组合配置
MA_COMBINATIONS = [
    (5, 10), (5, 20), (5, 40), (5, 60), (5, 120),
    (10, 20), (10, 40), (10, 60), (10, 120),
    (20, 40), (20, 60), (20, 120),
    (40, 60), (40, 120),
    (60, 120),
]

# 交易成本参数（基于中国A股ETF实际费率）
COMMISSION_RATE = 0.00003   # 佣金费率：单方万分之0.3
SLIPPAGE_RATE = 0.0005      # 滑点假设：0.05%/笔
TOTAL_COST_PER_SIDE = COMMISSION_RATE + SLIPPAGE_RATE

# 无风险利率（年化）
RISK_FREE_RATE = 0.025  # 2.5%

# =============================================================================
# V2 风控参数
# =============================================================================

MARKET_MA_PERIOD = 60       # 大盘择时均线周期（沪深300的MA60）
STOP_LOSS_PCT = 0.03        # 单笔止损比例: 3%
MAX_DRAWDOWN_PCT = 0.15     # 最大回撤止损: 15%（从净值高点）
ATR_PERIOD = 20             # ATR计算周期
RISK_PER_TRADE = 0.01       # 每笔交易风险敞口: 账户净值的1%
MAX_POSITION_PCT = 0.30     # 单只ETF最大持仓比例: 30%
INITIAL_CAPITAL = 1000000   # 初始资金：100万元

# 文件路径配置
DATA_DIR = '/home/jintz/ETF/data/etf'
OUTPUT_DIR = '/home/jintz/ETF/output'


# =============================================================================
# 数据加载模块
# =============================================================================

def load_etf_data(code):
    """
    加载ETF历史日线数据

    参数：
        code: ETF代码，如 '510310'

    返回：
        DataFrame: 以日期为索引的OHLCV数据
    """
    filepath = os.path.join(DATA_DIR, f'{code}.csv')
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期')
    df = df.sort_index()
    return df


# =============================================================================
# 信号计算模块
# =============================================================================

def calculate_ma_signal(close_prices, slow_period, fast_period):
    """
    计算双均线交叉信号

    参数：
        close_prices: 收盘价序列
        slow_period: 慢线周期（短周期均线）
        fast_period: 快线周期（长周期均线）

    返回：
        tuple: (信号序列, 慢线序列, 快线序列)
    """
    slow_ma = close_prices.rolling(window=slow_period).mean()
    fast_ma = close_prices.rolling(window=fast_period).mean()
    signals = pd.Series(np.nan, index=close_prices.index)
    golden_cross = (slow_ma > fast_ma) & (slow_ma.shift(1) <= fast_ma.shift(1))
    death_cross = (slow_ma < fast_ma) & (slow_ma.shift(1) >= fast_ma.shift(1))
    signals[golden_cross] = 1
    signals[death_cross] = 0
    return signals, slow_ma, fast_ma


# =============================================================================
# V2新增：ATR波动率计算模块
# =============================================================================

def calculate_atr(high_prices, low_prices, close_prices, period=ATR_PERIOD):
    """
    计算ATR（Average True Range）真实波动幅度

    参数：
        high_prices: 最高价序列
        low_prices: 最低价序列
        close_prices: 收盘价序列
        period: ATR周期，默认20日

    返回：
        pd.Series: ATR序列
    """
    prev_close = close_prices.shift(1)
    tr1 = high_prices - low_prices
    tr2 = (high_prices - prev_close).abs()
    tr3 = (low_prices - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


# =============================================================================
# V2新增：大盘择时模块
# =============================================================================

def calculate_market_timing(market_close, period=MARKET_MA_PERIOD):
    """
    计算大盘择时信号

    参数：
        market_close: 大盘（沪深300）收盘价序列
        period: 均线周期，默认60日

    返回：
        pd.Series: 1=多头（允许开仓）, 0=空头（禁止开仓）
    """
    ma = market_close.rolling(window=period).mean()
    timing = pd.Series(np.nan, index=market_close.index)
    timing[market_close > ma] = 1
    timing[market_close <= ma] = 0
    return timing


# =============================================================================
# V2新增：出场交易记录辅助函数
# =============================================================================

def _record_exit_trade(date, code, price, shares, entry_price, entry_buy_cost,
                       total_cost, direction, signal_label, slow_p, fast_p):
    """
    计算出场交易的损益并生成交易记录字典

    参数：
        date: 交易日期
        code: ETF代码
        price: 出场价格
        shares: 持仓股数
        entry_price: 买入价格
        entry_buy_cost: 买入时的交易成本
        total_cost: 累计交易成本（更新前）
        direction: 方向标签（'止损'/'止损清仓'/'卖出'/'平仓'）
        signal_label: 信号标签（'单笔止损'/'最大回撤止损'/'死叉'/'到期平仓'）
        slow_p: 慢线周期
        fast_p: 快线周期

    返回：
        tuple: (trade_dict, net_pnl, return_pct, trade_cost)
               trade_dict: 交易记录字典
               net_pnl: 净损益
               return_pct: 收益率(%)
               trade_cost: 本次交易成本
    """
    trade_amount = shares * price
    commission = trade_amount * COMMISSION_RATE
    slippage = trade_amount * SLIPPAGE_RATE
    total_trade_cost = commission + slippage

    gross_pnl = (price - entry_price) * shares
    net_pnl = gross_pnl - entry_buy_cost - total_trade_cost

    total_invested = entry_price * shares + entry_buy_cost
    return_pct = net_pnl / total_invested * 100 if total_invested > 0 else 0

    trade_dict = {
        '日期': date.strftime('%Y-%m-%d'),
        '代码': code,
        '方向': direction,
        '价格': round(price, 4),
        '成交量': shares,
        '交易金额': round(trade_amount, 2),
        '佣金': round(commission, 2),
        '滑点成本': round(slippage, 2),
        '总成本': round(total_trade_cost, 2),
        '持仓市值': 0,
        '累计成本': round(total_cost + total_trade_cost, 2),
        '信号': signal_label,
        '慢线周期': slow_p,
        '快线周期': fast_p,
        '盈亏': round(net_pnl, 2),
        '收益率': round(return_pct, 4),
    }

    return trade_dict, net_pnl, return_pct, total_trade_cost


# =============================================================================
# 交易模拟模块（V2增强版 — 含5层风控）
# =============================================================================

def simulate_trading(close_prices, signals, atr_series, market_timing, code, slow_p, fast_p):
    """
    模拟交易执行（V2增强版 — 含风控）

    参数：
        close_prices: 收盘价序列
        signals: 信号序列（1=买，0=卖，NaN=无）
        atr_series: ATR波动率序列（V2新增）
        market_timing: 大盘择时信号序列（V2新增，已前向填充）
        code: ETF代码
        slow_p: 慢线周期
        fast_p: 快线周期

    返回：
        tuple: (交易记录DataFrame, 绩效指标字典)

    风控逻辑（按优先级从高到低）：
        1. 最大回撤止损：净值回撤>=15% → 全仓清仓+暂停交易
        2. 单笔止损：持仓亏损>=3% → 强制平仓（每天检查）
        3. 大盘择时过滤：大盘收盘<MA60 → 禁止新开仓
        4. ATR仓位管理：根据ATR动态计算买入股数
        5. 总量风控：单笔持仓不超过总资金30%

    注意：回撤止损使用盘前净值计算（每日循环顶部），存在一天滞后，
    这是日频回测的标准局限性。
    """
    trades = []
    position = 0
    entry_price = 0
    entry_buy_cost = 0
    cash = INITIAL_CAPITAL
    shares = 0
    total_cost = 0

    # V2新增：风控状态
    peak_equity = INITIAL_CAPITAL
    drawdown_triggered = False
    drawdown_trigger_date = None
    last_date = close_prices.index[-1]

    # V2新增：每日净值曲线（用于精确计算策略级Sharpe/MaxDD/Calmar）
    equity_curve = []

    # 预缓存market_timing的日期集合，加速查找
    timing_index = market_timing.index

    for i in range(len(signals)):
        date = signals.index[i]
        price = close_prices.iloc[i]
        signal = signals.iloc[i]
        current_atr = atr_series.iloc[i] if i < len(atr_series) and not pd.isna(atr_series.iloc[i]) else np.nan

        # 计算当前净值（盘前）
        equity = cash + shares * price if position == 1 else cash

        # 记录每日净值
        equity_curve.append(equity)

        # 更新高水位
        if equity > peak_equity:
            peak_equity = equity

        # ================================================================
        # 风控1: 最大回撤止损检查
        # ================================================================
        if not drawdown_triggered and peak_equity > 0:
            current_drawdown = (peak_equity - equity) / peak_equity
            if current_drawdown >= MAX_DRAWDOWN_PCT:
                drawdown_triggered = True
                drawdown_trigger_date = date
                # 如果有持仓，立即清仓
                if position == 1:
                    trade_dict, _, _, trade_cost = _record_exit_trade(
                        date, code, price, shares, entry_price, entry_buy_cost,
                        total_cost, '止损清仓', '最大回撤止损', slow_p, fast_p
                    )
                    trades.append(trade_dict)
                    cash += shares * price - trade_cost
                    total_cost += trade_cost
                    shares = 0
                    position = 0
                    entry_buy_cost = 0
                continue

        # 回撤已触发 → 跳过所有交易
        if drawdown_triggered:
            continue

        # 跳过无信号日，但仍需检查单笔止损
        if pd.isna(signal):
            if position == 1 and price <= entry_price * (1 - STOP_LOSS_PCT):
                trade_dict, _, _, trade_cost = _record_exit_trade(
                    date, code, price, shares, entry_price, entry_buy_cost,
                    total_cost, '止损', '单笔止损', slow_p, fast_p
                )
                trades.append(trade_dict)
                cash += shares * price - trade_cost
                total_cost += trade_cost
                shares = 0
                position = 0
                entry_buy_cost = 0
            continue

        # 最后一天买入信号跳过（避免开仓后立即平仓）
        if signal == 1 and date == last_date:
            continue

        # ================================================================
        # 风控2: 单笔止损检查（有信号时也检查，优先级高于信号操作）
        # ================================================================
        if position == 1 and price <= entry_price * (1 - STOP_LOSS_PCT):
            trade_dict, _, _, trade_cost = _record_exit_trade(
                date, code, price, shares, entry_price, entry_buy_cost,
                total_cost, '止损', '单笔止损', slow_p, fast_p
            )
            trades.append(trade_dict)
            cash += shares * price - trade_cost
            total_cost += trade_cost
            shares = 0
            position = 0
            entry_buy_cost = 0
            continue

        # ========== 买入逻辑（V2增强） ==========
        if signal == 1 and position == 0:
            # 风控3: 大盘择时过滤
            # market_timing已前向填充，缺失日期继承最近已知信号
            if date in timing_index and not pd.isna(market_timing.get(date)) and market_timing[date] == 0:
                continue

            # 风控4: ATR仓位管理
            if pd.isna(current_atr) or current_atr <= 0:
                continue

            if price <= 0:
                continue

            equity_now = cash
            risk_budget = equity_now * RISK_PER_TRADE
            atr_based_shares = int(risk_budget / current_atr / 100) * 100

            # 风控5: 总量风控 — 持仓不超过总资金30%
            max_amount = equity_now * MAX_POSITION_PCT
            max_shares_by_limit = int(max_amount / price / 100) * 100

            # 取两者较小值
            shares_to_buy = min(atr_based_shares, max_shares_by_limit)

            if shares_to_buy <= 0:
                continue

            # 确保不超过可用现金
            cost_per_share = price * (1 + TOTAL_COST_PER_SIDE)
            shares_to_buy = min(shares_to_buy, int(cash / cost_per_share / 100) * 100)

            if shares_to_buy > 0:
                trade_amount = shares_to_buy * price
                commission = trade_amount * COMMISSION_RATE
                slippage = trade_amount * SLIPPAGE_RATE
                total_trade_cost = commission + slippage

                cash -= (trade_amount + total_trade_cost)
                shares = shares_to_buy
                position = 1
                entry_price = price
                entry_buy_cost = total_trade_cost
                total_cost += total_trade_cost

                trades.append({
                    '日期': date.strftime('%Y-%m-%d'),
                    '代码': code,
                    '方向': '买入',
                    '价格': round(price, 4),
                    '成交量': shares_to_buy,
                    '交易金额': round(trade_amount, 2),
                    '佣金': round(commission, 2),
                    '滑点成本': round(slippage, 2),
                    '总成本': round(total_trade_cost, 2),
                    '持仓市值': round(shares * price, 2),
                    '累计成本': round(total_cost, 2),
                    '信号': '金叉',
                    '慢线周期': slow_p,
                    '快线周期': fast_p,
                })

        # ========== 卖出逻辑（与V1相同） ==========
        elif signal == 0 and position == 1:
            trade_dict, _, _, trade_cost = _record_exit_trade(
                date, code, price, shares, entry_price, entry_buy_cost,
                total_cost, '卖出', '死叉', slow_p, fast_p
            )
            trades.append(trade_dict)
            cash += shares * price - trade_cost
            total_cost += trade_cost
            shares = 0
            entry_buy_cost = 0

    # ========== 到期平仓逻辑 ==========
    if position == 1:
        last_price = close_prices.iloc[-1]
        actual_last_date = close_prices.index[-1]

        today_has_sell = any(
            t['日期'] == actual_last_date.strftime('%Y-%m-%d') and t['方向'] in ['卖出', '止损', '止损清仓']
            for t in trades
        )

        if not today_has_sell:
            trade_dict, _, _, trade_cost = _record_exit_trade(
                actual_last_date, code, last_price, shares, entry_price, entry_buy_cost,
                total_cost, '平仓', '到期平仓', slow_p, fast_p
            )
            trades.append(trade_dict)
            cash += shares * last_price - trade_cost
            total_cost += trade_cost

    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    # 构建策略净值Series
    equity_series = pd.Series(equity_curve, index=close_prices.index[:len(equity_curve)])

    # 计算绩效指标（使用策略净值曲线而非ETF价格）
    metrics = calculate_performance(close_prices, equity_series, trades_df, cash, total_cost)

    # V2新增：风控状态
    metrics['回撤止损触发'] = '是' if drawdown_triggered else '否'
    if drawdown_triggered and drawdown_trigger_date:
        metrics['回撤止损日期'] = drawdown_trigger_date.strftime('%Y-%m-%d')

    return trades_df, metrics


# =============================================================================
# 绩效计算模块
# =============================================================================

def calculate_performance(close_prices, equity_series, trades_df, final_cash, total_cost):
    """
    计算策略绩效指标

    参数：
        close_prices: 原始收盘价序列（仅用于买入持有基准和年数计算）
        equity_series: 策略每日净值Series（V2新增，用于计算策略级指标）
        trades_df: 交易记录DataFrame
        final_cash: 回测结束时的现金余额
        total_cost: 累计交易成本

    返回：
        dict: 包含所有绩效指标的字典
    """
    total_return = (final_cash - INITIAL_CAPITAL) / INITIAL_CAPITAL

    days = (close_prices.index[-1] - close_prices.index[0]).days
    years = days / 365.25

    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

    buy_hold_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
    buy_hold_annual = (1 + buy_hold_return) ** (1/years) - 1 if years > 0 else 0

    # V2修复：使用策略净值曲线计算波动率和最大回撤（而非ETF价格）
    equity_daily_returns = equity_series.pct_change().dropna()
    annual_volatility = equity_daily_returns.std() * np.sqrt(252)

    excess_return = annual_return - RISK_FREE_RATE
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0

    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_drawdown = drawdown.min()

    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    excess_over_buyhold = annual_return - buy_hold_annual

    # 交易统计
    if not trades_df.empty and len(trades_df) > 0:
        sell_trades = trades_df[trades_df['方向'].isin(['卖出', '平仓', '止损', '止损清仓'])]
        num_trades = len(sell_trades)

        buy_dates = trades_df[trades_df['方向'] == '买入']['日期'].values
        sell_dates = sell_trades['日期'].values

        holding_days = 0
        for i in range(min(len(buy_dates), len(sell_dates))):
            delta = pd.Timestamp(sell_dates[i]) - pd.Timestamp(buy_dates[i])
            holding_days += delta.days

        total_days = (close_prices.index[-1] - close_prices.index[0]).days
        holding_ratio = holding_days / total_days if total_days > 0 else 0

        if '盈亏' in sell_trades.columns:
            wins = sell_trades[sell_trades['盈亏'] > 0]
            losses = sell_trades[sell_trades['盈亏'] <= 0]

            win_rate = len(wins) / len(sell_trades) if len(sell_trades) > 0 else 0
            avg_win = wins['盈亏'].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses['盈亏'].mean()) if len(losses) > 0 else 0
            profit_factor = avg_win / avg_loss if avg_loss > 0 else (9999.99 if avg_win > 0 else 0)
        else:
            win_rate = 0
            profit_factor = 0
    else:
        num_trades = 0
        holding_ratio = 0
        win_rate = 0
        profit_factor = 0

    metrics = {
        '总收益率': round(total_return * 100, 4),
        '年化收益率': round(annual_return * 100, 4),
        '年化波动率': round(annual_volatility * 100, 4),
        '最大回撤': round(max_drawdown * 100, 4),
        '夏普比率': round(sharpe_ratio, 4),
        '卡玛比率': round(calmar_ratio, 4),
        '交易次数': num_trades,
        '持仓时间占比': round(holding_ratio * 100, 4),
        '胜率': round(win_rate * 100, 4),
        '盈亏比': round(profit_factor, 4),
        '总交易成本': round(total_cost, 2),
        '买入持有年化': round(buy_hold_annual * 100, 4),
        '超额收益(年化)': round(excess_over_buyhold * 100, 4),
    }

    return metrics


# =============================================================================
# 趋势分析模块
# =============================================================================

def calculate_trend_strength(close_prices, slow_ma, fast_ma):
    """
    计算趋势强度指标

    参数：
        close_prices: 收盘价序列
        slow_ma: 慢线（短周期均线）
        fast_ma: 快线（长周期均线）

    返回：
        dict: 趋势强度指标字典
    """
    trend = (slow_ma > fast_ma).astype(int)
    trend_changes = trend.diff().abs()
    trend_groups = (trend != trend.shift()).cumsum()
    trend_lengths = trend.groupby(trend_groups).agg(['count', 'first', 'last'])
    trend_lengths.columns = ['持续天数', '趋势开始', '趋势结束']
    trend_lengths['趋势方向'] = trend.groupby(trend_groups).first().values

    avg_trend_length = trend_lengths['持续天数'].mean()
    up_ratio = trend.mean()

    price_deviation = ((close_prices - fast_ma) / fast_ma * 100).abs()
    avg_deviation = price_deviation.mean()
    max_deviation = price_deviation.max()

    return {
        '平均趋势持续天数': round(avg_trend_length, 2),
        '上升趋势占比': round(up_ratio * 100, 2),
        '平均偏离度': round(avg_deviation, 2),
        '最大偏离度': round(max_deviation, 2),
        '趋势变化次数': int(trend_changes.sum()),
    }


# =============================================================================
# 可视化模块
# =============================================================================

def generate_heatmaps(results_df):
    """
    生成绩效热力图（V2版）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for idx, (metric, title, cmap, fmt) in enumerate([
        ('夏普比率', '夏普比率', 'RdYlGn', '.2f'),
        ('年化收益率', '年化收益率(%)', 'RdYlGn', '.1f'),
        ('最大回撤', '最大回撤(%)', 'RdYlGn_r', '.1f'),
        ('卡玛比率', '卡玛比率', 'RdYlGn', '.2f'),
    ]):
        ax = axes[idx // 2][idx % 2]
        pivot = results_df.pivot_table(
            index='名称',
            columns=['慢线周期', '快线周期'],
            values=metric
        )
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=fmt,
                   center=0 if metric != '最大回撤' else None)
        ax.set_title(f'{title} (V2风控版)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_performance_v2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存热力图: heatmap_performance_v2.png")


def generate_trend_heatmap(results_df):
    """
    生成趋势强度热力图（V2版）
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (metric, title, cmap) in enumerate([
        ('平均趋势持续天数', '平均趋势持续天数(天)', 'YlOrRd'),
        ('上升趋势占比', '上升趋势占比(%)', 'RdYlGn'),
    ]):
        ax = axes[idx]
        pivot = results_df.pivot_table(
            index='名称',
            columns=['慢线周期', '快线周期'],
            values=metric
        )
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt='.1f')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_trend_v2.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存趋势热力图: heatmap_trend_v2.png")


# =============================================================================
# 汇总报告模块
# =============================================================================

def print_summary(results_df):
    """
    打印V2回测汇总报告
    """
    print("\n" + "="*80)
    print("策略回测汇总 (V2 — 风险控制增强版)")
    print("="*80)

    # 风控触发统计
    if '回撤止损触发' in results_df.columns:
        triggered = results_df[results_df['回撤止损触发'] == '是']
        print(f"\n【风控触发统计】")
        print(f"  最大回撤止损触发次数: {len(triggered)} / {len(results_df)}")
        if len(triggered) > 0:
            for _, row in triggered.iterrows():
                print(f"    {row['代码']} {row['名称']} {int(row['慢线周期'])}x{int(row['快线周期'])} → 触发于 {row.get('回撤止损日期', 'N/A')}")

    # 各ETF最优参数
    print("\n【各ETF最优参数（按夏普比率排序）】")
    for code in results_df['代码'].unique():
        etf_data = results_df[results_df['代码'] == code]
        best = etf_data.loc[etf_data['夏普比率'].idxmax()]

        print(f"\n{code} {best['名称']}:")
        print(f"  最优组合: 慢线{best['慢线周期']}日 vs 快线{best['快线周期']}日")
        print(f"  年化收益: {best['年化收益率']:.2f}%  夏普: {best['夏普比率']:.2f}  卡玛: {best['卡玛比率']:.2f}")
        print(f"  最大回撤: {best['最大回撤']:.2f}%  胜率: {best['胜率']:.1f}%  盈亏比: {best['盈亏比']:.2f}")
        print(f"  买入持有年化: {best['买入持有年化']:.2f}%  超额收益(年化): {best['超额收益(年化)']:.2f}%")
        print(f"  交易成本: {best['总交易成本']:.0f}元  交易次数: {best['交易次数']}")
        dd_flag = best.get('回撤止损触发', '否')
        if dd_flag == '是':
            print(f"  ⚠ 回撤止损触发于: {best.get('回撤止损日期', 'N/A')}")

    # 全局最优TOP 10
    print("\n【全局最优组合 TOP 10】")
    top10 = results_df.nlargest(10, '夏普比率')
    print(f"  {'代码':>6s} {'名称':6s} {'组合':>6s} {'夏普':>6s} {'年化':>8s} {'回撤':>8s} {'超额':>8s} {'风控':>4s}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")

    for _, row in top10.iterrows():
        combo = f"{int(row['慢线周期'])}x{int(row['快线周期'])}"
        dd_flag = row.get('回撤止损触发', '否')
        print(f"  {row['代码']:>6s} {row['名称']:6s} {combo:>6s} {row['夏普比率']:>6.2f} {row['年化收益率']:>7.2f}% {row['最大回撤']:>7.2f}% {row['超额收益(年化)']:>7.2f}% {dd_flag:>4s}")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """
    主函数：执行V2完整回测流程

    V2新增流程：
        0. 预计算沪深300大盘择时信号（全策略共享）
        1. 遍历每只ETF，预计算ATR
        2. 调用增强版simulate_trading()
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"ETF双均线策略 V2 — 风险控制增强版")
    print(f"{'='*50}")
    print(f"均线组合数: {len(MA_COMBINATIONS)}")
    print(f"交易成本: 佣金{COMMISSION_RATE*10000:.1f}%%/单方 + 滑点{SLIPPAGE_RATE*100:.2f}%%")
    print(f"风控参数:")
    print(f"  大盘择时: 沪深300 MA{MARKET_MA_PERIOD}")
    print(f"  单笔止损: {STOP_LOSS_PCT*100:.0f}%")
    print(f"  最大回撤止损: {MAX_DRAWDOWN_PCT*100:.0f}%")
    print(f"  ATR周期: {ATR_PERIOD}日, 风险敞口: {RISK_PER_TRADE*100:.0f}%")
    print(f"  单ETF最大持仓: {MAX_POSITION_PCT*100:.0f}%")
    print(f"{'='*50}")

    # V2新增：预计算大盘择时信号
    # 优先使用沪深300(510310)，若数据缺失则使用富时A50(512150)替代
    market_timing = pd.Series(1, index=pd.DatetimeIndex([]))  # 默认值：允许交易
    MARKET_BENCHMARK_CODE = '510310'
    MARKET_BENCHMARK_NAME = '沪深300'
    print(f"\n加载大盘基准...")
    try:
        market_df = load_etf_data(MARKET_BENCHMARK_CODE)
        print(f"使用 {MARKET_BENCHMARK_CODE} ({MARKET_BENCHMARK_NAME}) 作为大盘基准")
    except Exception:
        MARKET_BENCHMARK_CODE = '512150'
        MARKET_BENCHMARK_NAME = '富时A50'
        try:
            market_df = load_etf_data(MARKET_BENCHMARK_CODE)
            print(f"沪深300数据缺失，使用 {MARKET_BENCHMARK_CODE} ({MARKET_BENCHMARK_NAME}) 作为大盘基准替代")
        except Exception as e:
            print(f"所有大盘基准数据加载失败: {e}，大盘择时过滤将被禁用")
            market_df = None

    if market_df is not None:
        market_timing = calculate_market_timing(market_df['收盘'], MARKET_MA_PERIOD)
        # 前向填充：缺失日期继承最近已知信号，避免日期不匹配时漏过过滤
        market_timing = market_timing.ffill()
        # 填充开头的NaN为1（默认允许交易）
        market_timing = market_timing.fillna(1)
        valid_timing = market_timing.dropna()
        print(f"大盘择时信号已计算: {valid_timing.index[0].strftime('%Y-%m-%d')} 起, 共{len(valid_timing)}个交易日")

    all_results = []

    # 遍历每只ETF
    for code, name in ETFS.items():
        print(f"\n{'='*50}")
        print(f"处理 {code} {name}")
        print('='*50)

        try:
            df = load_etf_data(code)
            print(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}, 共{len(df)}行")
        except Exception as e:
            print(f"加载数据失败: {e}")
            continue

        close_prices = df['收盘']

        # V2: 预计算ATR
        atr_series = calculate_atr(df['最高'], df['最低'], df['收盘'], ATR_PERIOD)

        # 遍历每种均线组合
        for slow_p, fast_p in MA_COMBINATIONS:
            print(f"\n均线组合: 慢线{slow_p}日 vs 快线{fast_p}日")

            signals, slow_ma, fast_ma = calculate_ma_signal(close_prices, slow_p, fast_p)

            # V2: 传入ATR和大盘择时
            trades_df, metrics = simulate_trading(
                close_prices, signals, atr_series, market_timing,
                code, slow_p, fast_p
            )

            trend_metrics = calculate_trend_strength(close_prices, slow_ma, fast_ma)

            result = {
                '代码': code,
                '名称': name,
                '慢线周期': slow_p,
                '快线周期': fast_p,
                **metrics,
                **trend_metrics,
            }
            all_results.append(result)

            # 保存调仓记录
            if not trades_df.empty and len(trades_df) > 0:
                trade_file = os.path.join(OUTPUT_DIR, f'trades_v2_{code}_{slow_p}x{fast_p}.csv')
                trades_df.to_csv(trade_file, index=False, encoding='utf-8-sig')
                print(f"  调仓记录已保存: trades_v2_{code}_{slow_p}x{fast_p}.csv ({len(trades_df)}条)")

            # 打印关键指标
            dd_flag = metrics.get('回撤止损触发', '否')
            dd_info = f"  ⚠回撤止损" if dd_flag == '是' else ""
            print(f"  年化收益: {metrics['年化收益率']:.2f}%  夏普: {metrics['夏普比率']:.2f}  回撤: {metrics['最大回撤']:.2f}%  成本: {metrics['总交易成本']:.0f}元{dd_info}")

    # 生成绩效矩阵CSV
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(OUTPUT_DIR, 'performance_matrix_v2.csv')
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n保存绩效矩阵: performance_matrix_v2.csv")

    # 生成热力图
    generate_heatmaps(results_df)
    generate_trend_heatmap(results_df)

    # 打印汇总报告
    print_summary(results_df)

    return results_df


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == '__main__':
    main()
