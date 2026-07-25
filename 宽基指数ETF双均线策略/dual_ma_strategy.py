#!/usr/bin/env python3
"""
ETF双均线滚动策略回测
====================

策略原理：
    基于双均线交叉信号的趋势跟踪策略。当短周期均线（慢线）上穿长周期均线（快线）时，
    产生买入信号（金叉）；当短周期均线下穿长周期均线时，产生卖出信号（死叉）。

功能说明：
    - 支持9只宽基ETF的回测
    - 测试15种均线周期组合
    - 包含真实的交易成本（佣金+滑点）计算
    - 生成绩效矩阵、热力图、调仓记录

作者：ETF策略研究
日期：2026-07-19
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，支持无GUI环境
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 配置参数区域
# =============================================================================

# ETF标的池：覆盖A股主要宽基指数
# 键为ETF代码，值为指数名称
ETFS = {
    '512150': '富时A50',      # 富时中国A50指数
    '510310': '沪深300',      # 沪深300指数
    '510500': '中证500',      # 中证500指数
    '512100': '中证1000',     # 中证1000指数
    '563300': '中证2000',     # 中证2000指数
    '510210': '上证指数',     # 上证综合指数
    '159943': '深证成指',     # 深证成份指数
    '159915': '创业板指',     # 创业板指数
    '588000': '科创50',       # 科创50指数
}

# 均线周期组合配置
# 格式：(慢线周期, 快线周期)
# 说明：
#   - 慢线：短周期均线（5/10/20日），对价格变化反应快
#   - 快线：长周期均线（10/20/40/60/120日），对价格变化反应慢
#   - 金叉：慢线上穿快线 → 买入信号
#   - 死叉：慢线下穿快线 → 卖出信号
# 共15种组合
MA_COMBINATIONS = [
    (5, 10), (5, 20), (5, 40), (5, 60), (5, 120),    # 5日慢线 vs 各快线
    (10, 20), (10, 40), (10, 60), (10, 120),           # 10日慢线 vs 各快线
    (20, 40), (20, 60), (20, 120),                      # 20日慢线 vs 各快线
    (40, 60), (40, 120),                                 # 40日慢线 vs 各快线
    (60, 120),                                           # 60日慢线 vs 120日快线
]

# 交易成本参数（基于中国A股ETF实际费率）
COMMISSION_RATE = 0.00003   # 佣金费率：单方万分之0.3（买卖双边均收取）
SLIPPAGE_RATE = 0.0005      # 滑点假设：0.05%/笔（模拟市价单冲击成本）
TOTAL_COST_PER_SIDE = COMMISSION_RATE + SLIPPAGE_RATE  # 单边总成本约0.053%

# 无风险利率（年化），用于计算夏普比率
# 采用中国10年期国债收益率作为参考
RISK_FREE_RATE = 0.025  # 2.5%

# 文件路径配置
DATA_DIR = '/home/jintz/ETF/data/etf'    # ETF历史数据目录
OUTPUT_DIR = '/home/jintz/ETF/output'    # 回测结果输出目录


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
                   包含列：开盘、收盘、最高、最低、成交量、成交额等
    """
    filepath = os.path.join(DATA_DIR, f'{code}.csv')
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    
    # 转换日期格式并设为索引
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.set_index('日期')
    
    # 按时间升序排列
    df = df.sort_index()
    return df


# =============================================================================
# 信号计算模块
# =============================================================================

def calculate_ma_signal(close_prices, slow_period, fast_period):
    """
    计算双均线交叉信号
    
    参数：
        close_prices: 收盘价序列（pd.Series）
        slow_period: 慢线周期（短周期，如5/10/20日），对价格反应快
        fast_period: 快线周期（长周期，如40/60/120日），对价格反应慢
        
    返回：
        tuple: (信号序列, 慢线序列, 快线序列)
               信号值：1=买入（金叉），0=卖出（死叉），NaN=无信号
               
    信号逻辑：
        金叉：慢线从下方穿越快线（慢线由小于等于变为大于快线）
        死叉：慢线从上方穿越快线（慢线由大于等于变为小于快线）
    """
    # 计算移动平均线
    slow_ma = close_prices.rolling(window=slow_period).mean()  # 短周期均线
    fast_ma = close_prices.rolling(window=fast_period).mean()  # 长周期均线
    
    # 初始化信号序列（NaN表示无信号）
    signals = pd.Series(np.nan, index=close_prices.index)
    
    # 金叉条件：当前慢线>快线，且前一日慢线<=快线
    golden_cross = (slow_ma > fast_ma) & (slow_ma.shift(1) <= fast_ma.shift(1))
    
    # 死叉条件：当前慢线<快线，且前一日慢线>=快线
    death_cross = (slow_ma < fast_ma) & (slow_ma.shift(1) >= fast_ma.shift(1))
    
    # 标记信号
    signals[golden_cross] = 1   # 买入信号
    signals[death_cross] = 0    # 卖出信号
    
    return signals, slow_ma, fast_ma


# =============================================================================
# 交易模拟模块
# =============================================================================

def simulate_trading(close_prices, signals, code, slow_p, fast_p):
    """
    模拟交易执行，计算损益和交易成本
    
    参数：
        close_prices: 收盘价序列
        signals: 信号序列（1=买，0=卖，NaN=无）
        code: ETF代码
        slow_p: 慢线周期
        fast_p: 快线周期
        
    返回：
        tuple: (交易记录DataFrame, 绩效指标字典)
        
    交易规则：
        1. 金叉买入：按收盘价买入，股数取100的整数倍
        2. 死叉卖出：按收盘价卖出全部持仓
        3. 交易成本：买入和卖出时分别收取佣金和滑点成本
        4. 最后一天：如有持仓，按收盘价强制平仓
        5. 风控：最后一天有买入信号时跳过，避免开仓后立即平仓
    """
    trades = []           # 交易记录列表
    position = 0          # 持仓状态：0=空仓，1=持仓
    entry_price = 0       # 买入价格
    entry_buy_cost = 0    # 买入时的交易成本（用于计算真实盈亏）
    cash = 1000000        # 初始资金：100万元
    shares = 0            # 持仓股数
    total_cost = 0        # 累计交易成本
    
    # 获取最后一个交易日，用于处理到期平仓
    last_date = close_prices.index[-1]
    
    # 遍历每个交易日
    for i in range(len(signals)):
        date = signals.index[i]
        price = close_prices.iloc[i]
        signal = signals.iloc[i]
        
        # 跳过无信号的交易日
        if pd.isna(signal):
            continue
            
        # 风控：最后一天有买入信号时跳过（避免开仓后立即平仓）
        if signal == 1 and date == last_date:
            continue
            
        # ========== 买入逻辑 ==========
        if signal == 1 and position == 0:
            # 计算可买股数（100股整数倍）
            cost_per_share = price * (1 + TOTAL_COST_PER_SIDE)  # 含成本的每股成本
            shares_to_buy = int(cash / cost_per_share / 100) * 100
            
            if shares_to_buy > 0:
                # 计算交易成本
                trade_amount = shares_to_buy * price           # 交易金额
                commission = trade_amount * COMMISSION_RATE     # 佣金
                slippage = trade_amount * SLIPPAGE_RATE         # 滑点成本
                total_trade_cost = commission + slippage        # 本次总成本
                
                # 更新账户状态
                cash -= (trade_amount + total_trade_cost)  # 扣除交易金额和成本
                shares = shares_to_buy
                position = 1
                entry_price = price
                entry_buy_cost = total_trade_cost  # 记录买入成本（用于计算卖出时的真实盈亏）
                total_cost += total_trade_cost
                
                # 记录交易
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
                
        # ========== 卖出逻辑 ==========
        elif signal == 0 and position == 1:
            # 计算卖出收入和成本
            trade_amount = shares * price           # 卖出金额
            commission = trade_amount * COMMISSION_RATE     # 佣金
            slippage = trade_amount * SLIPPAGE_RATE         # 滑点成本
            total_trade_cost = commission + slippage        # 本次总成本
            
            # 盈亏计算：扣除双边成本（买入成本+卖出成本）
            gross_pnl = (price - entry_price) * shares                    # 毛利
            net_pnl = gross_pnl - entry_buy_cost - total_trade_cost      # 净利（扣除双边成本）
            
            # 更新账户状态
            cash += (trade_amount - total_trade_cost)  # 卖出收入减去成本
            position = 0
            total_cost += total_trade_cost
            
            # 计算收益率（基于总投入=买入金额+买入成本）
            total_invested = entry_price * shares + entry_buy_cost
            return_pct = net_pnl / total_invested * 100 if total_invested > 0 else 0
            
            # 记录交易
            trades.append({
                '日期': date.strftime('%Y-%m-%d'),
                '代码': code,
                '方向': '卖出',
                '价格': round(price, 4),
                '成交量': shares,
                '交易金额': round(trade_amount, 2),
                '佣金': round(commission, 2),
                '滑点成本': round(slippage, 2),
                '总成本': round(total_trade_cost, 2),
                '持仓市值': 0,
                '累计成本': round(total_cost, 2),
                '信号': '死叉',
                '慢线周期': slow_p,
                '快线周期': fast_p,
                '盈亏': round(net_pnl, 2),
                '收益率': round(return_pct, 4),
            })
            
            # 重置持仓状态
            shares = 0
            entry_buy_cost = 0
    
    # ========== 到期平仓逻辑 ==========
    # 如果回测结束时仍有持仓，按最后收盘价强制平仓
    if position == 1:
        last_price = close_prices.iloc[-1]
        actual_last_date = close_prices.index[-1]
        
        # 检查今天是否已有卖出信号（避免重复交易）
        today_has_sell = any(
            t['日期'] == actual_last_date.strftime('%Y-%m-%d') and t['方向'] == '卖出' 
            for t in trades
        )
        
        if not today_has_sell:
            # 计算卖出收入和成本
            trade_amount = shares * last_price
            commission = trade_amount * COMMISSION_RATE
            slippage = trade_amount * SLIPPAGE_RATE
            total_trade_cost = commission + slippage
            
            # 盈亏计算
            gross_pnl = (last_price - entry_price) * shares
            net_pnl = gross_pnl - entry_buy_cost - total_trade_cost
            cash += (trade_amount - total_trade_cost)
            total_cost += total_trade_cost
            
            # 收益率计算
            total_invested = entry_price * shares + entry_buy_cost
            return_pct = net_pnl / total_invested * 100 if total_invested > 0 else 0
            
            # 记录平仓交易
            trades.append({
                '日期': actual_last_date.strftime('%Y-%m-%d'),
                '代码': code,
                '方向': '平仓',
                '价格': round(last_price, 4),
                '成交量': shares,
                '交易金额': round(trade_amount, 2),
                '佣金': round(commission, 2),
                '滑点成本': round(slippage, 2),
                '总成本': round(total_trade_cost, 2),
                '持仓市值': 0,
                '累计成本': round(total_cost, 2),
                '信号': '到期平仓',
                '慢线周期': slow_p,
                '快线周期': fast_p,
                '盈亏': round(net_pnl, 2),
                '收益率': round(return_pct, 4),
            })
    
    # 转换为DataFrame
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    # 计算绩效指标
    metrics = calculate_performance(close_prices, trades_df, cash, total_cost)
    
    return trades_df, metrics


# =============================================================================
# 绩效计算模块
# =============================================================================

def calculate_performance(close_prices, trades_df, final_cash, total_cost):
    """
    计算策略绩效指标
    
    参数：
        close_prices: 原始收盘价序列
        trades_df: 交易记录DataFrame
        final_cash: 回测结束时的现金余额
        total_cost: 累计交易成本
        
    返回：
        dict: 包含所有绩效指标的字典
        
    指标说明：
        - 总收益率：策略整体收益百分比
        - 年化收益率：复合年化增长率（CAGR）
        - 年化波动率：日收益率标准差 × √252
        - 夏普比率：(年化收益-无风险利率)/年化波动率
        - 最大回撤：从峰值到谷值的最大跌幅
        - 卡玛比率：年化收益/最大回撤（衡量风险调整后收益）
        - 买入持有年化：同期买入持有策略的年化收益
        - 超额收益：策略年化收益 - 买入持有年化收益
    """
    initial_capital = 1000000  # 初始资金
    
    # 总收益率 = (最终资金 - 初始资金) / 初始资金
    total_return = (final_cash - initial_capital) / initial_capital
    
    # 计算回测年数
    days = (close_prices.index[-1] - close_prices.index[0]).days
    years = days / 365.25
    
    # 复合年化收益率（CAGR）
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # 买入持有基准收益（用于对比）
    buy_hold_return = (close_prices.iloc[-1] / close_prices.iloc[0]) - 1
    buy_hold_annual = (1 + buy_hold_return) ** (1/years) - 1 if years > 0 else 0
    
    # 日收益率序列（用于计算波动率和最大回撤）
    daily_returns = close_prices.pct_change().dropna()
    
    # 年化波动率 = 日收益率标准差 × √252（假设252个交易日/年）
    annual_volatility = daily_returns.std() * np.sqrt(252)
    
    # 夏普比率 = (年化收益 - 无风险利率) / 年化波动率
    excess_return = annual_return - RISK_FREE_RATE
    sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
    
    # 最大回撤计算
    # 1. 计算净值曲线：(1 + 日收益率).cumprod()
    # 2. 计算历史最高净值：cummax.cummax()
    # 3. 计算回撤：(当前净值 / 历史最高净值) - 1
    cummax = (1 + daily_returns).cumprod()
    drawdown = cummax / cummax.cummax() - 1
    max_drawdown = drawdown.min()  # 取最小值（最大回撤为负数）
    
    # 卡玛比率 = 年化收益 / |最大回撤|
    # 衡量每承受1%回撤能获得多少年化收益
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # 超额收益（相对买入持有策略）
    excess_over_buyhold = annual_return - buy_hold_annual
    
    # ========== 交易统计 ==========
    if not trades_df.empty and len(trades_df) > 0:
        # 筛选出卖出/平仓交易（完整交易对）
        sell_trades = trades_df[trades_df['方向'].isin(['卖出', '平仓'])]
        num_trades = len(sell_trades)  # 交易次数
        
        # 计算持仓天数
        buy_dates = trades_df[trades_df['方向'] == '买入']['日期'].values
        sell_dates = sell_trades['日期'].values
        
        holding_days = 0
        for i in range(min(len(buy_dates), len(sell_dates))):
            delta = pd.Timestamp(sell_dates[i]) - pd.Timestamp(buy_dates[i])
            holding_days += delta.days
        
        # 持仓时间占比 = 持仓天数 / 总回测天数
        total_days = (close_prices.index[-1] - close_prices.index[0]).days
        holding_ratio = holding_days / total_days if total_days > 0 else 0
        
        # 盈亏统计
        if '盈亏' in sell_trades.columns:
            wins = sell_trades[sell_trades['盈亏'] > 0]        # 盈利交易
            losses = sell_trades[sell_trades['盈亏'] <= 0]     # 亏损交易
            
            # 胜率 = 盈利交易次数 / 总交易次数
            win_rate = len(wins) / len(sell_trades) if len(sell_trades) > 0 else 0
            
            # 平均盈利和平均亏损
            avg_win = wins['盈亏'].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses['盈亏'].mean()) if len(losses) > 0 else 0
            
            # 盈亏比 = 平均盈利 / 平均亏损
            # 当无亏损交易时，返回9999.99（避免inf）
            profit_factor = avg_win / avg_loss if avg_loss > 0 else (9999.99 if avg_win > 0 else 0)
        else:
            win_rate = 0
            profit_factor = 0
    else:
        num_trades = 0
        holding_ratio = 0
        win_rate = 0
        profit_factor = 0
    
    # 组装绩效指标字典
    metrics = {
        # 收益指标
        '总收益率': round(total_return * 100, 4),           # 总收益率(%)
        '年化收益率': round(annual_return * 100, 4),        # 年化收益率(%)
        
        # 风险指标
        '年化波动率': round(annual_volatility * 100, 4),    # 年化波动率(%)
        '最大回撤': round(max_drawdown * 100, 4),           # 最大回撤(%)
        
        # 风险调整收益指标
        '夏普比率': round(sharpe_ratio, 4),                 # 夏普比率
        '卡玛比率': round(calmar_ratio, 4),                 # 卡玛比率
        
        # 交易统计
        '交易次数': num_trades,                              # 完整交易次数
        '持仓时间占比': round(holding_ratio * 100, 4),      # 持仓时间占比(%)
        '胜率': round(win_rate * 100, 4),                   # 胜率(%)
        '盈亏比': round(profit_factor, 4),                  # 盈亏比
        
        # 成本统计
        '总交易成本': round(total_cost, 2),                 # 累计交易成本(元)
        
        # 基准对比
        '买入持有年化': round(buy_hold_annual * 100, 4),    # 买入持有年化收益(%)
        '超额收益(年化)': round(excess_over_buyhold * 100, 4),  # 超额收益(年化, %)
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
        
    指标说明：
        - 平均趋势持续天数：每次趋势（上升/下降）平均持续多少天
        - 上升趋势占比：上升趋势天数占总天数的比例
        - 平均偏离度：价格相对于快线的平均偏离百分比
        - 最大偏离度：价格相对于快线的最大偏离百分比
        - 趋势变化次数：趋势方向改变的总次数
    """
    # 判断趋势方向：慢线在快线上方为上升趋势(1)，否则为下降趋势(0)
    trend = (slow_ma > fast_ma).astype(int)
    
    # 计算趋势变化点（趋势方向改变的位置）
    trend_changes = trend.diff().abs()
    
    # 计算趋势持续期
    # 方法：当趋势发生变化时，trend_groups会加1，从而将连续相同趋势分为一组
    trend_groups = (trend != trend.shift()).cumsum()
    
    # 统计每个趋势段的持续天数
    trend_lengths = trend.groupby(trend_groups).agg(['count', 'first', 'last'])
    trend_lengths.columns = ['持续天数', '趋势开始', '趋势结束']
    trend_lengths['趋势方向'] = trend.groupby(trend_groups).first().values
    
    # 平均趋势持续天数
    avg_trend_length = trend_lengths['持续天数'].mean()
    
    # 上升趋势占比（trend=1的比例）
    up_ratio = trend.mean()
    
    # 趋势强度：价格相对于快线的偏离度
    # 偏离度越大，说明趋势越强
    price_deviation = ((close_prices - fast_ma) / fast_ma * 100).abs()
    avg_deviation = price_deviation.mean()   # 平均偏离度
    max_deviation = price_deviation.max()    # 最大偏离度
    
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
    生成绩效热力图（2×2布局）
    
    包含4张热力图：
        1. 夏普比率：衡量风险调整后收益
        2. 年化收益率：策略绝对收益
        3. 最大回撤：风险水平
        4. 卡玛比率：收益/风险效率
    
    参数：
        results_df: 绩效矩阵DataFrame
    """
    # 设置中文字体（如不可用则使用DejaVu Sans）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 创建2×2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 生成4张热力图
    for idx, (metric, title, cmap, fmt) in enumerate([
        ('夏普比率', '夏普比率', 'RdYlGn', '.2f'),        # 绿色=好，红色=差
        ('年化收益率', '年化收益率(%)', 'RdYlGn', '.1f'),
        ('最大回撤', '最大回撤(%)', 'RdYlGn_r', '.1f'),   # 注意：回撤用反转色图
        ('卡玛比率', '卡玛比率', 'RdYlGn', '.2f'),
    ]):
        ax = axes[idx // 2][idx % 2]  # 2×2布局索引
        
        # 构建透视表：行=ETF名称，列=均线组合
        pivot = results_df.pivot_table(
            index='名称',
            columns=['慢线周期', '快线周期'],
            values=metric
        )
        
        # 绘制热力图
        # center=0 表示以0为中心（正值绿色，负值红色）
        sns.heatmap(pivot, ax=ax, cmap=cmap, annot=True, fmt=fmt, 
                   center=0 if metric != '最大回撤' else None)
        ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_performance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存热力图: heatmap_performance.png")


def generate_trend_heatmap(results_df):
    """
    生成趋势强度热力图（1×2布局）
    
    包含2张热力图：
        1. 平均趋势持续天数：趋势持续时间越长越好
        2. 上升趋势占比：上升趋势占比越高越好
    
    参数：
        results_df: 绩效矩阵DataFrame
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (metric, title, cmap) in enumerate([
        ('平均趋势持续天数', '平均趋势持续天数(天)', 'YlOrRd'),  # 黄色=短，红色=长
        ('上升趋势占比', '上升趋势占比(%)', 'RdYlGn'),          # 红色=低，绿色=高
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
    plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_trend.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存趋势热力图: heatmap_trend.png")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """
    主函数：执行完整的回测流程
    
    流程：
        1. 创建输出目录
        2. 遍历9只ETF
        3. 对每只ETF，遍历15种均线组合
        4. 计算信号 → 模拟交易 → 计算绩效
        5. 保存调仓记录
        6. 生成绩效矩阵和热力图
        7. 打印汇总报告
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 打印配置信息
    print(f"均线组合数: {len(MA_COMBINATIONS)}")
    print(f"交易成本: 佣金{COMMISSION_RATE*10000:.1f}%%/单方 + 滑点{SLIPPAGE_RATE*100:.2f}%%")
    
    all_results = []  # 存储所有回测结果
    
    # 遍历每只ETF
    for code, name in ETFS.items():
        print(f"\n{'='*50}")
        print(f"处理 {code} {name}")
        print('='*50)
        
        # 加载ETF数据
        try:
            df = load_etf_data(code)
            print(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}, 共{len(df)}行")
        except Exception as e:
            print(f"加载数据失败: {e}")
            continue
        
        # 提取收盘价序列
        close_prices = df['收盘']
        
        # 遍历每种均线组合
        for slow_p, fast_p in MA_COMBINATIONS:
            print(f"\n均线组合: 慢线{slow_p}日 vs 快线{fast_p}日")
            
            # 1. 计算信号
            signals, slow_ma, fast_ma = calculate_ma_signal(close_prices, slow_p, fast_p)
            
            # 2. 模拟交易
            trades_df, metrics = simulate_trading(close_prices, signals, code, slow_p, fast_p)
            
            # 3. 趋势强度分析
            trend_metrics = calculate_trend_strength(close_prices, slow_ma, fast_ma)
            
            # 4. 汇总结果
            result = {
                '代码': code,
                '名称': name,
                '慢线周期': slow_p,
                '快线周期': fast_p,
                **metrics,
                **trend_metrics,
            }
            all_results.append(result)
            
            # 5. 保存调仓记录到CSV
            if not trades_df.empty and len(trades_df) > 0:
                trade_file = os.path.join(OUTPUT_DIR, f'trades_{code}_{slow_p}x{fast_p}.csv')
                trades_df.to_csv(trade_file, index=False, encoding='utf-8-sig')
                print(f"  调仓记录已保存: trades_{code}_{slow_p}x{fast_p}.csv ({len(trades_df)}条)")
            
            # 打印本次回测关键指标
            print(f"  年化收益: {metrics['年化收益率']:.2f}%  夏普: {metrics['夏普比率']:.2f}  回撤: {metrics['最大回撤']:.2f}%  成本: {metrics['总交易成本']:.0f}元")
    
    # 生成绩效矩阵CSV
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(OUTPUT_DIR, 'performance_matrix.csv')
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n保存绩效矩阵: performance_matrix.csv")
    
    # 生成热力图
    generate_heatmaps(results_df)
    generate_trend_heatmap(results_df)
    
    # 打印汇总报告
    print_summary(results_df)
    
    return results_df


def print_summary(results_df):
    """
    打印回测汇总报告
    
    包含：
        1. 各ETF最优参数（按夏普比率排序）
        2. 全局最优组合TOP 10
    """
    print("\n" + "="*80)
    print("策略回测汇总")
    print("="*80)
    
    # 按ETF汇总最优参数
    print("\n【各ETF最优参数（按夏普比率排序）】")
    for code in results_df['代码'].unique():
        etf_data = results_df[results_df['代码'] == code]
        
        # 找到夏普比率最高的组合
        best = etf_data.loc[etf_data['夏普比率'].idxmax()]
        
        print(f"\n{code} {best['名称']}:")
        print(f"  最优组合: 慢线{best['慢线周期']}日 vs 快线{best['快线周期']}日")
        print(f"  年化收益: {best['年化收益率']:.2f}%  夏普: {best['夏普比率']:.2f}  卡玛: {best['卡玛比率']:.2f}")
        print(f"  最大回撤: {best['最大回撤']:.2f}%  胜率: {best['胜率']:.1f}%  盈亏比: {best['盈亏比']:.2f}")
        print(f"  买入持有年化: {best['买入持有年化']:.2f}%  超额收益(年化): {best['超额收益(年化)']:.2f}%")
        print(f"  交易成本: {best['总交易成本']:.0f}元  交易次数: {best['交易次数']}")
    
    # 全局最优组合TOP 10
    print("\n【全局最优组合 TOP 10】")
    top10 = results_df.nlargest(10, '夏普比率')
    
    # 打印表头
    print(f"  {'代码':>6s} {'名称':6s} {'组合':>6s} {'夏普':>6s} {'年化':>8s} {'回撤':>8s} {'超额':>8s}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")
    
    # 打印每一行
    for _, row in top10.iterrows():
        combo = f"{int(row['慢线周期'])}x{int(row['快线周期'])}"
        print(f"  {row['代码']:>6s} {row['名称']:6s} {combo:>6s} {row['夏普比率']:>6.2f} {row['年化收益率']:>7.2f}% {row['最大回撤']:>7.2f}% {row['超额收益(年化)']:>7.2f}%")


# =============================================================================
# 程序入口
# =============================================================================

if __name__ == '__main__':
    main()
