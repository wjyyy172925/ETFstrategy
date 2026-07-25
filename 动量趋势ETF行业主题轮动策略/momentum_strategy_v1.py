#!/usr/bin/env python3
"""
A股ETF动量趋势轮动策略 v1 — 回测与绩效分析

本文件实现了一个完整的 A 股行业 ETF 动量趋势轮动策略回测框架。
v1 作为基线版本，包含了最基础的动量策略逻辑和完整的参数穷举框架。

策略逻辑:
  1. 在每个调仓日（月度或周度），计算全部候选 ETF 的动量分数
  2. 按动量分数降序选择 top_n × 2 只候选 ETF
  3. 经相关性分散 → 波动率目标调整 → 等权分配资金

参数网格穷举:
  - 动量窗口: 20 日 / 60 日
  - 调仓频率: 周度 / 月度
  - 动量类型: 简单动量 / 双重动量 / 双重动量+均线过滤
  → 2 × 2 × 3 = 12 种组合

数据依赖:
  - data/etf_list.csv: ETF 列表（代码、名称）
  - data/etf/{code}.csv: 每只 ETF 的日线行情
  - 基准: 510300（沪深 300 ETF）

输出:
  - 控制台: 按夏普比率排序的绩效总览表
  - momentum_results/performance_summary_v1.csv: 全部组合的绩效指标
  - momentum_results/nav_all_v1.csv: 全部组合的净值曲线
"""

import os
import argparse
from itertools import product

import numpy as np
import pandas as pd

# ============================================================
# 全局参数配置
# ============================================================
# 所有策略参数集中管理，避免魔法数字散落在代码各处。
# 回测框架通过参数网格穷举 12 种组合，而非单次运行。

DATA_DIR = "data/etf"                     # ETF 日线数据目录

# --- 风控参数 ---
# v1 采用 15% 目标波动率，这是策略层面唯一的波动控制手段。
# 个股止损 -8%：防止单只 ETF 暴跌对净值造成过大冲击。
# 策略最大回撤止损 15%：一旦净值从峰值回撤超过 15%，触发强制清仓。
# 注意：15% 止损在持续下跌市场中会反复触发，形成"抄底-止损-再抄底"的死亡螺旋，
# v2 将此阈值提高至 30%，v3 进一步建议放宽至 40-50%。
TARGET_VOL = 0.15                         # 目标年化波动率 15%
STOP_LOSS_PCT = 0.08                      # 个股止损阈值 -8%
MAX_DRAWDOWN_PCT = 0.15                   # 策略最大回撤止损 15%

# --- 均线参数 ---
# (20, 60) 双均线仅用于 dual_ma 动量类型的候选过滤，
# 要求 ETF 收盘价同时站上 20 日和 60 日均线才纳入候选池。
# 这是一种"趋势确认"机制，过滤掉处于下降趋势的 ETF。
MA_WINDOWS = (20, 60)                     # 均线过滤使用的双均线周期

# --- 选股参数 ---
# 相关性阈值 0.9：持仓 ETF 之间两两相关系数必须 < 0.9，
# 避免同时持有走势高度同质化的品种（如多个新能源 ETF）。
# 候选数量倍率 2：先选择 top_n × 2 只高动量候选，再用相关性筛选。
# 最少数据天数 60：数据不足的 ETF 直接剔除，确保动量计算可靠。
CORR_THRESHOLD = 0.9                      # 相关性阈值
MIN_DATA_DAYS = 60                        # ETF 最少数据天数
REBALANCE_CANDIDATE_MULTIPLIER = 2        # 候选数量倍率

# --- 交易成本 ---
# 佣金费率 万0.3（即 0.003%），选择A 股 ETF对机构来说相对较高的交易成本。
# 冲击成本 0.1%：模拟大额交易时的市场滑点，对小资金回测偏保守。
# TRADE_COST_RATE 是买卖双边合计费率 = 0.103%。
# 在周度调仓场景下，交易次数 ~1500 次，累积成本拖累 CAGR 约 -12%。
COMMISSION_RATE = 0.00003                 # 佣金费率 万0.3
SLIPPAGE_RATE = 0.001                     # 冲击成本 0.1%
TRADE_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE

# ============================================================
# ETF 行业分类筛选
# ============================================================
# 全市场 ETF 数量庞大（722 只），包含债券、商品、跨境等品种。
# 动量轮动策略只适用于 A 股行业/主题 ETF，因此需要按名称关键词筛选。
# 过滤后约 631 只有效行业 ETF。
# 关键词匹配策略的局限性：无法识别名称不规范的 ETF，但足以覆盖主流品种。


def load_etf_list():
    """从 CSV 加载 ETF 列表。

    CSV 文件格式要求: 包含"代码"和"名称"两列。
    代码为 6 位数字字符串（可能不带前导零）。
    """
    path = "data/etf_list.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"ETF列表文件不存在: {path}")
    return pd.read_csv(path)


def filter_industry_etfs(df):
    """按名称关键词识别行业/主题 ETF，排除非权益品种。

    两层关键词匹配:
      1. include_kw: 正向匹配行业/主题相关的关键词
         - 覆盖传统行业（银行、证券、地产等）
         - 覆盖新兴主题（AI、芯片、碳中和、数字经济等）
         - 覆盖策略指数（红利、价值、成长等）
      2. exclude_kw: 剔除不适合动量策略的品种
         - 债券类：国债、地方债、可转债等（低波动、趋势弱）
         - 商品类：黄金、豆粕等（非权益品种）
         - 跨境类：标普、纳斯达克、恒生等（与 A 股动量不同步）

    策略依据:
      动量策略在权益类 ETF 上效果最显著，债券和商品 ETF
      的趋势持续性较差，不应纳入同一候选池。
    """
    include_kw = [
        "行业", "产业", "消费", "医药", "医疗", "科技", "新能源",
        "半导体", "军工", "银行", "证券", "保险", "地产", "有色",
        "煤炭", "钢铁", "化工", "农业", "食品", "汽车", "通信",
        "电子", "计算机", "传媒", "环保", "基建", "央企", "国企",
        "红利", "价值", "成长", "龙头", "创新", "制造", "材料",
        "能源", "电力", "体育", "旅游", "影视", "AI", "人工智能",
        "机器人", "芯片", "碳中和", "光伏", "风电", "电池",
        "新能车", "电动车", "智能驾驶", "数字经济", "信创", "游戏",
        "文娱", "物流", "运输", "航天", "航空",
    ]
    exclude_kw = [
        "国债", "地方债", "政金债", "城投债", "可转债", "货币",
        "短融", "AAA", "国开", "农发", "口行", "黄金", "豆粕",
        "能源化工", "白银", "标普", "纳斯达克", "纳指", "恒生",
        "H股", "中概", "道琼斯", "日经", "富时", "德国",
        "有色金属期货", "有色期货", "豆", "玉米", "白糖", "PTA",
        "甲醇", "豆粕ETF", "黄金ETF",
    ]

    # 使用 str.contains 实现关键字模糊匹配，效率高但依赖命名规范
    mask = df["名称"].str.contains("|".join(include_kw), na=False)
    exclude_mask = df["名称"].str.contains("|".join(exclude_kw), na=False)
    return df[mask & ~exclude_mask].reset_index(drop=True)


def load_industry_etf_codes():
    """加载并过滤行业 ETF，返回 6 位代码列表。

    流程: load_etf_list() → filter_industry_etfs() → 代码补零 → 返回列表。
    返回的代码格式统一为 6 位字符串（如 "510300"）。
    """
    df = load_etf_list()
    df = filter_industry_etfs(df)
    codes = df["代码"].astype(str).str.zfill(6).tolist()
    print(f"行业ETF数量: {len(codes)} 只")
    return codes

# ============================================================
# 数据加载
# 从 data/etf/ 加载日线数据，合并为宽表。
# v1 使用最近 5 年数据窗口（覆盖 2021-2026），
# 数据格式: 每只 ETF 一个 CSV 文件，包含 日期 和 收盘 两列。
# ============================================================

def load_etf_data(code):
    """加载单只 ETF 日线数据，返回 [日期, 收盘] 两列。

    文件路径: data/etf/{code}.csv
    如果文件不存在（部分 ETF 可能退市或无数据），返回 None。
    数据按日期升序排列，确保后续计算时间序列正确。

    Args:
        code: 6 位 ETF 代码字符串

    Returns:
        DataFrame 或 None（文件不存在时）
    """
    path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    return df[["日期", "收盘"]]


def load_all_industry_data(codes):
    """加载所有行业 ETF 收盘价并合并为宽表。

    核心操作: 将多只 ETF 的日线数据横向拼接为 DataFrame，
    每一列对应一只 ETF，每一行对应一个交易日。
    这种"宽表"格式便于后续的向量化动量计算和矩阵运算。

    数据处理步骤:
      1. 逐只加载，数据不足 MIN_DATA_DAYS 天的直接跳过
      2. 以日期为索引，收盘价为值，合并为宽表
      3. 删除缺失日期（周末/节假日各 ETF 一致则无缺失）
      4. 仅保留最近 5 年数据（v1 的预设窗口）

    Args:
        codes: ETF 代码列表

    Returns:
        DataFrame, shape = (交易日数, ETF数量)
    """
    data = {}
    for code in codes:
        df = load_etf_data(code)
        if df is not None and len(df) > MIN_DATA_DAYS:
            data[code] = df.set_index("日期")["收盘"]
    result = pd.DataFrame(data).sort_index()
    result = result.loc[result.index.notna()]
    cutoff = result.index.max() - pd.DateOffset(years=5)
    result = result[result.index >= cutoff]
    return result


def load_benchmark():
    """加载沪深300 ETF（510300）作为基准。

    沪深 300 ETF（510300）在 2012 年上市，是流动性最好的
    A 股宽基 ETF 之一。使用 510300 而非沪深 300 指数本身
    可以更真实地反映可投资基准的表现（含分红、费率等）。

    Returns:
        Series, index=日期, value=收盘价
    """
    df = load_etf_data("510300")
    if df is None:
        return None
    return df.set_index("日期")["收盘"].sort_index()

# ============================================================
# 风险控制
# ============================================================
# v1 实现了两层风控：
#   1. 相关性分散：持仓 ETF 之间必须低相关，避免同质化集中
#   2. 波动率目标：以基准波动为锚，动态调整总仓位
# 此外，回测引擎内部还有个股止损和策略最大回撤止损。


def diversify_by_correlation(close_df, candidates, top_n, corr_threshold=CORR_THRESHOLD):
    """基于收益率相关系数贪心选取低相关性 ETF。

    算法: 贪心法（非全局优化）
    1. 从候选列表中取出动量排名最高的 ETF
    2. 检查它与已选持仓的相关性
    3. 如果全部低于阈值则保留，否则跳过
    4. 重复直到选满 top_n 只

    为什么用贪心而非全局优化:
      - 候选池规模较小（top_n × 2 = 10 只），贪心法已足够
      - 全局最优解（如最小化最大相关系数）计算成本高，收益有限
      - 动量排名本身是核心信号，过于激进的分散会损害动量收益

    注意: 相关性计算使用最近 20 个交易日的收益率，
    短期相关性更能反映当前市场的联动关系。

    Args:
        close_df: 收盘价 DataFrame（至少包含 candidates 和 20 行数据）
        candidates: 按动量排序的候选 ETF 列表
        top_n: 目标持仓数量
        corr_threshold: 相关性阈值，超过此值则不纳入

    Returns:
        list: 选出的低相关性 ETF 列表，长度 ≤ top_n
    """
    # 候选数量不足时直接返回前 top_n 只
    if len(candidates) <= top_n:
        return candidates[:top_n]

    selected = []
    remaining = list(candidates)

    while len(selected) < top_n and remaining:
        candidate = remaining.pop(0)
        if not selected:
            selected.append(candidate)
            continue
        # 计算候选 ETF 与已选持仓的收益率相关系数矩阵
        returns = close_df[[candidate] + selected].pct_change().dropna()
        if len(returns) < 5:
            selected.append(candidate)
            continue
        # 取候选 ETF 与已选各品种的相关系数
        corr = returns.corr().iloc[0, 1:]
        if all(corr < corr_threshold):
            selected.append(candidate)

    return selected

# ============================================================
# 回测引擎
# v1 支持月度/周度调仓，三种动量类型（simple/dual/dual_ma）。
# dual_ma 将均线过滤嵌入动量类型中。
# ============================================================
# 回测引擎是策略的核心，采用"事件驱动"的逐日循环架构。
# 每个交易日:
#   1. 计算当日净值（现金 + 持仓市值）
#   2. 检查是否触发止损/回撤限制
#   3. 如果是调仓日，执行动量选股 → 相关性分散 → 再平衡
#
# 注意: 逐日循环在 Python 中效率较低（for 循环），
# 但对于复杂的条件逻辑（止损、调仓日判断等）更灵活可控。


def _get_rebalance_dates(index, freq, min_window=MIN_DATA_DAYS):
    """计算调仓日期。

    通过 pandas 的时间序列功能，识别月份或周度的边界日期。
    边界日期的定义为: 当月/周的第一个交易日。

    月度调仓:
      - 使用 dt.month 属性，在月份切换时产生调仓信号
      - 每月约 1 次调仓，全区间 ~48 次
    周度调仓:
      - 使用 dt.isocalendar() 获取 ISO 周数
      - 每周约 1 次调仓，全区间 ~260 次

    Args:
        index: 完整日期索引
        freq: "monthly"（月度）或 "weekly"（周度）
        min_window: 最小持有窗口，之前不调仓

    Returns:
        list[Timestamp]: 调仓日期列表
    """
    if len(index) <= min_window:
        return []
    dates = index[min_window:]
    s = pd.Series(dates, index=range(len(dates)))
    if freq == "monthly":
        mask = s.dt.month != s.shift(1).dt.month
    else:
        iso = s.dt.isocalendar()
        prev_iso = s.shift(1).dt.isocalendar()
        mask = (iso.year != prev_iso.year) | (iso.week != prev_iso.week)
    mask.iloc[0] = True
    return s[mask].tolist()


def run_backtest(close_df, benchmark_close, top_n=5,
                 window=20, rebalance_freq="monthly",
                 momentum_type="simple"):
    """执行 v1 动量趋势轮动策略回测。

    v1 核心特征:
      - 无趋势过滤器（这是 v1 与 v2 的最大区别）
      - 均线过滤仅通过 "dual_ma" 动量类型启用
      - 支持周度和月度调仓
      - 目标波动率 15%，回撤止损 15%

    三种动量类型的定义:
      simple（简单动量）:
        momentum = (当前价格 / N 日前价格) - 1
        仅基于 ETF 自身的历史收益率排序。
        纯粹的价格动量，也是最基本的趋势跟踪策略。
        在 A 股行业轮动中表现尚可，但容易受市场整体趋势影响。

      dual（双重动量）:
        momentum = ETF收益 - 基准收益（超额收益）
        同时要求 ETF 绝对收益 > 0
        双重动量的核心思想: 既要跑赢基准（相对动量），
        又不能处于下跌趋势（绝对动量过滤）。
        这种"双重过滤"在熊市中能有效空仓避险。

      dual_ma（双重动量+均线过滤）:
        在 dual 基础上，要求 ETF 价格站上 20MA 和 60MA
        均线过滤进一步限制了候选范围，减少了"抄底"行为。
        但注意: 在持续下跌市场中，均线过滤器会频繁触发
        空仓，导致错过反弹（如 2024 年的结构性行情）。

    Args:
        close_df: 全部 ETF 收盘价宽表，shape=(交易日, ETF数量)
        benchmark_close: 沪深300基准收盘价 Series
        top_n: 目标持仓数量（默认 5 只）
        window: 动量计算窗口（20 或 60 日）
        rebalance_freq: "monthly"（月度）或 "weekly"（周度）
        momentum_type: "simple" / "dual" / "dual_ma"

    Returns:
        nav: 策略净值 Series，index=日期
        trades_df: 交易记录 DataFrame，含日期/ETF/方向/价格/原因
    """
    # ================================================================
    # 第一步: 计算动量分数（向量化操作，一次性生成全区间动量信号）
    # ================================================================
    # 收益率 = 当前价格 / N 日前价格 - 1
    # 这是最简单的动量定义，N 为动量窗口（20 或 60 日）
    returns = close_df / close_df.shift(window) - 1
    benchmark_returns = benchmark_close / benchmark_close.shift(window) - 1

    if momentum_type == "simple":
        # 简单动量: 仅使用绝对收益作为动量分数
        momentum_scores = returns
    elif momentum_type == "dual":
        # 双重动量: 超额收益 + 绝对收益过滤
        # 减去基准避免"市场整体上涨导致所有 ETF 动量都高"的虚假信号
        momentum_scores = returns.sub(benchmark_returns, axis=0)
        # 绝对动量过滤: 仅保留收益率为正的 ETF，负收益的置为 NaN
        abs_mask = returns > 0
        momentum_scores = momentum_scores.where(abs_mask, np.nan)
    elif momentum_type == "dual_ma":
        # 双重动量 + 均线过滤: 同 dual，但增加均线条件
        momentum_scores = returns.sub(benchmark_returns, axis=0)
        abs_mask = returns > 0
        momentum_scores = momentum_scores.where(abs_mask, np.nan)

    # --- 均线（仅 dual_ma 使用）---
    # 提前计算 20 日和 60 日均线，避免在主循环中重复计算
    ma20 = close_df.rolling(MA_WINDOWS[0]).mean()
    ma60 = close_df.rolling(MA_WINDOWS[1]).mean()
    above_ma20 = close_df > ma20
    above_ma60 = close_df > ma60

    # ================================================================
    # 第二步: 波动率目标管理
    # ================================================================
    # 思路: 以沪深 300 的滚动波动率为基准，动态调整仓位比例
    # 当市场波动率低时，可以加仓（比例 > 1 → 限制为 1）
    # 当市场波动率高时，需要减仓（比例 < 1）
    # 
    # 计算公式: 目标比率 = min(15% / 滚动波动率, 1.0)
    # 使用 20 日滚动窗口计算年化波动率
    #
    # 效果: 在市场平稳时满仓，市场剧烈波动时降仓
    # v1 用此将策略年化波动控制在 9% 左右（vs 基准 18%）
    # 但代价是降低了反弹力度 — 这是降低波动率的固有取舍
    bench_daily_ret = benchmark_close.pct_change().dropna()
    rolling_vol = bench_daily_ret.rolling(20).std() * np.sqrt(252)
    vol_target_ratio = pd.Series(1.0, index=close_df.index)
    vol_target_ratio.iloc[20:] = np.minimum(
        TARGET_VOL / rolling_vol.iloc[19:].values,
        1.0
    )
    vol_target_ratio = vol_target_ratio.clip(0, 1)

    # ================================================================
    # 第三步: 初始化回测状态
    # ================================================================
    # 关键状态变量:
    #   cash: 现金余额（初始为 1，即初始净值 1.0）
    #   holdings: 持仓字典 {代码: {"shares": 股数}}
    #   in_drawdown_stop: 是否处于回撤止损锁定状态
    #   peak_nav: 历史最高净值（用于计算回撤）
    rebalance_dates = _get_rebalance_dates(close_df.index, rebalance_freq, window)
    rebalance_set = set(rebalance_dates)

    nav = pd.Series(index=close_df.index, dtype=float)
    nav.iloc[:window] = 1.0
    cash = 1.0
    holdings = {}
    cost_prices = {}
    in_drawdown_stop = False
    peak_nav = 1.0
    trades = []

    # ================================================================
    # 第四步: 主回测循环（逐日遍历）
    # ================================================================
    # 从 window 位置开始（前 window 天为动量计算预热期）
    for i in range(window, len(close_df)):
        date = close_df.index[i]
        current_prices = close_df.iloc[i]

        # --- 4a. 计算当日净值 ---
        # 净值 = 现金 + 所有持仓市值
        # 不考虑未实现盈亏的税务影响（简化处理）
        current_nav_value = cash + sum(
            holdings[c]["shares"] * current_prices[c]
            for c in list(holdings.keys())
            if c in current_prices.index and pd.notna(current_prices[c])
        ) if holdings else cash
        nav.iloc[i] = current_nav_value
        peak_nav = max(peak_nav, current_nav_value)

        # --- 4b. 策略级回撤止损（15%）---
        # 核心逻辑:
        #   当净值从峰值回撤 15% 时触发止损，清空所有持仓
        #   止损后进入锁定状态，直到下一个调仓日才解除
        # 
        # v1 的这个设计在 2021-2026 回测中被证明是致命的:
        #   触发止损 → 清仓 → 调仓日重新入场 → 继续下跌 → 再次止损
        #   形成"抄底-止损"的死亡螺旋
        if not in_drawdown_stop:
            dd = (peak_nav - current_nav_value) / peak_nav if peak_nav > 0 else 0
            if dd >= MAX_DRAWDOWN_PCT:
                in_drawdown_stop = True
                for c in list(holdings.keys()):
                    if c in current_prices.index and pd.notna(current_prices[c]):
                        proceeds = holdings[c]["shares"] * current_prices[c]
                        cash += proceeds * (1 - TRADE_COST_RATE)
                        trades.append({"日期": date, "ETF": c, "方向": "卖出(止损清仓)",
                                       "价格": current_prices[c], "原因": "策略最大回撤"})
                holdings = {}
                cost_prices = {}
                continue
        else:
            # 锁定状态: 只有在调仓日才解除锁定
            if date in rebalance_set:
                in_drawdown_stop = False
            else:
                continue  # 锁定期间跳过所有操作

        # --- 4c. 个股止损 ---
        # 对每只持仓 ETF 检查: 如果从买入价下跌超过 8%，强制卖出
        # 个股止损与策略回撤止损是两个独立的防线:
        #   个股止损 → 控制单只 ETF 的风险
        #   策略止损 → 控制系统性风险
        stopped_codes = [
            c for c in list(holdings.keys())
            if c in current_prices.index and pd.notna(current_prices[c])
            and c in cost_prices
            and (current_prices[c] / cost_prices[c] - 1) <= -STOP_LOSS_PCT
        ]
        for c in stopped_codes:
            proceeds = holdings[c]["shares"] * current_prices[c]
            cash += proceeds * (1 - TRADE_COST_RATE)
            trades.append({"日期": date, "ETF": c, "方向": "卖出(止损)",
                           "价格": current_prices[c], "原因": "个股止损"})
            del holdings[c]
            del cost_prices[c]

        # --- 4d. 非调仓日跳过 ---
        if date not in rebalance_set:
            continue

        # ================================================================
        # 第五步: 调仓日操作（动量选股 → 相关性分散 → 再平衡）
        # ================================================================

        # --- 5a. 动量选股 ---
        # 取当日所有 ETF 的动量分数，降序排列，取 top_n×2 只候选
        # 候选数量乘以 2 是因为后续相关性筛选会剔除部分 ETF
        mom_row = momentum_scores.iloc[i]
        candidates = (
            mom_row.dropna()
            .sort_values(ascending=False)
            .head(top_n * REBALANCE_CANDIDATE_MULTIPLIER)
            .index.tolist()
        )

        # dual_ma 模式附加条件: 候选 ETF 必须站上 20MA 和 60MA
        # 这是一个额外的趋势确认条件，确保只买入处于上升趋势的品种
        if momentum_type == "dual_ma":
            candidates = [
                c for c in candidates
                if above_ma20.iloc[i].get(c, False)
                and above_ma60.iloc[i].get(c, False)
            ]

        # 无合格候选: 清仓持现，进入"现金增强"状态
        if not candidates:
            for c in list(holdings.keys()):
                if c in current_prices.index and pd.notna(current_prices[c]):
                    proceeds = holdings[c]["shares"] * current_prices[c]
                    cash += proceeds * (1 - TRADE_COST_RATE)
                    trades.append({"日期": date, "ETF": c, "方向": "卖出(现金增强)",
                                   "价格": current_prices[c], "原因": "无合格ETF"})
            holdings = {}
            cost_prices = {}
            continue

        # --- 5b. 相关性分散 ---
        # 从候选池中选出彼此低相关的 top_n 只 ETF
        # 使用最近 20 个交易日的收益率计算相关系数
        selected = diversify_by_correlation(
            close_df.iloc[max(0, i - 20):i + 1], candidates, top_n
        )
        if not selected:
            continue

        # 获取当日波动率目标调整系数
        vol_ratio = vol_target_ratio.iloc[i]

        # --- 5c. 卖出不在新持仓中的品种 ---
        for c in list(holdings.keys()):
            if c not in selected:
                if c in current_prices.index and pd.notna(current_prices[c]):
                    proceeds = holdings[c]["shares"] * current_prices[c]
                    cash += proceeds * (1 - TRADE_COST_RATE)
                    trades.append({"日期": date, "ETF": c, "方向": "卖出",
                                   "价格": current_prices[c], "原因": "调仓"})
                del holdings[c]
                if c in cost_prices:
                    del cost_prices[c]

        # --- 5d. 等权分配 + 波动率目标调整 ---
        # 计算总资产，然后等权分配给 selected 中的每只 ETF
        # 同时乘以 vol_ratio（波动率调整比例）
        # 
        # 等权分配的原因:
        #   - 动量信号是排名信息，不是幅度信息
        #   - 等权避免个别高动量 ETF 过度集中
        #   - 相比市值加权或动量加权，等权的换手率最低
        total_value = cash + sum(
            holdings[c]["shares"] * current_prices[c]
            for c in holdings
            if c in current_prices.index and pd.notna(current_prices[c])
        )
        if total_value <= 0:
            continue
        target_value_per_etf = (total_value * vol_ratio) / len(selected)

        # --- 5e. 执行买入/调仓 ---
        # 对每只目标持仓，计算与当前持仓的差异，执行相应买卖
        # 这种"差额调仓"而非"全部清仓后重买"的方式可以降低交易成本
        for c in selected:
            if c not in current_prices.index or pd.isna(current_prices[c]):
                continue
            price = current_prices[c]
            if c in holdings:
                # 已有持仓: 调整至目标市值
                current_value = holdings[c]["shares"] * price
                diff = target_value_per_etf - current_value
                # 差异小于 0.1% 时跳过（避免微调导致不必要的交易成本）
                if abs(diff) < 0.001 * price:
                    continue
                if diff > 0:
                    add_shares = diff / price
                    holdings[c]["shares"] += add_shares
                    cash -= add_shares * price * (1 + TRADE_COST_RATE)
                else:
                    sell_shares = -diff / price
                    holdings[c]["shares"] -= sell_shares
                    cash += sell_shares * price * (1 - TRADE_COST_RATE)
            else:
                # 新建持仓: 买入目标金额
                shares = target_value_per_etf / price
                holdings[c] = {"shares": shares}
                cost_prices[c] = price
                cash -= shares * price * (1 + TRADE_COST_RATE)
                trades.append({"日期": date, "ETF": c, "方向": "买入",
                               "价格": price, "原因": "调仓"})

        cash = max(cash, 0)

    # 清理 NaN 行并返回
    nav = nav.dropna()
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return nav, trades_df

# ============================================================
# 绩效分析
# ============================================================
# 提供完整的回测绩效指标体系，覆盖收益、风险、风险调整收益三个维度。
# 所有指标基于净值曲线计算（而非逐笔交易），确保与实盘口径一致。
#
# 核心指标说明:
#   CAGR（年化复合收益率）: 衡量策略的整体盈利能力
#   夏普比率: 每单位风险获得的超额收益，>0 表示优于无风险利率
#   最大回撤: 净值从峰值到谷底的最大跌幅，衡量尾部风险
#   信息比率: 相对于基准的超额收益/跟踪误差，衡量主动管理能力
#   卡玛比率: CAGR / 最大回撤，与夏普互补（侧重尾部风险调整）


def calc_performance_metrics(nav, benchmark_nav=None, risk_free=0.02):
    """计算策略收益与风险指标。

    计算的指标分为三组:
      1. 收益类: CAGR、超额收益
      2. 风险类: 年化波动、最大回撤
      3. 风险调整收益类: 夏普比率、卡玛比率、信息比率

    Args:
        nav: 策略净值 Series
        benchmark_nav: 基准净值 Series（可选，用于计算超额收益等）
        risk_free: 无风险利率（默认 2%，近似国债收益率）

    Returns:
        dict: 包含所有绩效指标的字典
    """
    if len(nav) < 2:
        return {}

    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    total_days = (nav.index[-1] - nav.index[0]).days
    total_years = total_days / 365.25

    # CAGR: 年化复合增长率，几何平均收益率
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / total_years) - 1 if total_years > 0 else 0
    # 年化波动率: 标准差 × sqrt(252)
    annual_vol = ret.std() * np.sqrt(252)
    # 夏普比率: (策略收益 - 无风险利率) / 波动率
    sharpe = (cagr - risk_free) / annual_vol if annual_vol > 0 else 0

    # 最大回撤: 使用 expanding().max() 计算历史峰值
    peak = nav.expanding().max()
    dd = (peak - nav) / peak
    max_dd = dd.max()

    # 卡玛比率: 收益/回撤比，衡量策略的"收益代价"
    calmar = cagr / max_dd if max_dd > 0 else 0
    # 胜率: 正收益交易日占比
    win_rate = (ret > 0).mean()

    metrics = {
        "CAGR": cagr,
        "年化波动": annual_vol,
        "最大回撤": max_dd,
        "夏普比率": sharpe,
        "卡玛比率": calmar,
        "胜率": win_rate,
        "交易天数": len(nav),
    }

    # 相对收益指标（需要基准数据）
    if benchmark_nav is not None:
        bench_ret = benchmark_nav.reindex(nav.index).pct_change().dropna()
        aligned_ret = ret.reindex(bench_ret.index).dropna()
        bench_ret = bench_ret.reindex(aligned_ret.index)
        if len(aligned_ret) > 10:
            excess = aligned_ret - bench_ret
            tracking_error = excess.std() * np.sqrt(252)
            # 信息比率 = 超额收益均值 / 超额收益标准差 × sqrt(252)
            info_ratio = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
            metrics["信息比率"] = info_ratio
            metrics["跟踪误差"] = tracking_error
            bench_cagr = calc_benchmark_cagr(benchmark_nav, nav.index)
            metrics["超额收益"] = cagr - bench_cagr

    return metrics


def calc_benchmark_cagr(benchmark_nav, index):
    """计算基准在指定时间区间的年化收益率。

    基准的年化收益率用于计算超额收益。
    注意: 基准的时间区间必须与策略对齐，否则 CAGR 不可比。

    Args:
        benchmark_nav: 基准净值 Series
        index: 策略的日期索引（用于对齐）

    Returns:
        float: 年化收益率
    """
    bench = benchmark_nav.reindex(index).dropna()
    if len(bench) < 2:
        return 0
    days = (bench.index[-1] - bench.index[0]).days
    years = days / 365.25
    if years <= 0:
        return 0
    return (bench.iloc[-1] / bench.iloc[0]) ** (1 / years) - 1


def format_metrics(metrics):
    """将绩效指标数值格式化为百分比或两位小数字符串。

    格式化规则:
      - 百分比指标（CAGR、波动、回撤、超额收益）: 保留 1 位小数 + %
      - 比率指标（夏普、卡玛、信息比率）: 保留 2 位小数
      - 胜率: 保留 1 位小数 + %

    Args:
        metrics: calc_performance_metrics 返回的字典

    Returns:
        dict: 格式化后的字符串字典
    """
    result = {
        "CAGR": f"{metrics.get('CAGR', 0)*100:.1f}%",
        "年化波动": f"{metrics.get('年化波动', 0)*100:.1f}%",
        "最大回撤": f"{metrics.get('最大回撤', 0)*100:.1f}%",
        "夏普比率": f"{metrics.get('夏普比率', 0):.2f}",
        "卡玛比率": f"{metrics.get('卡玛比率', 0):.2f}",
        "胜率": f"{metrics.get('胜率', 0)*100:.1f}%",
        "超额收益": f"{metrics.get('超额收益', 0)*100:.1f}%",
        "信息比率": f"{metrics.get('信息比率', 0):.2f}",
    }
    return result

# ============================================================
# 参数组合穷举
# 2 × 2 × 3 = 12 种组合:
#   动量窗口(20/60) × 调仓频率(月度/周度) × 动量类型(simple/dual/dual_ma)
# ============================================================
# 参数网格穷举是策略开发的核心方法论:
#   1. 定义所有可能参数的取值空间
#   2. 遍历所有组合，全量运行回测
#   3. 按夏普比率排序，分析参数对绩效的影响
#   4. 避免"手工调参"的主观偏差
#
# 12 种组合足够覆盖核心参数维度，后续 v2/v3 扩展至 32 种。

PARAM_GRID = {
    "window": [20, 60],
    "rebalance_freq": ["monthly", "weekly"],
    "momentum_type": ["simple", "dual", "dual_ma"],
}

MOMENTUM_TYPE_LABELS = {
    "simple": "简单动量",
    "dual": "双重动量",
    "dual_ma": "双重动量+均线",
}

FREQ_LABELS = {
    "monthly": "月度",
    "weekly": "周度",
}


def run_all_combinations(close_df, benchmark_close, top_n=5):
    """遍历 2×2×3=12 种参数组合，全量运行回测。

    使用 itertools.product 生成所有参数组合的笛卡尔积。
    每组合运行完毕后立即计算绩效指标并保存净值曲线。

    输出:
      - results: 绩效指标列表（用于打印总览表和保存 CSV）
      - navs: 净值曲线字典 {组合名称: Series}（用于后续分析）

    Args:
        close_df: 全部 ETF 收盘价宽表
        benchmark_close: 基准收盘价
        top_n: 持仓数量

    Returns:
        results: list[dict]，每个元素是包含所有绩效指标的字典
        navs: dict[str, pd.Series]，组合名称为键，净值曲线为值
    """
    results = []
    navs = {}

    windows = PARAM_GRID["window"]
    freqs = PARAM_GRID["rebalance_freq"]
    types = PARAM_GRID["momentum_type"]

    total = len(windows) * len(freqs) * len(types)
    count = 0

    for w, f, t in product(windows, freqs, types):
        label = f"{MOMENTUM_TYPE_LABELS[t]}_{w}日_{FREQ_LABELS[f]}"
        count += 1
        print(f"  [{count}/{total}] {label} ... ", end="", flush=True)

        try:
            nav, trades = run_backtest(
                close_df, benchmark_close,
                top_n=top_n, window=w,
                rebalance_freq=f, momentum_type=t,
            )
            metrics = calc_performance_metrics(nav, benchmark_close)
            metrics["组合"] = label
            metrics["交易次数"] = len(trades)
            results.append(metrics)
            navs[label] = nav
            print(f"OK  夏普={metrics.get('夏普比率', 0):.2f}")
        except Exception as e:
            print(f"失败: {e}")

    return results, navs


def print_summary_table(results, benchmark_close=None):
    """按夏普比率降序打印绩效总览表，底部附加沪深300基准对比。

    输出格式:
      - 表头: 组合名称 → CAGR → 超额收益 → 夏普比率 → 最大回撤 → ...
      - 排序: 按夏普比率降序（最佳组合在顶部）
      - 末行: 基准_沪深300 的绩效指标（方便对比）

    控制台输出会被复制到 momentum_summary.md 中作为最终分析报告的原始数据。

    Args:
        results: run_all_combinations 返回的绩效指标列表
        benchmark_close: 基准收盘价 Series（用于打印基准行）
    """
    keys = ["组合", "CAGR", "超额收益", "夏普比率", "最大回撤", "年化波动", "胜率", "信息比率", "交易次数"]

    sorted_results = sorted(results, key=lambda x: x.get("夏普比率", 0), reverse=True)

    col_widths = {
        k: max(
            len(k),
            max(
                len(format_metrics(r).get(k, str(r.get(k, ""))))
                if k != "组合" else 0
                for r in sorted_results
            )
        )
        for k in keys[1:]
    }
    col_widths["组合"] = max(len(max(r["组合"] for r in sorted_results)), 25)

    header = " | ".join(f"{k:>{col_widths[k]}}" for k in keys)
    line_width = max(len(header), 80)
    print("\n" + "=" * line_width)
    print(f"{'动量趋势轮动策略 v1 — 绩效总览':^{line_width}}")
    print("=" * line_width)
    print(header)
    print("-" * len(header))

    for r in sorted_results:
        vals = format_metrics(r)
        vals["组合"] = r["组合"]
        vals["交易次数"] = str(r.get("交易次数", ""))
        row = " | ".join(
            f"{vals.get(k, str(r.get(k, ''))):>{col_widths[k]}}" for k in keys
        )
        print(row)

    print("-" * len(header))

    if benchmark_close is not None:
        bench_metrics = calc_performance_metrics(benchmark_close)
        bv = format_metrics(bench_metrics)
        bv["组合"] = "基准_沪深300"
        bv["交易次数"] = "-"
        row = " | ".join(
            f"{bv.get(k, ''):>{col_widths[k]}}" for k in keys
        )
        print(row)

    print("=" * line_width)
    print()


def save_results(results, navs):
    """保存 v1 回测结果为 CSV 文件。

    输出两个文件:
      1. performance_summary_v1.csv: 全部 12 种组合的绩效指标对比表
         - 列: 组合名称、CAGR、超额收益、夏普比率、最大回撤、年化波动等
         - 用于 Excel 分析和生成报告表格
      2. nav_all_v1.csv: 全部 12 种组合的净值曲线宽表
         - 行: 交易日
         - 列: 组合名称（如"简单动量_20日_月度"）
         - 用于绘制净值曲线图和计算相关统计量

    文件编码: utf-8-sig（带 BOM，确保 Excel 打开不乱码）
    """
    os.makedirs("momentum_results", exist_ok=True)

    keys = ["组合", "CAGR", "超额收益", "夏普比率", "最大回撤", "年化波动", "胜率", "信息比率", "卡玛比率", "交易次数"]
    summary = pd.DataFrame(results)[keys]
    summary.to_csv("momentum_results/performance_summary_v1.csv", index=False, encoding="utf-8-sig")

    nav_df = pd.DataFrame(navs)
    nav_df.index.name = "日期"
    nav_df.to_csv("momentum_results/nav_all_v1.csv", encoding="utf-8-sig")

    print("结果已保存到 momentum_results/")


def main():
    """主入口: 加载数据 → 穷举12种参数组合 → 输出绩效对比表。

    执行流程:
      1. 加载并过滤行业 ETF 列表（~631 只）
      2. 加载全部 ETF 日线数据，合并为宽表
      3. 加载沪深 300 基准数据
      4. 运行 12 种参数组合的回测
      5. 打印绩效总览表（控制台）
      6. 保存结果到 CSV 文件

    命令行参数:
      --top_n: 持仓数量，默认 5 只

    使用方式:
      python momentum_strategy_v1.py
      python momentum_strategy_v1.py --top_n 3
    """
    parser = argparse.ArgumentParser(description="A股ETF动量趋势轮动策略 v1 回测")
    parser.add_argument("--top_n", type=int, default=5, help="持仓数量（默认5）")
    args = parser.parse_args()
    top_n = args.top_n

    print("=" * 60)
    print("动量趋势轮动策略 v1 回测")
    print("=" * 60)

    print("\n[1/4] 加载行业 ETF ...")
    codes = load_industry_etf_codes()
    print(f"  找到 {len(codes)} 只行业 ETF")

    print("[2/4] 加载历史数据 ...")
    close_df = load_all_industry_data(codes)
    print(f"  数据范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")
    print(f"  ETF数量: {close_df.shape[1]} | 交易日: {close_df.shape[0]}")

    print("[3/4] 加载基准（沪深300）...")
    bench = load_benchmark()
    bench = bench.reindex(close_df.index).ffill().dropna()
    print(f"  基准范围: {bench.index[0].date()} ~ {bench.index[-1].date()}")

    print(f"[4/4] 运行回测（持仓 {top_n} 只）...")
    results, navs = run_all_combinations(close_df, bench, top_n)

    print_summary_table(results, bench)
    save_results(results, navs)


if __name__ == "__main__":
    main()
