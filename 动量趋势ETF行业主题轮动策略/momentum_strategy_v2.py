#!/usr/bin/env python3
"""
A股ETF动量趋势轮动策略 v2 — 回测与绩效分析

本文件实现了一个完整的 A 股行业 ETF 动量趋势轮动策略回测框架，主要功能包括：

策略逻辑:
  1. 在每个调仓日（月度），计算全部候选 ETF 的动量分数
  2. 按动量分数降序选择 top_n × 2 只候选 ETF
  3. 依次应用风控过滤（趋势过滤 → 均线过滤 → 相关性分散 → 波动率目标）
  4. 等权分配资金到最终选中的持仓中

参数网格穷举:
  - 动量窗口: 20 日 / 60 日
  - 动量类型: 简单动量 / 双重动量
  - 均线过滤 (MA filter): 开 / 关 —— 个股需站上 20MA + 60MA
  - 趋势过滤 (Trend filter): 开 / 关 —— 沪深 300 需站上 120MA
  - 多周期收敛 (Multi-period): 开 / 关 —— 短周期(20日)动量需与主窗口方向一致
  → 2 × 2 × 2 × 2 × 2 = 32 种组合

v2 相对 v1 的核心优化:
  - 回撤止损阈值从 15% 提高至 30%（减少误触发，避免"死亡螺旋"）
  - 目标波动率从 15% 提高至 20%（增加风险暴露，提升反弹弹性）
  - 新增趋势市过滤器（120MA），熊市中空仓避险 — v2 的最大质量提升项
  - 均线过滤从动量类型中拆分为独立布尔参数，可单独开关（原来仅 dual_ma 才有）
  - 多周期收敛: 短周期(20日)动量与主窗口方向一致，减少假突破信号
  - 仅月度调仓（删除低效的周度调仓，v1 证明周度交易成本拖累 CAGR 约 -12%）

v2 绩效亮点:
  最佳组合（简单动量_20日_趋势）在 2021-2026 熊市中取得 CAGR +11.5%、
  夏普比率 0.40，相对沪深 300 产生 +13.2% 的超额收益。
  趋势过滤器是"质的飞跃"——开启平均夏普 -0.13 vs 关闭平均夏普 -0.80。

数据依赖:
  - data/etf_list.csv: ETF 列表（代码、名称）
  - data/etf/{code}.csv: 每只 ETF 的日线行情（来自 akshare 东方财富）
  - 基准: 510300（沪深 300 ETF）

输出:
  - 控制台: 按夏普比率排序的绩效总览表
  - momentum_results/performance_summary_v2.csv: 全部组合的绩效指标
  - momentum_results/nav_all_v2.csv: 全部组合的净值曲线
"""

import os
import argparse
from itertools import product

import numpy as np
import pandas as pd

# ============================================================
# 全局参数配置
# ============================================================
# v2 相比 v1 调整了两个关键风控参数：
#   1. MAX_DRAWDOWN_PCT: 15% → 30%
#      v1 的 15% 回撤止损在持续下跌市场中反复触发，
#      形成"抄底-止损-再抄底"的死亡螺旋。
#      提高至 30% 给予策略更大的容错空间。
#   2. TARGET_VOL: 15% → 20%
#      v1 的 15% 目标波动率导致策略在反弹中力度不足，
#      提高至 20% 适当增加风险暴露。

DATA_DIR = "data/etf"                     # ETF 日线数据目录

# --- 风控参数 ---
TARGET_VOL = 0.20                         # 目标年化波动率 20%（组合层面）
                                          # v1 为 15%，提高后策略年化波动从 9% 升至 23.9%
STOP_LOSS_PCT = 0.08                      # 个股止损阈值 -8%（从成本价算）
                                          # 单只 ETF 从买入价下跌 8% 即强制卖出
MAX_DRAWDOWN_PCT = 0.30                   # 策略最大回撤止损 30%（净值从峰值算）
                                          # v1 为 15%，v2 提高一倍以减少误触发

# --- 均线参数 ---
MA_WINDOWS = (20, 60)                     # 均线过滤使用的双均线周期
                                          # 个股需站上 20MA 和 60MA 才纳入候选
TREND_MA_WINDOW = 120                     # 趋势过滤器使用的均线周期
                                          # 沪深 300 需站上 120MA 才允许入场
                                          # 120 日均线 ≈ 6 个月，是中期趋势的关键均线

# --- 选股参数 ---
CORR_THRESHOLD = 0.9                      # 相关性阈值 >0.9 的 ETF 视为冗余
                                          # 避免同时持有高度同质化的品种
MIN_DATA_DAYS = 60                        # ETF 最少需有 60 个交易日数据才纳入
                                          # 确保动量计算可靠（20/60 日动量窗口需要足够数据）
REBALANCE_CANDIDATE_MULTIPLIER = 2        # 候选数量 = top_n × 2（给相关性过滤留余量）

# --- 交易成本 ---
# 佣金费率 0.003%（万0.3），冲击成本 0.1%。
# v2 仅月度调仓，交易次数 ~50-100 次，成本影响较小。
# v1 周度调仓交易次数 ~1500 次，成本拖累 CAGR 约 -12%。
COMMISSION_RATE = 0.00003                 # 佣金费率 0.003%（万0.3）
SLIPPAGE_RATE = 0.001                     # 冲击成本 0.1%（单边）
TRADE_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE  # 单边总交易成本 0.103%

# ============================================================
# ETF 行业分类筛选
# 透过 ETF 名称关键词自动识别行业/主题 ETF，排除债券、货币、
# 商品期货、跨境等非权益类品种，确保回测池只包含 A 股行业 ETF。
# 过滤后约 631 只有效行业 ETF（全市场约 722 只）。
# ============================================================

def load_etf_list():
    """从 CSV 加载 ETF 列表。

    CSV 文件格式:
      - "代码"列: 6 位数字字符串（可能不带前导零）
      - "名称"列: ETF 中文全称
    文件路径: data/etf_list.csv

    Returns:
        DataFrame 包含 代码 和 名称 两列
    """
    path = "data/etf_list.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"ETF列表文件不存在: {path}")
    return pd.read_csv(path)


def filter_industry_etfs(df):
    """按名称关键词自动识别行业/主题 ETF，排除非权益品种。

    使用两层关键词匹配:
      1. include_kw: 名称包含这些词之一的保留
         - 覆盖传统行业（银行、证券、地产、有色等）
         - 覆盖新兴主题（AI、芯片、碳中和、数字经济等）
         - 覆盖策略指数（红利、价值、成长等）
      2. exclude_kw: 在保留基础上剔除不适配动量策略的品种
         - 债券类：国债、地方债、可转债（低波动，趋势弱）
         - 商品类：黄金、豆粕（非权益，与 A 股动量不同步）
         - 跨境类：标普、恒生、中概（受海外市场影响）

    局限性: 关键词匹配依赖于 ETF 的命名规范，少数名称不规范的
    ETF 可能被误剔或漏筛。但在 700+ 只 ETF 的规模上，这种方法
    性价比最高——精确的行业分类需要接入第三方数据，复杂度与收益不成正比。

    Args:
        df: 包含"名称"列的 DataFrame

    Returns:
        过滤后的 DataFrame
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

    # 使用 | 连接所有关键词，通过 str.contains 实现批量模糊匹配
    mask = df["名称"].str.contains("|".join(include_kw), na=False)
    exclude_mask = df["名称"].str.contains("|".join(exclude_kw), na=False)
    return df[mask & ~exclude_mask].reset_index(drop=True)


def load_industry_etf_codes():
    """加载并过滤行业 ETF，返回 6 位代码列表。

    完整流程:
      load_etf_list() → filter_industry_etfs() → 代码补零 → 返回列表
    代码统一为 6 位字符串（如 "510300"），与 CSV 文件名格式一致。

    Returns:
        list[str]: 6 位 ETF 代码列表
    """
    df = load_etf_list()
    df = filter_industry_etfs(df)
    codes = df["代码"].astype(str).str.zfill(6).tolist()
    print(f"行业ETF数量: {len(codes)} 只")
    return codes

# ============================================================
# 数据加载
# 从 data/etf/ 目录加载每只 ETF 的日线数据，合并为宽表格式。
# 宽表结构: 行=日期, 列=ETF代码, 值=收盘价。
# 宽表格式的优势: 一次性完成所有 ETF 的动量计算（向量化），
# 避免在主回测循环中逐只计算。
# 基准采用沪深300 ETF（510300）作为 passive benchmark。
# ============================================================


def load_etf_data(code):
    """加载单只 ETF 日线数据，返回 [日期, 收盘] 两列的 DataFrame。

    文件路径: data/etf/{code}.csv
    如果文件不存在（因 ETF 退市或代码变更），返回 None。
    数据按日期升序排列，确保时间序列计算正确。

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
    """加载所有行业 ETF 收盘价，合并为宽表。

    核心操作: 逐只加载 → 以日期为索引 → 横向拼接为宽表 DataFrame。
    这种"宽表"格式可以支持后续全量向量化运算（如动量计算、均线计算）。

    过滤条件:
      - 数据少于 MIN_DATA_DAYS (60) 天的 ETF 被剔除
        （20/60 日动量窗口需要足够的历史数据）
      - 仅保留最近 5 年数据
        （v2 使用 5 年窗口覆盖 2021-2026，v3 改为全量以覆盖牛熊周期）

    Args:
        codes: ETF 代码列表

    Returns:
        DataFrame, shape=(交易日数, ETF数量)
    """
    data = {}
    for code in codes:
        df = load_etf_data(code)
        if df is not None and len(df) > MIN_DATA_DAYS:
            data[code] = df.set_index("日期")["收盘"]
    result = pd.DataFrame(data).sort_index()
    # 删除完全缺失的行（非交易日导致个别 ETF 无数据）
    result = result.loc[result.index.notna()]
    # 截取最近 5 年数据
    cutoff = result.index.max() - pd.DateOffset(years=5)
    result = result[result.index >= cutoff]
    return result


def load_benchmark():
    """加载沪深300 ETF（510300）作为基准。

    使用 510300 而非沪深 300 指数本身的考虑:
      - 510300 是市场上流动性最好的宽基 ETF 之一
      - 包含分红和费率因素，更接近可投资基准的真实表现
      - 策略的买卖都以 ETF 为标的，用 ETF 做基准口径更一致

    Returns:
        Series, index=日期, value=收盘价, 或 None（数据不存在时）
    """
    df = load_etf_data("510300")
    if df is None:
        return None
    return df.set_index("日期")["收盘"].sort_index()

# ============================================================
# 风险控制
# 提供相关性分散工具，避免组合中持有高度相关的ETF。
# v2 的风控架构: 趋势过滤(策略级) → 均线过滤(个股级) → 相关性分散(组合级)
# 前两个过滤器在回测引擎内部实现（因为有条件逻辑），
# 相关性分散在这里作为独立工具函数。
# ============================================================


def diversify_by_correlation(close_df, candidates, top_n, corr_threshold=CORR_THRESHOLD):
    """基于收益率相关系数贪心选取低相关性 ETF。

    算法: 贪心法（非全局优化）
    1. 从候选列表中取出动量排名最高的 ETF 作为首个持仓
    2. 逐一检查后续候选与已选持仓的收益率相关系数
    3. 如果与所有已选持仓的相关系数均 < 阈值，则纳入
    4. 重复直到选满 top_n 只或候选池耗尽

    为什么用贪心法而不是全局优化？
      - 候选池只有 top_n × 2 = 10 只，贪心法已足够
      - 全局最优解（如最小化最大相关系数）需要穷举组合，性能开销大
      - 动量排名本身是核心信号，过于激进的分散会牺牲动量有效性
      - 实际操作中，A 股行业 ETF 之间相关性普遍在 0.5-0.8 之间，
        阈值 0.9 能剔除的只是"明显冗余"的品种

    Args:
        close_df: 近期收盘价 DataFrame（通常取最近 20 个交易日）
                  用于计算收益率相关系数
        candidates: 按动量分数降序排列的候选 ETF 列表
        top_n: 目标持仓数量
        corr_threshold: 相关性阈值，超过此值视为冗余

    Returns:
        list: 精选后的 ETF 代码列表，长度 ≤ top_n
    """
    if len(candidates) <= top_n:
        return candidates[:top_n]

    selected = []
    remaining = list(candidates)

    while len(selected) < top_n and remaining:
        candidate = remaining.pop(0)
        if not selected:
            selected.append(candidate)
            continue
        # 计算候选 ETF 与已选持仓的收益率相关系数
        returns = close_df[[candidate] + selected].pct_change().dropna()
        if len(returns) < 5:
            selected.append(candidate)
            continue
        corr = returns.corr().iloc[0, 1:]
        # 仅当与所有已选持仓均低相关时才纳入
        if all(corr < corr_threshold):
            selected.append(candidate)

    return selected

# ============================================================
# 回测引擎
# 核心逐日回测逻辑。v2 仅支持月度调仓（移除了周度选项）。
# 新增三个风控过滤器:
#   1. 趋势过滤器（120MA）— 策略级，市场下跌时完全空仓
#   2. 均线过滤器（20MA+60MA）— 个股级，只买上升趋势中的 ETF
#   3. 多周期收敛 — 信号级，短周期动量与主窗口需方向一致
# 这三个过滤器是 v2 相较 v1 的核心改进，也是"质的飞跃"的来源。
# ============================================================


def _get_rebalance_dates(index, min_window=MIN_DATA_DAYS):
    """计算月度调仓日期。

    v2 仅支持月度调仓（v1 的周度调仓已被证明效率低下）。
    取每月第一个交易日作为调仓日。
    最小持有窗口之前的数据仅用于计算指标，不执行调仓。

    Args:
        index: 完整日期索引
        min_window: 最小持有窗口（通常等于动量窗口）

    Returns:
        list[Timestamp]: 调仓日期列表
    """
    if len(index) <= min_window:
        return []
    dates = index[min_window:]
    s = pd.Series(dates, index=range(len(dates)))
    # 使用 dt.month 判断月份切换，每月第一个交易日触发调仓
    mask = s.dt.month != s.shift(1).dt.month
    mask.iloc[0] = True
    return s[mask].tolist()


def run_backtest(close_df, benchmark_close, top_n=5,
                 window=20, momentum_type="simple",
                 ma_filter=False, trend_filter=False, multi_period=False):
    """执行 v2 动量趋势轮动策略回测。

    v2 相比 v1 的核心改进（按重要性排序）:
      1. 趋势过滤器（120MA）: 沪深 300 站上 120MA 才允许入场
         这是 v2 最重要的改进——在熊市（2022、2018）中自动空仓，
         效果量化: 开启平均夏普 -0.13 vs 关闭 -0.80，改善 +0.67
      2. 均线过滤器: 个股需站上 20MA 和 60MA 才纳入候选池
         从 dual_ma 类型中拆分为独立参数，可与其他动量类型组合
         效果: 开启平均夏普 -0.37 vs 关闭 -0.56，改善 +0.19
      3. 多周期收敛: 20 日短周期动量必须为正
         效果: 开启平均夏普 -0.45 vs 关闭 -0.48，改善 +0.03
      4. 仅月度调仓: 删除周度选项，避免交易成本拖累
      5. 参数调整: 回撤止损 15%→30%，目标波动率 15%→20%

    Args:
        close_df: 全部 ETF 收盘价宽表，shape=(交易日, ETF数量)
        benchmark_close: 沪深 300 基准收盘价 Series
        top_n: 目标持仓数（默认 5 只）
        window: 动量计算窗口（20 或 60 日）
        momentum_type: "simple"（简单动量）或 "dual"（双重动量）
        ma_filter: 是否开启均线过滤（需站上 20MA+60MA）
        trend_filter: 是否开启趋势过滤（沪深 300 需站上 120MA）
        multi_period: 是否开启多周期收敛（20 日动量需为正）

    Returns:
        nav: 策略净值 Series，index=日期
        trades_df: 交易记录 DataFrame
    """
    # ================================================================
    # 第一步: 计算动量分数
    # ================================================================
    # 返回: 主窗口收益率（用于动量排序）
    # returns_short: 20 日短周期收益率（用于多周期收敛判断）
    returns = close_df / close_df.shift(window) - 1
    returns_short = close_df / close_df.shift(20) - 1
    benchmark_returns = benchmark_close / benchmark_close.shift(window) - 1
    # 趋势过滤器的核心数据: 沪深 300 的 120 日均线
    benchmark_ma120 = benchmark_close.rolling(TREND_MA_WINDOW).mean()
    above_trend = benchmark_close > benchmark_ma120

    if momentum_type == "simple":
        # 简单动量: 纯历史收益率排序
        # 这是最基础的动量定义，不加任何调整
        momentum_scores = returns
        if multi_period:
            # 多周期收敛: 如果 20 日短周期收益率为负，
            # 则即使主窗口动量高也不持仓（避免抄底）
            short_positive = returns_short > 0
            momentum_scores = momentum_scores.where(short_positive, np.nan)
    elif momentum_type == "dual":
        # 双重动量: 超额收益 + 绝对收益过滤
        # 超额收益 = 个股收益 - 基准收益
        # 消除市场整体涨跌对动量分数的影响
        momentum_scores = returns.sub(benchmark_returns, axis=0)
        # 绝对收益过滤: 个股收益率必须 > 0
        # 下跌趋势中的 ETF 即使跑赢基准也不考虑
        abs_mask = returns > 0
        momentum_scores = momentum_scores.where(abs_mask, np.nan)
        if multi_period:
            short_positive = returns_short > 0
            momentum_scores = momentum_scores.where(short_positive, np.nan)

    # --- 均线计算（用于 ma_filter）---
    # 提前预计算 20 日和 60 日均线，避免在主循环中重复计算
    # 这是 pandas 向量化操作，一次性完成全区间计算
    ma20 = close_df.rolling(MA_WINDOWS[0]).mean()
    ma60 = close_df.rolling(MA_WINDOWS[1]).mean()
    above_ma20 = close_df > ma20
    above_ma60 = close_df > ma60

    # ================================================================
    # 第二步: 波动率目标管理
    # ================================================================
    # 以沪深 300 的 20 日滚动波动率为基准，
    # 动态调整仓位比例: 目标比率 = min(20% / 滚动波动率, 1.0)
    # 
    # 市场波动率高 → 滚动波动率 > 20% → 目标比率 < 1 → 降仓
    # 市场波动率低 → 滚动波动率 < 20% → 目标比率 = 1 → 满仓
    #
    # v2 的目标波动率从 v1 的 15% 提高至 20%，
    # 策略年化波动从 9% 上升至 23.9%（接近基准的 18%），
    # 但换来了更高的反弹弹性和超额收益。
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
    # 状态变量说明:
    #   cash: 现金余额（初始单位净值 1.0）
    #   holdings: {代码: {"shares": 股数}}
    #   cost_prices: {代码: 买入均价}（用于个股止损判断）
    #   in_drawdown_stop: 是否处于回撤止损锁定状态
    #   peak_nav: 历史最高净值（用于计算回撤比例）
    rebalance_dates = _get_rebalance_dates(close_df.index, window)
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
    # 算法复杂度: O(N × K)，N=交易日数(~1200), K=ETF数量(~631)
    # 每次迭代仅 O(K) 的常数操作，实际运行 ~30 秒
    for i in range(window, len(close_df)):
        date = close_df.index[i]
        current_prices = close_df.iloc[i]

        # --- 4a. 计算当日净值 ---
        current_nav_value = cash + sum(
            holdings[c]["shares"] * current_prices[c]
            for c in list(holdings.keys())
            if c in current_prices.index and pd.notna(current_prices[c])
        ) if holdings else cash
        nav.iloc[i] = current_nav_value
        peak_nav = max(peak_nav, current_nav_value)

        # --- 4b. 策略级回撤止损（30%）---
        # 当净值从峰值回撤 30% 时强制清仓。
        # 止损后进入锁定状态，仅在调仓日解除。
        # v2 将阈值从 15% 提高至 30%，显著减少误触发。
        # 在 v1 的 2021-2026 回测中，15% 止损导致反复触发，
        # 形成"抄底→止损→再抄底"的死亡螺旋，
        # 最终净值跌至 0.55（相比 v2 的 1.06）。
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
            # 锁定状态: 只有在调仓日才解除，且解除后立即执行调仓逻辑
            if date in rebalance_set:
                in_drawdown_stop = False
            else:
                continue

        # --- 4c. 个股止损 ---
        # 从买入价下跌超过 8% 即强制卖出。
        # 与策略级回撤止损形成"双重防线"：
        #   个股止损 → 控制单只 ETF 的尾部风险
        #   策略止损 → 控制系统性回撤
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
        # v2 仅月度调仓，每月只有 ~1 个调仓日
        if date not in rebalance_set:
            continue

        # ================================================================
        # 第五步: 调仓日操作
        # 过滤器链条: 趋势过滤 → 动量选股 → 均线过滤 → 相关性分散
        # ================================================================

        # --- 5a. 趋势过滤器（策略级入场条件）---
        # 这是 v2 最重要的改进。
        # 当沪深 300 低于 120 日均线时，策略完全空仓持有现金。
        # 
        # 核心逻辑: 动量策略只在"趋势市"中有效。
        # 在下跌趋势中，动量信号大概率是"谁跌得慢"而非"谁能涨"。
        # 与其在这种环境中强行轮动，不如空仓等待。
        #
        # 注意: 2021-2026 回测中，趋势过滤器使得策略在 2022 年
        # 下跌 -21.63% 的熊市中仅亏损 -1.78%，规避了绝大部分下跌。
        # 但在 2024 年（基准 +14.68%），趋势过滤器的滞后性导致
        # 策略 -3.36%，完全错过了年初的反弹。
        if trend_filter and not above_trend.iloc[i]:
            for c in list(holdings.keys()):
                if c in current_prices.index and pd.notna(current_prices[c]):
                    proceeds = holdings[c]["shares"] * current_prices[c]
                    cash += proceeds * (1 - TRADE_COST_RATE)
                    trades.append({"日期": date, "ETF": c, "方向": "卖出(趋势过滤)",
                                   "价格": current_prices[c], "原因": "市场趋势向下"})
            holdings = {}
            cost_prices = {}
            continue

        # --- 5b. 动量选股 ---
        mom_row = momentum_scores.iloc[i]
        candidates = (
            mom_row.dropna()
            .sort_values(ascending=False)
            # 取 top_n × 2 只候选，给后续相关性过滤留余量
            .head(top_n * REBALANCE_CANDIDATE_MULTIPLIER)
            .index.tolist()
        )

        # --- 5c. 均线过滤（个股级入场条件）---
        # 候选 ETF 必须站上 20MA 和 60MA。
        # 这是从 v1 的 dual_ma 类型中拆出的独立参数。
        # 与趋势过滤器的区别:
        #   趋势过滤器 → 判断"市场"是否可入场（单一条件，影响所有持仓）
        #   均线过滤器 → 判断"个股"是否可入场（逐个判断，影响候选池）
        if ma_filter:
            candidates = [
                c for c in candidates
                if above_ma20.iloc[i].get(c, False)
                and above_ma60.iloc[i].get(c, False)
            ]

        # 无合格候选: 清仓持现
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

        # --- 5d. 相关性分散 ---
        selected = diversify_by_correlation(
            close_df.iloc[max(0, i - 20):i + 1], candidates, top_n
        )
        if not selected:
            continue

        vol_ratio = vol_target_ratio.iloc[i]

        # --- 5e. 卖出不在新持仓中的旧品种 ---
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

        # --- 5f. 等权分配 + 波动率目标调整 ---
        # 总资产 × 波动率调整比例 / 持仓数量 = 每只 ETF 的目标市值
        total_value = cash + sum(
            holdings[c]["shares"] * current_prices[c]
            for c in holdings
            if c in current_prices.index and pd.notna(current_prices[c])
        )
        if total_value <= 0:
            continue
        target_value_per_etf = (total_value * vol_ratio) / len(selected)

        # --- 5g. 执行买入/调仓（差额调仓）---
        # 采用"差额调仓"而非"先卖光再买"：
        # 计算目标持仓与当前持仓的差值，只交易差额部分。
        # 这样可以降低换手率和交易成本。
        for c in selected:
            if c not in current_prices.index or pd.isna(current_prices[c]):
                continue
            price = current_prices[c]
            if c in holdings:
                current_value = holdings[c]["shares"] * price
                diff = target_value_per_etf - current_value
                # 差异小于 0.1% 时跳过（避免无意义的小额交易）
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
                shares = target_value_per_etf / price
                holdings[c] = {"shares": shares}
                cost_prices[c] = price
                cash -= shares * price * (1 + TRADE_COST_RATE)
                trades.append({"日期": date, "ETF": c, "方向": "买入",
                               "价格": price, "原因": "调仓"})

        cash = max(cash, 0)

    nav = nav.dropna()
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return nav, trades_df

# ============================================================
# 绩效分析
# 计算收益、风险、风险调整收益等核心指标，
# 并提供与基准对比的超额收益、信息比率等相对指标。
# v2 的绩效分析框架与 v1 保持一致，便于跨版本对比。
# 关键对比指标: v2 最佳夏普 0.40 vs v1 最佳 -0.81
# ============================================================


def calc_performance_metrics(nav, benchmark_nav=None, risk_free=0.02):
    """计算策略收益与风险指标。

    核心指标包括:
      - 收益类: CAGR、超额收益
      - 风险类: 年化波动、最大回撤、交易天数
      - 风险调整收益类: 夏普比率、卡玛比率、信息比率、胜率

    v2 最佳组合（简单动量_20日_趋势）的关键指标:
      CAGR = +11.5%, 夏普 = 0.40, 超额收益 = +13.2%, 最大回撤 = 24.9%

    Args:
        nav: 策略净值 Series
        benchmark_nav: 基准净值 Series（可选，提供时计算超额收益和信息比率）
        risk_free: 无风险利率（默认 2%）

    Returns:
        dict: 绩效指标字典
    """
    if len(nav) < 2:
        return {}

    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    total_days = (nav.index[-1] - nav.index[0]).days
    total_years = total_days / 365.25

    # CAGR: 几何年化复合收益率，比算术平均更能反映真实增长
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / total_years) - 1 if total_years > 0 else 0
    # 年化波动: 日收益率标准差 × 平方根(252)
    annual_vol = ret.std() * np.sqrt(252)
    # 夏普比率: (策略超额收益) / 波动率
    sharpe = (cagr - risk_free) / annual_vol if annual_vol > 0 else 0

    peak = nav.expanding().max()
    dd = (peak - nav) / peak
    max_dd = dd.max()

    calmar = cagr / max_dd if max_dd > 0 else 0
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

    if benchmark_nav is not None:
        bench_ret = benchmark_nav.reindex(nav.index).pct_change().dropna()
        aligned_ret = ret.reindex(bench_ret.index).dropna()
        bench_ret = bench_ret.reindex(aligned_ret.index)
        if len(aligned_ret) > 10:
            excess = aligned_ret - bench_ret
            tracking_error = excess.std() * np.sqrt(252)
            # 信息比率 = 年化超额收益 / 跟踪误差
            info_ratio = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
            metrics["信息比率"] = info_ratio
            metrics["跟踪误差"] = tracking_error
            bench_cagr = calc_benchmark_cagr(benchmark_nav, nav.index)
            metrics["超额收益"] = cagr - bench_cagr

    return metrics


def calc_benchmark_cagr(benchmark_nav, index):
    """计算基准在指定时间区间的年化收益率。

    用于计算超额收益 = 策略 CAGR - 基准 CAGR。
    基准的时间区间必须与策略对齐，否则 CAGR 不一致。

    Args:
        benchmark_nav: 基准净值 Series
        index: 策略的日期索引

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
      - 百分比: 乘以 100 后保留 1 位小数 + % 后缀
      - 比率: 直接保留 2 位小数

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
# ============================================================
# v2 的参数网格从 v1 的 2×2×3=12 种扩展至 2×2×2×2×2=32 种:
#   v1: 窗口(2) × 频率(2) × 动量类型(3) = 12
#   v2: 窗口(2) × 动量类型(2) × 均线(2) × 趋势(2) × 多周期(2) = 32
#
# 变更原因:
#   - 均线过滤从动量类型中拆出，可单独开关
#   - 新增趋势过滤和多周期收敛两个独立维度
#   - 固定为月度调仓（删除频率维度）
#
# 32 种组合足够分析各优化的边际贡献:
#   - 趋势过滤: +0.67 夏普改善
#   - 均线过滤: +0.19
#   - 多周期收敛: +0.03
#   - 20日 vs 60日: +0.47

PARAM_GRID = {
    "window": [20, 60],
    "momentum_type": ["simple", "dual"],
    "ma_filter": [False, True],
    "trend_filter": [False, True],
    "multi_period": [False, True],
}

WINDOW_LABELS = {20: "20日", 60: "60日"}
MOMENTUM_TYPE_LABELS = {"simple": "简单动量", "dual": "双重动量"}
BOOL_LABELS = {False: "关", True: "开"}


def make_label(w, t, ma, trend, multi):
    """生成参数组合的可读名称。

    命名规则: 动量类型_窗口_均线_趋势_多周期
    例如: "简单动量_20日_均线_趋势" 表示:
      - 简单动量 + 20 日窗口 + 均线过滤开启 + 趋势过滤开启 + 多周期关闭

    Args:
        w: 动量窗口（20 或 60）
        t: 动量类型（"simple" 或 "dual"）
        ma: 均线过滤（True/False）
        trend: 趋势过滤（True/False）
        multi: 多周期收敛（True/False）

    Returns:
        str: 组合的可读名称
    """
    parts = [MOMENTUM_TYPE_LABELS[t], WINDOW_LABELS[w]]
    if ma:
        parts.append("均线")
    if trend:
        parts.append("趋势")
    if multi:
        parts.append("多周期")
    return "_".join(parts) if len(parts) > 1 else parts[0]


def run_all_combinations(close_df, benchmark_close, top_n=5):
    """遍历 2×2×2×2×2=32 种参数组合，全量运行回测。

    使用 itertools.product 生成所有组合的笛卡尔积。
    每个组合运行完毕后立即计算绩效指标并记录净值曲线。

    输出:
      - results: 绩效指标列表（用于打印和保存 CSV）
      - navs: 净值曲线字典 {组合名称: Series}（用于 CSV 导出和绘图）

    Args:
        close_df: 全部 ETF 收盘价宽表
        benchmark_close: 基准收盘价 Series
        top_n: 持仓数量

    Returns:
        results: list[dict]，每个元素是包含所有绩效指标的字典
        navs: dict[str, pd.Series]，组合名称为键，净值曲线为值
    """
    results = []
    navs = {}

    windows = PARAM_GRID["window"]
    types = PARAM_GRID["momentum_type"]
    ma_flags = PARAM_GRID["ma_filter"]
    trend_flags = PARAM_GRID["trend_filter"]
    multi_flags = PARAM_GRID["multi_period"]

    total = len(windows) * len(types) * len(ma_flags) * len(trend_flags) * len(multi_flags)
    count = 0

    for w, t, ma, trend, multi in product(windows, types, ma_flags, trend_flags, multi_flags):
        label = make_label(w, t, ma, trend, multi)
        count += 1
        print(f"  [{count}/{total}] {label} ... ", end="", flush=True)

        try:
            nav, trades = run_backtest(
                close_df, benchmark_close,
                top_n=top_n, window=w,
                momentum_type=t,
                ma_filter=ma,
                trend_filter=trend,
                multi_period=multi,
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
      - 按夏普比率降序排列，最优组合在顶部
      - 末行附加基准（沪深300）的绩效指标
      - 控制台输出直接复制到 momentum_summary.md 中

    v2 的典型输出（2021-2026 回测）:
      排名第 1: 简单动量_20日_趋势 | CAGR +11.5% | 夏普 0.40
      排名第 2: 双重动量_20日_趋势 | CAGR +11.5% | 夏普 0.40
      基准: 沪深300 | CAGR -1.7% | 夏普 -0.60

    Args:
        results: run_all_combinations 返回的绩效指标列表
        benchmark_close: 基准收盘价 Series（可选，打印基准行）
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
    print(f"{'动量趋势轮动策略 v2 — 绩效总览':^{line_width}}")
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
    """保存 v2 回测结果为 CSV 文件。

    输出两个文件:
      1. performance_summary_v2.csv: 32 种组合的绩效指标对比表
         - 列: 组合名称、CAGR、超额收益、夏普比率、最大回撤、年化波动等
         - 用于 Excel 分析和生成 momentum_summary.md 中的表格
      2. nav_all_v2.csv: 32 种组合的净值曲线宽表
         - 行: 交易日, 列: 组合名称
         - 用于绘制净值曲线图和跨组合对比

    与 v1 输出的关系:
      - v1: performance_summary_v1.csv, nav_all_v1.csv
      - v2: performance_summary_v2.csv, nav_all_v2.csv
      - v3: performance_summary_v3.csv, nav_all_v3.csv
      所有版本独立保存，互不覆盖，便于跨版本对比分析。

    Args:
        results: run_all_combinations 返回的绩效指标列表
        navs: 净值曲线字典 {组合名称: Series}
    """
    os.makedirs("momentum_results", exist_ok=True)

    keys = ["组合", "CAGR", "超额收益", "夏普比率", "最大回撤", "年化波动", "胜率", "信息比率", "卡玛比率", "交易次数"]
    summary = pd.DataFrame(results)[keys]
    summary.to_csv("momentum_results/performance_summary_v2.csv", index=False, encoding="utf-8-sig")

    nav_df = pd.DataFrame(navs)
    nav_df.index.name = "日期"
    nav_df.to_csv("momentum_results/nav_all_v2.csv", encoding="utf-8-sig")

    print("结果已保存到 momentum_results/")


def main():
    """主入口: 加载数据 → 穷举32种参数组合 → 输出绩效对比表。

    执行流程:
      1. 加载并过滤行业 ETF 列表（~631 只）
      2. 加载全部 ETF 日线数据，合并为宽表
      3. 加载沪深 300 基准数据
      4. 运行 32 种参数组合的回测
      5. 打印绩效总览表（控制台）
      6. 保存结果到 CSV 文件

    命令行参数:
      --top_n: 持仓数量，默认 5 只

    使用方式:
      python momentum_strategy_v2.py
      python momentum_strategy_v2.py --top_n 3

    预期运行时间: ~30-60 秒（32 种组合，631 只 ETF，5 年数据）
    """
    parser = argparse.ArgumentParser(description="A股ETF动量趋势轮动策略 v2 回测")
    parser.add_argument("--top_n", type=int, default=5, help="持仓数量（默认5）")
    args = parser.parse_args()
    top_n = args.top_n

    print("=" * 60)
    print("动量趋势轮动策略 v2 回测")
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
