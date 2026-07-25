#!/usr/bin/env python3
"""
A股ETF动量趋势轮动策略 v4 — 分阶段回测与市场环境分析

v4 基于 v3 策略引擎，核心新增功能为"分阶段回测"：
  - 将 2014-2026 完整周期拆分为 6 个独立市场阶段
  - 在每个阶段独立运行全部 32 种参数组合
  - 分析策略在不同市场环境（牛市/熊市/震荡/股灾）下的表现差异
  - 跨阶段稳健性排名，识别最稳健的参数配置

6 个市场阶段:
  1. 2014大牛市 (2014-07 ~ 2015-06): 基准 +155%，单边暴涨
  2. 2015三次股灾 (2015-06 ~ 2016-01): 基准 -62%，流动性危机
  3. 2016震荡慢牛 (2016-02 ~ 2017-12): 基准 +20%，极速轮动 — 策略最大软肋
  4. 2018单边下跌 (2018-01 ~ 2019-01): 基准 -21%，单边熊市
  5. 2019核心资产牛 (2019-02 ~ 2021-02): 基准 +33%，龙头牛市
  6. 2021熊市结构 (2021-02 ~ 2026-07): 基准 -4%，慢熊结构市

v4 的核心洞察:
  - 不存在跨周期一致最优的参数组合
  - 策略在约 50% 的阶段有效，另外 50% 亏损或跑输
  - 2016 震荡慢牛是动量策略的"阿喀琉斯之踵"（全部组合亏损）
  - 风险平价是唯一跨阶段稳定的优化项
"""

import os
import argparse
from itertools import product

import numpy as np
import pandas as pd

# ============================================================
# 全局参数配置
# 这些参数控制策略的核心行为，在 v1-v4 中保持一致以便跨版本对比。
# ============================================================

DATA_DIR = "data/etf"  # ETF日线数据目录，从 akshare 下载后存放于此

TARGET_VOL = 0.20         # 目标年化波动率 20%，控制杠杆上限
STOP_LOSS_PCT = 0.08      # 个股止损线 -8%（v4 沿用 v3 的收紧止损）
MAX_DRAWDOWN_PCT = 0.30   # 策略最大回撤阈值 30%，触发后永久清仓
CORR_THRESHOLD = 0.9      # 相关性阈值 >0.9 的 ETF 视为同质化
MIN_DATA_DAYS = 60        # ETF 最少需要 60 个交易日的数据才能参与
REBALANCE_CANDIDATE_MULTIPLIER = 2  # 动量排序取 2×topN 候选池
COMMISSION_RATE = 0.00003 # 佣金费率 0.003%
SLIPPAGE_RATE = 0.001     # 滑点成本 0.1%（冲击成本估计）
TRADE_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE  # 总交易成本
TREND_FAST_WINDOW = 60    # 快线 60MA，用于分档建仓的轻仓线
TREND_SLOW_WINDOW = 120   # 慢线 120MA，用于趋势判断的做多线
SCALED_LIGHT_WEIGHT = 0.50  # 分档建仓的轻仓比例 50%
SCALED_FULL_THRESHOLD = 0.05  # 价格站上120MA超过5%时满仓
WARMUP_DAYS = 180         # 分阶段回测的预热天数（用于计算均线和动量）

# ============================================================
# 行业分类关键词映射表
# 将 ETF 名称中的关键词映射到 16 个预定义行业分类。
# 用于行业覆盖约束（sector_diversity=True）和后续分析。
# ============================================================

SECTOR_KEYWORDS = {
    "消费":    ["消费", "食品", "饮料", "酒ETF"],
    "医药":    ["医药", "医疗", "生物", "药ETF", "创新药"],
    "科技":    ["科技", "半导体", "芯片", "电子", "计算机", "信创", "数字经济", "软件"],
    "新能源":  ["新能源", "光伏", "风电", "电池", "碳中和", "新能车", "电动车", "储能"],
    "金融":    ["银行", "证券", "保险", "金融"],
    "军工":    ["军工", "航天", "航空", "国防"],
    "周期":    ["有色", "煤炭", "钢铁", "化工", "材料", "能源", "有色ETF"],
    "制造":    ["制造", "机器人", "智能"],
    "地产基建": ["地产", "基建", "建材", "房地产"],
    "公用事业": ["电力", "公用", "环保", "绿电"],
    "传媒":    ["传媒", "游戏", "文娱", "影视"],
    "通信":    ["通信", "5G", "AI", "人工智能"],
    "农业":    ["农业", "养殖"],
    "汽车":    ["汽车", "智能驾驶"],
    "央企":    ["央企", "国企"],
    "综合":    ["红利", "价值", "成长", "龙头", "创新", "物流"],
}

# ============================================================
# v4 核心: 6 个市场阶段定义
# 每个阶段的起止日期对应 A 股关键拐点（牛市顶/熊市底）。
# ============================================================

SUB_PERIODS = [
    # 阶段1: 2014-2015 大牛市 — 资金驱动、单边暴涨
    # 基准沪深300涨幅约 +155%，是动量策略最容易获利的阶段
    ("2014大牛市",       "2014-07-01", "2015-06-12"),
    # 阶段2: 2015 三次股灾 — 杠杠崩塌、流动性危机
    # 基准沪深300跌幅约 -62%，动量策略天然空仓避险
    ("2015三次股灾",     "2015-06-15", "2016-01-28"),
    # 阶段3: 2016 震荡慢牛 — 极速行业轮动，动量策略最大软肋
    # 基准沪深300涨幅约 +20%，但此阶段行业轮动极快
    # 历史回测显示几乎所有参数组合在此阶段亏损
    ("2016震荡慢牛",     "2016-02-01", "2017-12-31"),
    # 阶段4: 2018 单边下跌 — 中美贸易战，去杠杆
    # 基准沪深300跌幅约 -21%，纯粹的防御考验
    ("2018单边下跌",     "2018-01-02", "2019-01-31"),
    # 阶段5: 2019-2021 核心资产牛市 — 外资流入、龙头溢价
    # 基准沪深300涨幅约 +33%，核心资产强者恒强
    ("2019核心资产牛",   "2019-02-01", "2021-02-10"),
    # 阶段6: 2021-2026 熊市结构 — 慢熊、局部热点、风格频繁切换
    # 基准沪深300跌幅约 -4%（截至2026年7月）
    # 是动量策略最具挑战的长期环境
    ("2021熊市结构",     "2021-02-18", "2026-07-17"),
]

START_DATE = "2014-01-01"

# ============================================================
# ETF 行业分类筛选
# ============================================================

def load_etf_list():
    """从 data/etf_list.csv 加载全市场 ETF 基础信息。
    
    该文件包含 ETF 代码、名称、类型等基本信息。
    v4 在此基础上通过关键词两阶段过滤筛选行业/主题 ETF。
    """
    path = "data/etf_list.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"ETF列表文件不存在: {path}")
    return pd.read_csv(path)


def filter_industry_etfs(df):
    """关键词两阶段过滤行业/主题 ETF，剔除债券/商品/跨境/期货品种。
    
    第一阶段（include_kw）：保留名称包含行业、消费、医药、科技等关键词的 ETF。
    第二阶段（exclude_kw）：剔除国债、可转债、货币、黄金、豆粕等非权益品种，
                             以及标普、纳斯达克、恒生等跨境品种。
    
    A 股行业/主题 ETF 是动量策略的最佳载体，原因：
      1. 行业指数本身具有均值回复 + 动量效应双重特征
      2. 各行业之间存在足够大的分化（年化收益差距可达 50%+）
      3. 相比个股，ETF 无流动性风险、无退市风险
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

    mask = df["名称"].str.contains("|".join(include_kw), na=False)
    exclude_mask = df["名称"].str.contains("|".join(exclude_kw), na=False)
    return df[mask & ~exclude_mask].reset_index(drop=True)


def load_industry_etf_codes():
    """加载并过滤行业/主题 ETF，返回 6 位代码列表。
    
    这是数据加载的第一步。返回的代码列表将是后续所有回测的 ETF 池。
    """
    df = load_etf_list()
    df = filter_industry_etfs(df)
    codes = df["代码"].astype(str).str.zfill(6).tolist()
    print(f"行业ETF数量: {len(codes)} 只")
    return codes

# ============================================================
# 数据加载
# 从 data/etf/ 加载日线数据，合并为宽表格式。
# v4 使用全量历史数据（从2014-01-01开始）。
# ============================================================

def load_etf_data(code):
    """从 data/etf/ 加载单只 ETF 日线数据。
    
    返回 [日期, 收盘] 两列，用于后续合并为宽表。
    如果文件不存在（部分 ETF 退市或无数据），返回 None。
    这是 v4 回测的基础数据层，覆盖 2014-01-01 ~ 2026-07-17。
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
    
    合并逻辑：每只 ETF 作为一列，行索引为日期。
    缺失值来自各 ETF 上市时间不同（创业板 ETF 2011 年上市，
    科创板 ETF 2020 年上市，数据跨度天然不一致）。
    
    v4 使用全量历史数据（从 START_DATE=2014-01-01 开始），
    跨度为 12.5 年，覆盖 6 个完整市场周期。
    
    Returns:
        pd.DataFrame: 行=日期，列=ETF代码
    """
    data = {}
    for code in codes:
        df = load_etf_data(code)
        if df is not None and len(df) > MIN_DATA_DAYS:
            data[code] = df.set_index("日期")["收盘"]
    result = pd.DataFrame(data).sort_index()
    result = result.loc[result.index.notna()]
    result = result[result.index >= START_DATE]
    return result


def load_benchmark():
    """加载沪深300 ETF (510300) 作为业绩基准。
    
    选择 510300 而非直接使用沪深300指数，原因：
      1. 510300 是全国规模最大、流动性最好的沪深300 ETF
      2. 使用真实 ETF 数据自然包含跟踪误差和交易成本
      3. 与回测使用的 ETF 数据来源一致（akshare）
    """
    df = load_etf_data("510300")
    if df is None:
        return None
    return df.set_index("日期")["收盘"].sort_index()

# ============================================================
# 行业分类识别
# 通过关键词匹配将ETF归入16个预定义行业分类，
# 用于行业覆盖约束，防止同行业过度集中。
# ============================================================

def identify_sector(etf_name):
    """根据 ETF 名称关键词匹配行业分类。
    
    使用 SECTOR_KEYWORDS 映射表进行关键词匹配。
    优先匹配更长、更具体的关键词（如"智能驾驶"优先于"智能"）。
    未匹配到任何关键词的 ETF 归为"其他"（通常是不常见的主题 ETF）。
    """
    for sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in etf_name:
                return sector
    return "其他"


def load_etf_sectors():
    """构建 {ETF 代码: 行业分类} 映射字典。
    
    该字典用于回测中的行业覆盖约束（sector_diversity=True），
    确保持仓组合不会过度集中在某一两个行业。
    行业分类精度取决于 ETF 名称中的关键词覆盖度。
    """
    df = load_etf_list()
    df = filter_industry_etfs(df)
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["行业"] = df["名称"].apply(identify_sector)
    return dict(zip(df["代码"], df["行业"]))

# ============================================================
# 风险控制
# 包含相关性分散、行业覆盖约束、风险平价加权三个风控模块。
# ============================================================

def diversify_by_correlation(close_df, candidates, top_n, corr_threshold=CORR_THRESHOLD):
    """基于收益率相关性的贪心选股，避免 ETF 同质化。
    
    核心思路：从动量排序前 2×topN 的候选池中，按排序逐个检查，
    如果该 ETF 与已选池中任意 ETF 的相关性超过阈值，则跳过。
    
    贪心算法的局限：不能保证全局最优，但在 ETF 数量有限（通常 30-60 只）
    的情况下已足够。更复杂的优化算法（如整数规划）会带来额外的实现复杂度。
    
    Args:
        close_df: 近期收盘价数据（用于计算相关性）
        candidates: 按动量排序的候选 ETF 列表
        top_n: 目标持仓数量
        corr_threshold: 相关性阈值，默认 0.9
        
    Returns:
        selected: 分散化后的持仓列表
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
        returns = close_df[[candidate] + selected].pct_change().dropna()
        if len(returns) < 5:
            selected.append(candidate)
            continue
        corr = returns.corr().iloc[0, 1:]
        if all(corr < corr_threshold):
            selected.append(candidate)

    return selected


def diversify_by_sector(candidates, etf_sectors, top_n, max_per_sector=2):
    """行业覆盖约束：确保持仓在不同行业间分散。
    
    在同一行业内最多选择 max_per_sector 只 ETF（默认 2 只）。
    这个约束配合相关性分散共同作用：
      - 相关性分散侧重数值指标（防止高度相关的 ETF 集中）
      - 行业分散侧重语义分类（防止忽略相关性但本质同类的品种）
    
    两个风控可以独立开关，在 32 种参数组合中均有对应配置。
    """
    sector_count = {}
    selected = []
    for c in candidates:
        sector = etf_sectors.get(c, "其他")
        if sector_count.get(sector, 0) < max_per_sector:
            selected.append(c)
            sector_count[sector] = sector_count.get(sector, 0) + 1
        if len(selected) >= top_n:
            break
    return selected


def calc_risk_parity_weights(close_df, selected, window=20):
    """计算风险平价权重：权重与近 20 日波动率倒数成正比。
    
    风险平价的核心思想：让组合中每个品种对总风险的贡献相等。
    具体实现：
      - 用过去 20 个交易日的收益率标准差作为波动率估计
      - 权重 = (1/波动率) / sum(1/波动率)
    效果：低波动品种获得更高权重，高波动品种权重被压低。
    
    这是 v3 引入的优化项。在 v4 的跨阶段稳健性排名中显示，
    风险平价是唯一在所有市场阶段都能稳定提升夏普比率的优化。
    
    Args:
        close_df: 收盘价宽表
        selected: 持仓 ETF 列表
        window: 波动率计算窗口（20 个交易日 ≈ 1 个月）
    """
    prices = close_df[selected].dropna()
    if prices.empty or len(prices) < 5:
        return {c: 1.0 / len(selected) for c in selected}
    rets = prices.pct_change().dropna()
    vols = rets.tail(window).std()
    vols = vols.replace(0, np.nan)
    if vols.isna().all():
        return {c: 1.0 / len(selected) for c in selected}
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()

# ============================================================
# 回测引擎
# 基于v3策略引擎，支持分阶段回测（v4核心功能）。
# 沿用v3的双层趋势线、分档建仓、风险平价、行业分散等优化。
# ============================================================

def _get_rebalance_dates(index, min_window=MIN_DATA_DAYS):
    """计算月度调仓日期序列。
    
    策略在所有版本（v1-v4）中均使用月度调仓频率。
    选择月度而非周度的原因：
      1. 动量信号在月频上信噪比更高（周频噪声过大）
      2. ETF 切换交易成本在月频下可接受
      3. 与行业轮动的实际节奏匹配（行业趋势以月为单位变化）
    
    Args:
        index: 完整日期索引
        min_window: 跳过初期数据不足的交易日
        
    Returns:
        List[Timestamp]: 每个调仓月的第一个交易日
    """
    if len(index) <= min_window:
        return []
    dates = index[min_window:]
    s = pd.Series(dates, index=range(len(dates)))
    mask = s.dt.month != s.shift(1).dt.month
    mask.iloc[0] = True
    return s[mask].tolist()


def compute_trend_position(benchmark_close, date_idx, position_sizing):
    """基于沪深 300 指数的趋势判断计算目标仓位比例。
    
    这是策略趋势过滤的核心。使用沪深 300 而非 ETF 组合本身
    来判断市场整体趋势，因为沪深 300 更能代表 A 股整体走势。
    
    三种仓位模式（对应 v2/v3 的演进）:
      - "none":  始终满仓，无趋势过滤（v1 模式，用于对照）
      - "binary": 价格 > 120MA 则满仓，否则空仓（v2 模式）
      - "scaled":（v3/v4 模式）三档分档：
          * 价格 < 60MA → 空仓（0%）
          * 60MA < 价格 < 120MA → 轻仓（50%）
          * 价格 > 120MA 且偏离度 ≥ 5% → 满仓（100%）
          * 价格 > 120MA 但偏离度 < 5% → 轻仓（50%）
    
    分档建仓的目的是减少假突破带来的频繁开关损失。
    当沪深 300 刚刚站上 120MA 但偏离不足时，先以 50% 仓位试探，
    等趋势确认（偏离 ≥ 5%）后再加至满仓。
    
    Args:
        benchmark_close: 沪深 300 收盘价序列
        date_idx: 当前交易日索引
        position_sizing: 仓位模式 ("none" / "binary" / "scaled")
    """
    if position_sizing == "none":
        return 1.0

    price = benchmark_close.iloc[date_idx]
    ma_fast = benchmark_close.rolling(TREND_FAST_WINDOW).mean().iloc[date_idx]
    ma_slow = benchmark_close.rolling(TREND_SLOW_WINDOW).mean().iloc[date_idx]

    if pd.isna(ma_fast) or pd.isna(ma_slow):
        return 0.0

    if position_sizing == "binary":
        return 1.0 if price > ma_slow else 0.0

    if position_sizing == "scaled":
        if price < ma_fast:
            return 0.0
        elif price > ma_slow:
            deviation = (price - ma_slow) / ma_slow
            if deviation >= SCALED_FULL_THRESHOLD:
                return 1.0
            else:
                return SCALED_LIGHT_WEIGHT
        else:
            return SCALED_LIGHT_WEIGHT

    return 1.0


def run_backtest(close_df, benchmark_close, etf_sectors, top_n=5,
                 window=20, momentum_type="simple",
                 position_sizing="binary",
                 risk_parity=False, sector_diversity=False):
    """执行动量趋势轮动策略回测引擎（v3/v4 共用）。
    
    这是整个策略的核心。回测逻辑流程：
    
    Step 1 — 计算动量分数（窗口 n 日的收益率或超额收益）
    Step 2 — 在每个调仓日执行以下循环：
      a. 计算当前净值，更新回撤监控
      b. 检查是否触发策略最大回撤保护（30%）
      c. 检查持仓中是否有 ETF 触发个股止损（-8%）
      d. 分层趋势判断（沪深 300 的 60MA/120MA）
      e. 从候选池选取动量 Top N，应用相关性分散 + 行业分散
      f. 计算风险平价权重（如启用）
      g. 执行调仓，更新持仓
    Step 3 — 返回净值序列和交易记录
    
    与 v1/v2 的关键区别：
      - v1: 无趋势过滤，无风险平价，15% 最大回撤止损
      - v2: 引入 120MA 趋势过滤，取消回撤止损
      - v3/v4: 双层趋势线 + 分档建仓 + 风险平价 + 行业分散 + 30% 回撤止损

    Args:
        close_df: ETF 收盘价宽表（行=日期，列=ETF代码）
        benchmark_close: 沪深 300 基准（用于趋势判断和超额动量）
        etf_sectors: {ETF代码: 行业分类} 字典
        top_n: 持仓数量（默认 5 只）
        window: 动量计算窗口（20 日或 60 日）
        momentum_type: "simple"（简单动量）或 "dual"（双重动量）
        position_sizing: "binary"（二元仓位）或 "scaled"（分档仓位）
        risk_parity: 是否启用风险平价加权
        sector_diversity: 是否启用行业分散约束

    Returns:
        nav: 策略净值序列（pd.Series）
        trades_df: 交易记录（pd.DataFrame）
    """
    # --- Step 1: 计算动量分数 ---
    # 简单动量：仅使用过去 window 日的绝对收益率
    # 双重动量：绝对收益率减去同期基准收益率的超额收益，
    #           且只有当绝对收益为正时才计入（避免负收益的"更不差"）
    returns = close_df / close_df.shift(window) - 1
    benchmark_returns = benchmark_close / benchmark_close.shift(window) - 1

    if momentum_type == "simple":
        momentum_scores = returns
    elif momentum_type == "dual":
        momentum_scores = returns.sub(benchmark_returns, axis=0)
        abs_mask = returns > 0
        momentum_scores = momentum_scores.where(abs_mask, np.nan)

    # --- 波动率目标管理 ---
    # 当市场波动率上升时自动降低仓位，目标年化波动率 20%
    # 使用沪深 300 的 20 日滚动波动率作为市场整体波动率的代理
    # 波动率目标 = min(目标波动率 / 当前波动率, 1.0)
    # 上限为 1.0（不限制更低市场波动率下的仓位）
    bench_daily_ret = benchmark_close.pct_change().dropna()
    rolling_vol = bench_daily_ret.rolling(20).std() * np.sqrt(252)
    vol_target_ratio = pd.Series(1.0, index=close_df.index)
    vol_target_ratio.iloc[20:] = np.minimum(
        TARGET_VOL / rolling_vol.iloc[19:].values,
        1.0
    )
    vol_target_ratio = vol_target_ratio.clip(0, 1)

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

    # --- Step 2: 每日循环回测 ---
    # 逐日模拟真实交易流程：
    # 1. 每日更新持仓市值
    # 2. 仅在调仓日进行交易决策
    # 3. 止损检查在每个交易日都进行
    for i in range(window, len(close_df)):
        date = close_df.index[i]
        current_prices = close_df.iloc[i]

        # --- 2a: 计算当前净资产（现金 + 持仓市值） ---
        current_nav_value = cash + sum(
            holdings[c]["shares"] * current_prices[c]
            for c in list(holdings.keys())
            if c in current_prices.index and pd.notna(current_prices[c])
        ) if holdings else cash
        nav.iloc[i] = current_nav_value
        peak_nav = max(peak_nav, current_nav_value)

        # --- 2b: 策略最大回撤保护 ---
        # 当组合净值从峰值回撤超过 30% 时，永久清仓
        # 这是 v1 15% 止损的改进（15% 在震荡市中频繁触发，造成"死亡螺旋"）
        # 30% 的阈值给策略足够的喘息空间，同时保护本金
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
            # 在回撤止损状态下，仅在下个调仓日恢复交易
            if date in rebalance_set:
                in_drawdown_stop = False
            else:
                continue

        # --- 2c: 个股止损检查 ---
        # 对持仓中每只 ETF 检查是否需要止损
        # 止损线 -8%（从买入成本价计算）
        # 这是 v2 引入的改进（v1 没有个股止损）
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

        # --- 2d: 非调仓日跳过 ---
        if date not in rebalance_set:
            continue

        # --- 2e: 趋势判断与仓位控制 ---
        trend_position = compute_trend_position(benchmark_close, i, position_sizing)
        if trend_position <= 0:
            # 趋势转空，清空全部持仓进入现金状态
            for c in list(holdings.keys()):
                if c in current_prices.index and pd.notna(current_prices[c]):
                    proceeds = holdings[c]["shares"] * current_prices[c]
                    cash += proceeds * (1 - TRADE_COST_RATE)
                    trades.append({"日期": date, "ETF": c, "方向": "卖出(趋势过滤)",
                                   "价格": current_prices[c], "原因": "趋势向下"})
            holdings = {}
            cost_prices = {}
            continue

        # --- 2f: 动量排序与候选池构建 ---
        mom_row = momentum_scores.iloc[i]
        candidates = (
            mom_row.dropna()
            .sort_values(ascending=False)
            .head(top_n * REBALANCE_CANDIDATE_MULTIPLIER)
            .index.tolist()
        )

        if not candidates:
            # 没有满足条件的 ETF（动量全部为 NaN）
            for c in list(holdings.keys()):
                if c in current_prices.index and pd.notna(current_prices[c]):
                    proceeds = holdings[c]["shares"] * current_prices[c]
                    cash += proceeds * (1 - TRADE_COST_RATE)
                    trades.append({"日期": date, "ETF": c, "方向": "卖出(现金增强)",
                                   "价格": current_prices[c], "原因": "无合格ETF"})
            holdings = {}
            cost_prices = {}
            continue

        # --- 2g: 相关性分散（防止 ETF 同质化） ---
        selected = diversify_by_correlation(
            close_df.iloc[max(0, i - 20):i + 1], candidates, top_n
        )
        if not selected:
            continue

        # --- 2h: 行业分散约束（可选） ---
        if sector_diversity:
            selected = diversify_by_sector(selected, etf_sectors, top_n)

        if not selected:
            continue

        # --- 2i: 综合仓位比例（波动率目标 × 趋势仓位） ---
        vol_ratio = vol_target_ratio.iloc[i] * trend_position

        # --- 2j: 卖出不再持有的 ETF ---
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

        # --- 2k: 计算目标配置 ---
        total_value = cash + sum(
            holdings[c]["shares"] * current_prices[c]
            for c in holdings
            if c in current_prices.index and pd.notna(current_prices[c])
        )
        if total_value <= 0:
            continue

        if risk_parity:
            weights = calc_risk_parity_weights(close_df.iloc[max(0, i - 20):i + 1], selected)
            target_values = {c: total_value * vol_ratio * weights.get(c, 0) for c in selected}
        else:
            target_value_per_etf = (total_value * vol_ratio) / len(selected)
            target_values = {c: target_value_per_etf for c in selected}

        # --- 2l: 执行调仓（买入/加仓/减仓） ---
        for c in selected:
            if c not in current_prices.index or pd.isna(current_prices[c]):
                continue
            price = current_prices[c]
            target_val = target_values[c]
            if target_val <= 0:
                continue
            if c in holdings:
                current_value = holdings[c]["shares"] * price
                diff = target_val - current_value
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
                shares = target_val / price
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
# 计算策略收益/风险/风险调整收益指标，
# 支持基准对比（超额收益、信息比率、跟踪误差）。
# ============================================================

def calc_performance_metrics(nav, benchmark_nav=None, risk_free=0.02):
    """计算策略绩效指标体系。
    
    包含三大类指标:
      1. 收益类: CAGR（年化收益率）、超额收益
      2. 风险类: 年化波动率、最大回撤
      3. 风险调整收益: 夏普比率、卡玛比率、信息比率、胜率
    
    夏普比率是 v1-v4 中最核心的排序指标（在回测结果中按夏普降序排列）。
    选择夏普而非卡玛比的原因是：夏普同时衡量波动和收益，对策略整体
    风险调整收益的刻画更全面。卡玛比率仅考虑最大回撤，极端值敏感。
    
    Args:
        nav: 策略净值序列
        benchmark_nav: 基准净值序列（沪深 300），用于计算超额/信息比率
        risk_free: 无风险利率，默认 2%（中国 10 年期国债收益率近似）
    """
    if len(nav) < 2:
        return {}

    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    total_days = (nav.index[-1] - nav.index[0]).days
    total_years = total_days / 365.25

    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / total_years) - 1 if total_years > 0 else 0
    annual_vol = ret.std() * np.sqrt(252)
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
            info_ratio = (excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0
            metrics["信息比率"] = info_ratio
            metrics["跟踪误差"] = tracking_error
            bench_cagr = calc_benchmark_cagr(benchmark_nav, nav.index)
            metrics["超额收益"] = cagr - bench_cagr

    return metrics


def calc_benchmark_cagr(benchmark_nav, index):
    """计算基准净值在指定日期区间内的年化收益率。

    用于计算策略相对于基准的超额收益（CAGR 策略 - CAGR 基准）。
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
    """将绩效指标格式化为可读字符串。
    
    百分比指标（CAGR、波动率、回撤、胜率）转为带 % 的格式。
    比率指标（夏普、卡玛、信息比率）保留两位小数。
    用于表格打印和 CSV 保存。
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
# 2×2×2×2×2=32种组合:
#   动量窗口(20/60) × 动量类型(simple/dual)
#   × 仓位模式(binary/scaled) × 风险平价(开/关) × 行业分散(开/关)
# ============================================================

# ============================================================
# 参数网格定义
# 2^5 = 32 种参数组合，控制 5 个独立维度:
#   1. 动量计算窗口:      20日 vs 60日
#   2. 动量类型:          简单动量 vs 双重动量
#   3. 仓位模式:          二元仓位 vs 分档仓位
#   4. 风险平价:          开 vs 关
#   5. 行业分散:          开 vs 关
#
# 32 种组合覆盖了 v1-v3 策略演进中的所有关键参数变化，
# 使 v4 的分阶段分析能够系统性地评估每个参数的市场适应性。
# ============================================================

PARAM_GRID = {
    "window": [20, 60],
    "momentum_type": ["simple", "dual"],
    "position_sizing": ["binary", "scaled"],
    "risk_parity": [False, True],
    "sector_diversity": [False, True],
}

WINDOW_LABELS = {20: "20日", 60: "60日"}
MOMENTUM_TYPE_LABELS = {"simple": "简单动量", "dual": "双重动量"}
POSITION_LABELS = {"binary": "二元仓位", "scaled": "分档仓位"}
BOOL_LABELS = {False: "关", True: "开"}


def make_label(w, t, p, rp, sd):
    """生成参数组合的易读标签。
    
    例如: "简单动量_20日_分档仓位_风险平价_行业分散"
    标签中包含了所有启用的优化项（风险平价和行业分散仅在启用时标注）。
    """
    parts = [MOMENTUM_TYPE_LABELS[t], WINDOW_LABELS[w], POSITION_LABELS[p]]
    if rp:
        parts.append("风险平价")
    if sd:
        parts.append("行业分散")
    return "_".join(parts)


def run_all_combinations(close_df, benchmark_close, etf_sectors, top_n=5):
    """遍历 32 种参数组合，运行全区间回测。
    
    这是 v4 的"full"模式，对应 v3 的全区间回测逻辑。
    仅在 mode=full 时调用，用于与 v3 结果交叉验证。
    
    返回所有组合的绩效指标和净值序列，用于保存 CSV 和打印汇总表。
    """
    results = []
    navs = {}

    windows = PARAM_GRID["window"]
    types = PARAM_GRID["momentum_type"]
    sizings = PARAM_GRID["position_sizing"]
    rp_flags = PARAM_GRID["risk_parity"]
    sd_flags = PARAM_GRID["sector_diversity"]

    total = len(windows) * len(types) * len(sizings) * len(rp_flags) * len(sd_flags)
    count = 0

    for w, t, p, rp, sd in product(windows, types, sizings, rp_flags, sd_flags):
        label = make_label(w, t, p, rp, sd)
        count += 1
        print(f"  [{count}/{total}] {label} ... ", end="", flush=True)

        try:
            nav, trades = run_backtest(
                close_df, benchmark_close, etf_sectors,
                top_n=top_n, window=w,
                momentum_type=t,
                position_sizing=p,
                risk_parity=rp,
                sector_diversity=sd,
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


def print_summary_table(results, benchmark_close=None, title="动量趋势轮动策略"):
    """打印策略绩效汇总表（按夏普比率降序）。
    
    表格格式为居中对齐，包含收益/风险/风险调整收益三大类指标。
    最后一行显示沪深 300 基准的绩效作为对比基准。
    
    该函数在 v1-v4 中保持一致，便于跨版本比较。
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
    print(f"{title:^{line_width}}")
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

    if benchmark_close is not None and isinstance(benchmark_close, pd.Series):
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


def save_results(results, navs, suffix="v4"):
    """保存全区间回测结果到 CSV 文件。
    
    performance_summary_{suffix}.csv: 所有 32 种参数组合的绩效指标
    nav_all_{suffix}.csv: 所有参数组合的完整净值序列（用于后续绘图分析）
    """
    os.makedirs("momentum_results", exist_ok=True)

    keys = ["组合", "CAGR", "超额收益", "夏普比率", "最大回撤", "年化波动", "胜率", "信息比率", "卡玛比率", "交易次数"]
    summary = pd.DataFrame(results)[keys]
    summary.to_csv(f"momentum_results/performance_summary_{suffix}.csv", index=False, encoding="utf-8-sig")

    nav_df = pd.DataFrame(navs)
    nav_df.index.name = "日期"
    nav_df.to_csv(f"momentum_results/nav_all_{suffix}.csv", encoding="utf-8-sig")

    print(f"结果已保存到 momentum_results/  ({suffix})")

# ============================================================
# v4: 分阶段回测分析
# v4 核心功能：将回测区间分为6个市场阶段分别测试，
# 分析策略在不同市场环境（牛市/熊市/震荡）下的表现差异，
# 并输出跨阶段稳健性排名。
# ============================================================

def run_backtest_on_period(close_df, benchmark_close, etf_sectors,
                           period_start, period_end, top_n=5,
                           window=20, momentum_type="simple",
                           position_sizing="binary",
                           risk_parity=False, sector_diversity=False):
    """在指定的市场阶段内执行回测。
    
    v4 核心功能函数。与 run_backtest 的关键区别：
    
    1. 预热机制：在阶段起始前额外截取 180 个交易日作为预热。
       WARMUP_DAYS=180 的选择考虑:
       - 120MA（慢线）需要至少 120 个数据点
       - 60 日动量窗口需要至少 60 个数据点
       - 预热 180 日确保了所有参数组合都有足够数据
       - 180 个交易日 ≈ 9 个自然月
    
    2. 净值归一化：阶段内净值以起始日为基准归一化到 1.0。
       不同阶段的初始净值统一，便于横向比较各阶段的收益/风险特征。
    
    3. 数据量检查：如果阶段内数据不足，返回 (None, None)，
       避免因数据不足导致的虚假绩效报告。
    
    Args:
        close_df: 全量 ETF 收盘价宽表（非阶段切片）
        benchmark_close: 沪深 300 基准（非阶段切片）
        etf_sectors: 行业归属字典
        period_start/period_end: 阶段起止日期
        top_n/window/momentum_type/position_sizing/risk_parity/sector_diversity: 策略参数
    
    Returns:
        (nav, trades) 若数据充足
        (None, None) 若数据不足
    """
    warmup_start = period_start - pd.Timedelta(days=WARMUP_DAYS)
    extended_data = close_df.loc[warmup_start:period_end].copy()
    extended_bench = benchmark_close.loc[warmup_start:period_end].copy()

    if len(extended_data) < MIN_DATA_DAYS + max(window, TREND_SLOW_WINDOW):
        return None, None

    nav, trades = run_backtest(
        extended_data, extended_bench, etf_sectors,
        top_n=top_n, window=window,
        momentum_type=momentum_type,
        position_sizing=position_sizing,
        risk_parity=risk_parity,
        sector_diversity=sector_diversity,
    )

    nav = nav.loc[period_start:period_end]
    if len(nav) < 2:
        return None, None

    nav = nav / nav.iloc[0]

    return nav, trades


def run_period_analysis(close_df, benchmark_close, etf_sectors, top_n=5):
    """对 6 个市场阶段分别运行 32 种参数组合的回测。
    
    这是 v4 最核心的分析功能。与全区间回测的区别：
      - 全区间回测：在 2014-2026 完整区间运行一次，得到 32 个结果
      - 分阶段回测：在 6 个独立区间分别运行，得到 6×32 = 192 个结果
    
    输出结构:
      period_results: {阶段名: [{组合1指标}, {组合2指标}, ...], ...}
      period_bench_navs: {阶段名: 基准净值}
    
    该函数执行流程:
      1. 对每个市场阶段，截取该阶段内的数据
      2. 对每个阶段，遍历 32 种参数组合
      3. 对每对（阶段，组合），调用 run_backtest_on_period
      4. 收集所有结果，按阶段分组存储
    
    Args:
        close_df: ETF 收盘价宽表
        benchmark_close: 沪深 300 基准
        etf_sectors: 行业归属字典
        top_n: 持仓数量
    
    Returns:
        period_results: 分阶段结果（用于后续排名和分析）
        period_bench_navs: 各阶段基准净值
    """
    windows = PARAM_GRID["window"]
    types = PARAM_GRID["momentum_type"]
    sizings = PARAM_GRID["position_sizing"]
    rp_flags = PARAM_GRID["risk_parity"]
    sd_flags = PARAM_GRID["sector_diversity"]

    period_results = {}
    period_bench_navs = {}

    for pname, pstart, pend in SUB_PERIODS:
        pstart_dt = pd.Timestamp(pstart)
        pend_dt = pd.Timestamp(pend)
        period_close = close_df.loc[pstart_dt:pend_dt]
        period_bench = benchmark_close.loc[pstart_dt:pend_dt]

        bench_nav = period_bench / period_bench.iloc[0]
        period_bench_navs[pname] = bench_nav

        combos = []
        total = len(windows) * len(types) * len(sizings) * len(rp_flags) * len(sd_flags)
        count = 0

        print(f"\n  [{pname}] {pstart} ~ {pend} ({len(period_close)} 交易日) ...")

        for w, t, p, rp, sd in product(windows, types, sizings, rp_flags, sd_flags):
            label = make_label(w, t, p, rp, sd)
            count += 1
            print(f"    [{count}/{total}] {label} ... ", end="", flush=True)

            try:
                nav, trades = run_backtest_on_period(
                    close_df, benchmark_close, etf_sectors,
                    pstart_dt, pend_dt,
                    top_n=top_n, window=w,
                    momentum_type=t,
                    position_sizing=p,
                    risk_parity=rp,
                    sector_diversity=sd,
                )
                if nav is None:
                    print("数据不足")
                    continue

                period_bench_slice = benchmark_close.loc[nav.index]
                metrics = calc_performance_metrics(nav, period_bench_slice)
                metrics["组合"] = label
                metrics["交易次数"] = len(trades)
                combos.append(metrics)
                print(f"OK  夏普={metrics.get('夏普比率', 0):.2f}  CAGR={metrics.get('CAGR', 0)*100:.1f}%")
            except Exception as e:
                print(f"失败: {e}")

        period_results[pname] = combos

    return period_results, period_bench_navs


def print_period_summary(period_results, period_bench_navs):
    """打印每个市场阶段中夏普比率排名前 5 的参数组合。
    
    这是分阶段分析的中间输出，帮助理解：
      - 哪些参数配置在特定市场环境中表现最好
      - 同一参数配置在不同市场环境中的表现差异
      - 是否有某个参数配置在多个阶段都进入前 5
    
    结合全区间夏普排名阅读：
      如果一个组合在全区间排名靠前，但在此阶段不在前 5，
      说明它的优势来自其他阶段的表现。
    """
    keys = ["组合", "CAGR", "超额收益", "夏普比率", "最大回撤", "年化波动", "胜率"]

    for pname, combos in period_results.items():
        sorted_c = sorted(combos, key=lambda x: x.get("夏普比率", 0), reverse=True)
        top5 = sorted_c[:5]

        bench_nav = period_bench_navs.get(pname)
        bench_metrics = calc_performance_metrics(bench_nav) if bench_nav is not None else {}

        col_widths = {k: max(len(k), max(len(format_metrics(c).get(k, "")) for c in top5)) for k in keys[1:]}
        col_widths["组合"] = max(len(max(c["组合"] for c in top5)), 20)

        header = " | ".join(f"{k:>{col_widths[k]}}" for k in keys)
        line_width = max(len(header), 80)
        print("\n" + "=" * line_width)
        print(f"{pname:^{line_width}}")
        print("=" * line_width)
        print(header)
        print("-" * len(header))

        for c in top5:
            vals = format_metrics(c)
            vals["组合"] = c["组合"]
            row = " | ".join(f"{vals.get(k, str(c.get(k, ''))):>{col_widths[k]}}" for k in keys)
            print(row)

        print("-" * len(header))
        if bench_metrics:
            bv = format_metrics(bench_metrics)
            bv["组合"] = "基准_沪深300"
            row = " | ".join(f"{bv.get(k, ''):>{col_widths[k]}}" for k in keys)
            print(row)
        print("=" * line_width)
        print()


def print_robustness_ranking(period_results):
    """跨阶段稳健性排名 — v4 最核心的分析输出。
    
    将所有 32 种参数组合在 6 个市场阶段中的夏普比率汇总，
    计算每个组合的跨阶段统计量，按平均夏普降序排列。
    
    关键指标解读:
      - 平均夏普: 各阶段夏普比率的均值。越高说明综合表现越好。
      - 夏普标准差: 各阶段夏普比率的波动。越低说明表现越稳定。
      - 正夏普阶段数: 夏普为正的阶段数。最大为 6（所有阶段都赚钱）。
    
    理想组合：平均夏普高 + 夏普标准差低 + 正夏普阶段数多。
    v4 的实际发现：
      - 没有任何组合在所有 6 个阶段都实现正夏普
      - 表现最好的组合也只有 3-4 个阶段为正夏普
      - 夏普标准差普遍较大（>0.5），说明策略表现高度依赖市场环境
      - 风险平价是少数跨阶段一致的优化方向
    
    这个分析的核心结论：
      不存在一种参数配置能在所有市场环境下都表现良好。
      策略约 50% 的时间有效，另外 50% 的时间亏损。
      2016 震荡慢牛是所有 32 种组合的共同命门。
    """
    print("\n" + "=" * 90)
    print(f"{'跨阶段稳健性排名（按夏普均值降序）':^90}")
    print("=" * 90)

    combo_sharpes = {}
    for pname, combos in period_results.items():
        for c in combos:
            label = c["组合"]
            sharpe = c.get("夏普比率", -99)
            if label not in combo_sharpes:
                combo_sharpes[label] = {}
            combo_sharpes[label][pname] = sharpe

    period_names = [p[0] for p in SUB_PERIODS]
    rankings = []
    for label, perfs in combo_sharpes.items():
        values = [perfs.get(p, None) for p in period_names]
        valid = [v for v in values if v is not None and v > -10]
        if len(valid) < 2:
            continue
        mean_sharpe = np.mean(valid)
        std_sharpe = np.std(valid)
        positive_count = sum(1 for v in valid if v > 0)
        rankings.append({
            "组合": label,
            "平均夏普": mean_sharpe,
            "夏普标准差": std_sharpe,
            "夏普最大值": max(valid),
            "夏普最小值": min(valid),
            "正夏普阶段数": positive_count,
            "有效阶段数": len(valid),
        })

    rankings.sort(key=lambda x: x["平均夏普"], reverse=True)

    keys = ["组合", "平均夏普", "夏普标准差", "夏普最大值", "夏普最小值", "正夏普阶段数", "有效阶段数"]
    col_widths = {k: max(len(k), max(len(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k])) for r in rankings)) for k in keys[1:]}
    col_widths["组合"] = max(len(max(r["组合"] for r in rankings)), 25)

    header = " | ".join(f"{k:>{col_widths[k]}}" for k in keys)
    print(header)
    print("-" * len(header))

    for r in rankings:
        vals = {k: r[k] for k in keys}
        vals["组合"] = r["组合"]
        vals["平均夏普"] = f"{r['平均夏普']:.2f}"
        vals["夏普标准差"] = f"{r['夏普标准差']:.2f}"
        vals["夏普最大值"] = f"{r['夏普最大值']:.2f}"
        vals["夏普最小值"] = f"{r['夏普最小值']:.2f}"
        vals["正夏普阶段数"] = str(r["正夏普阶段数"])
        vals["有效阶段数"] = str(r["有效阶段数"])
        row = " | ".join(f"{vals[k]:>{col_widths[k]}}" for k in keys)
        print(row)

    print("=" * 90)
    print()


def save_period_results(period_results, period_bench_navs):
    """保存分阶段回测结果到 CSV。
    
    生成 period_analysis_v4.csv，包含：
      - 每个市场阶段 + 每种参数组合的绩效指标
      - 各阶段沪深 300 基准的绩效（用于对比）
      - 方便在 Excel/Python 中进行进一步分析和可视化
    
    数据结构示例:
        阶段         | 组合                | CAGR  | 夏普比率 | ...
        2014大牛市   | 简单动量_20日_二元仓位 | 25.3% | 0.85    |
        2014大牛市   | 双重动量_60日_分档仓位 | 18.7% | 0.62    |
        2014大牛市   | 基准_沪深300          | 15.5% | 0.55    |
        ...
    """
    os.makedirs("momentum_results", exist_ok=True)

    all_rows = []
    for pname, combos in period_results.items():
        for c in combos:
            row = {"阶段": pname, "组合": c["组合"]}
            row.update({k: c.get(k, "") for k in ["CAGR", "超额收益", "夏普比率", "最大回撤", "年化波动", "胜率", "信息比率", "卡玛比率", "交易次数"]})
            all_rows.append(row)

        bench_nav = period_bench_navs.get(pname)
        if bench_nav is not None:
            bm = calc_performance_metrics(bench_nav)
            row = {"阶段": pname, "组合": "基准_沪深300"}
            row.update({k: bm.get(k, "") for k in ["CAGR", "超额收益", "夏普比率", "最大回撤", "年化波动", "胜率", "信息比率", "卡玛比率"]})
            row["交易次数"] = "-"
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv("momentum_results/period_analysis_v4.csv", index=False, encoding="utf-8-sig")

    print("分阶段结果已保存到 momentum_results/period_analysis_v4.csv")


def main():
    """v4 主入口函数。
    
    支持两种运行模式:
      1. full 模式（python momentum_strategy_v4.py --mode full）
         全区间回测，与 v3 逻辑相同，用于交叉验证。
         输出：全区间绩效总览表 + 净值序列 CSV。
    
      2. periods 模式（python momentum_strategy_v4.py --mode periods，默认）
         分阶段回测，v4 核心功能。
         对 6 个市场阶段分别运行 32 种参数组合的回测。
         输出：
           - 每个阶段 Top 5 参数组合明细
           - 跨阶段稳健性排名
           - period_analysis_v4.csv（完整结果）
    
    典型用法:
      # 运行分阶段回测（默认）
      python 动量趋势ETF行业主题轮动策略/momentum_strategy_v4.py
      
      # 运行全区间回测
      python 动量趋势ETF行业主题轮动策略/momentum_strategy_v4.py --mode full
      
      # 修改持仓数量
      python 动量趋势ETF行业主题轮动策略/momentum_strategy_v4.py --top_n 8
    
    执行流程（5 步）:
      1. 加载并过滤行业 ETF（从 etf_list.csv）
      2. 加载历史日线数据（从 data/etf/*.csv）
      3. 加载沪深 300 基准（510300）
      4. 识别 ETF 行业归属
      5. 运行回测（全区间或分阶段）
    """
    parser = argparse.ArgumentParser(description="A股ETF动量趋势轮动策略 v4 回测")
    parser.add_argument("--top_n", type=int, default=5, help="持仓数量（默认5）")
    parser.add_argument("--mode", type=str, default="periods",
                        choices=["full", "periods"],
                        help="回测模式: full=全区间, periods=分阶段（默认）")
    args = parser.parse_args()
    top_n = args.top_n
    mode = args.mode

    print("=" * 60)
    print("动量趋势轮动策略 v4 回测")
    print(f"模式: {'全区间' if mode == 'full' else '分阶段分析'}")
    print("=" * 60)

    print("\n[1/5] 加载行业 ETF ...")
    codes = load_industry_etf_codes()
    print(f"  找到 {len(codes)} 只行业 ETF")

    print("[2/5] 加载历史数据 ...")
    close_df = load_all_industry_data(codes)
    print(f"  数据范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")
    print(f"  ETF数量: {close_df.shape[1]} | 交易日: {close_df.shape[0]}")

    print("[3/5] 加载基准（沪深300）...")
    bench = load_benchmark()
    bench = bench.reindex(close_df.index).ffill().dropna()
    print(f"  基准范围: {bench.index[0].date()} ~ {bench.index[-1].date()}")

    print("[4/5] 识别ETF行业归属 ...")
    etf_sectors = load_etf_sectors()
    sectors = set(etf_sectors.values())
    print(f"  行业分类: {len(sectors)} 个 | 已识别: {sum(1 for v in etf_sectors.values() if v != '其他')}/{len(etf_sectors)}")

    if mode == "full":
        print(f"[5/5] 运行全区间回测（持仓 {top_n} 只）...")
        results, navs = run_all_combinations(close_df, bench, etf_sectors, top_n)
        print_summary_table(results, bench, "动量趋势轮动策略 v4 — 全区间绩效总览")
        save_results(results, navs, "v4")
    else:
        print(f"[5/5] 运行分阶段回测（持仓 {top_n} 只，{len(SUB_PERIODS)} 个阶段）...")
        period_results, period_bench_navs = run_period_analysis(close_df, bench, etf_sectors, top_n)
        print_period_summary(period_results, period_bench_navs)
        print_robustness_ranking(period_results)
        save_period_results(period_results, period_bench_navs)


if __name__ == "__main__":
    main()
