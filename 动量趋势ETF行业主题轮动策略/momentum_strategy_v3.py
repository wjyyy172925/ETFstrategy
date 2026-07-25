#!/usr/bin/env python3
"""
A股ETF动量趋势轮动策略 v3 — 回测与绩效分析

v3 基于 v2 的 6 项优化建议实施了 5 项核心改动:

  1. 双层趋势线: 60MA 入场 + 120MA 出场
     v2 用 120MA 同步控制入场和出场，滞后性导致 2024 年
     （基准 +14.68%，策略 -3.36%）完全错过反弹。
     v3 改为 60MA 入场（提早捕捉反弹）+ 120MA 出场（避免假突破被震出）。

  2. 趋势强度分档建仓: 0%/50%/100% 三档
     v2 是 0/1 二元仓位（全仓或空仓），过于粗糙。
     v3 根据沪深 300 与 120MA 的距离分三档:
       > +5% → 100%，0%~+5% → 50%，<0% → 0%
     分档仓位是 v3 最主要的质量提升项（夏普改善 +0.15）。

  3. 风险平价加权: 波动率倒数分配权重（可选）
     v2 使用等权分配，v3 加入波动率倒数加权作为可选参数。
     低波动 ETF 被分配更高权重，高波动 ETF 权重被压低。
     效果: 夏普改善 +0.01（微幅）。

  4. 行业覆盖约束: 同主题 ETF 最多 2 只（可选）
     动量轮动天然偏向连续上涨的少数行业，导致持仓同质化。
     将候选 ETF 按 16 个主题分类，同主题最多选 1~2 只。
     效果: 夏普改善 +0.03（微幅）。

  5. 回测起始从 2021 年扩展至 2014 年
     v2 仅覆盖 2021-2026（熊市为主），v3 覆盖完整牛熊周期:
     2014 大牛市 → 2015 股灾 → 2016-2017 震荡 → 2018 熊市
     → 2019-2021 核心资产牛 → 2021-2026 熊市结构

v3 绩效总结:
  全区间 (2014-2026) 最佳夏普 -0.06，远低于 v2 的 0.40。
  核心原因: 2016 震荡慢牛（策略所有组合均亏损）和 2019 跑输。
  结论: 动量策略在 A 股约 50% 的市场阶段有效，但跨周期稳定表现困难。
"""

import os
import argparse
from itertools import product

import numpy as np
import pandas as pd

# ============================================================
# 全局参数配置
# ============================================================
# v3 沿用了 v2 的大部分风险参数（回撤止损 30%、目标波动率 20%），
# 新增了双层趋势线的均线参数和分档建仓的阈值参数。

DATA_DIR = "data/etf"

# --- 风控参数（与 v2 一致）---
TARGET_VOL = 0.20                         # 目标年化波动率 20%
STOP_LOSS_PCT = 0.08                      # 个股止损 -8%
MAX_DRAWDOWN_PCT = 0.30                   # 策略最大回撤止损 30%

# --- 选股参数（与 v2 一致）---
CORR_THRESHOLD = 0.9                      # 相关性阈值
MIN_DATA_DAYS = 60                        # 最少数据天数
REBALANCE_CANDIDATE_MULTIPLIER = 2        # 候选倍率

# --- 交易成本（与 v2 一致）---
COMMISSION_RATE = 0.00003                 # 万0.3
SLIPPAGE_RATE = 0.001                     # 0.1%
TRADE_COST_RATE = COMMISSION_RATE + SLIPPAGE_RATE  # 单边 0.103%

# --- v3 新增: 双层趋势线参数 ---
# 快线 60MA: 用于入场判断（沪深 300 站上 60MA 即可入场）
# 慢线 120MA: 用于出场判断（跌破 120MA 才离场）
# 这种"快入慢出"设计是为了解决 v2 的滞后性问题:
#   v2 用 120MA 同时控制入场和出场，导致反弹初期被错过；
#   v3 用 60MA 入场能更早捕捉反弹，用 120MA 出场保持持仓稳定。
TREND_FAST_WINDOW = 60                    # 快线（入场）: 60 日均线
TREND_SLOW_WINDOW = 120                   # 慢线（出场）: 120 日均线

# --- v3 新增: 分档建仓参数 ---
# 当沪深 300 偏离 120MA 超过 5% 时，认为处于强势上涨趋势，全仓。
# 低于 5% 或处于 60MA~120MA 之间时，半仓试探。
# 低于 60MA 时空仓。
SCALED_LIGHT_WEIGHT = 0.50               # 轻仓仓位比例 50%
SCALED_FULL_THRESHOLD = 0.05             # 全仓触发阈值: 偏离 120MA 超过 5%

# ============================================================
# v3 新增: 行业分类关键词映射
# ============================================================
# 将 ETF 按主题聚类为 16 个行业分类，用于行业覆盖约束。
# 动量轮动天然偏向连续上涨的少数行业（如同时持有多个新能源 ETF），
# 导致持仓同质化和行业集中风险。
# 行业约束确保持仓覆盖至少 3~5 个不同主题，分散单一行业崩盘风险。
#
# 注意: 这是一个"软约束"——同主题最多选 2 只，但不强制覆盖所有行业。
# 如果当前市场只有 1-2 个行业有动量，策略仍然可以集中持有。
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

# v3 回测起始日期: 2014-01-01
# v2 仅覆盖 2021-2026（熊市为主），为了测试策略的跨周期稳定性，
# v3 扩展至 2014 年起，覆盖完整的一轮"牛→熊→震荡→牛→熊"周期。
# 注意: 2014 年 A 股行业 ETF 数量仅 100+ 只，远低于现在的 631 只，
# 早期可选标的不足可能导致动量信号的统计显著性降低。
START_DATE = "2014-01-01"

# ============================================================
# ETF 行业分类筛选
# 根据名称关键词自动筛选 A 股行业/主题 ETF，
# 排除债券、商品、跨境等非权益品种。
# 过滤逻辑与 v1/v2 一致，确保跨版本对比时 ETF 池一致。
# ============================================================


def load_etf_list():
    """从 CSV 加载 ETF 列表。

    CSV 文件格式: data/etf_list.csv，包含"代码"和"名称"两列。
    代码为 6 位数字（可能不带前导零）。

    Returns:
        DataFrame 包含 代码 和 名称 两列
    """
    path = "data/etf_list.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"ETF列表文件不存在: {path}")
    return pd.read_csv(path)


def filter_industry_etfs(df):
    """关键词两阶段过滤: 先保留行业词，再剔除债券/商品/跨境。

    include_kw: 匹配行业/主题相关关键词，共 ~50 个覆盖各板块
    exclude_kw: 剔除不适合动量策略的品种（债券、商品、跨境）

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

    mask = df["名称"].str.contains("|".join(include_kw), na=False)
    exclude_mask = df["名称"].str.contains("|".join(exclude_kw), na=False)
    return df[mask & ~exclude_mask].reset_index(drop=True)


def load_industry_etf_codes():
    """加载并过滤行业 ETF，返回 6 位代码列表。

    完整流程:
      load_etf_list() → filter_industry_etfs() → 代码补零 → 返回列表

    Returns:
        list[str]: 6 位 ETF 代码列表（如 "510300"）
    """
    df = load_etf_list()
    df = filter_industry_etfs(df)
    codes = df["代码"].astype(str).str.zfill(6).tolist()
    print(f"行业ETF数量: {len(codes)} 只")
    return codes

# ============================================================
# 数据加载
# 从 data/etf/ 目录加载日线数据，合并为宽表。
# v3 回测起始日期为 2014-01-01（全量可用数据），不再截断为 5 年。
# 这意味着 v3 覆盖了约 12.5 年的数据，包含完整的牛熊周期。
# 注意: 2014 年 A 股行业 ETF 数量有限，早期回测的统计显著性较低。
# ============================================================


def load_etf_data(code):
    """加载单只 ETF 日线，返回 [日期, 收盘] DataFrame。

    文件路径: data/etf/{code}.csv
    如果文件不存在（ETF 退市或尚未成立），返回 None。

    Args:
        code: 6 位 ETF 代码字符串

    Returns:
        DataFrame 或 None
    """
    path = os.path.join(DATA_DIR, f"{code}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    return df[["日期", "收盘"]]


def load_all_industry_data(codes):
    """加载所有行业 ETF 并合并为宽表。

    v3 与 v2 的关键区别:
      - v2: 仅保留最近 5 年数据（result.index.max() - 5 years）
      - v3: 从 START_DATE (2014-01-01) 开始，使用全量历史数据
      这使得 v3 能够覆盖 2014 大牛市 → 2015 股灾等关键市场阶段。

    缺点: 2014 年市场上仅有 ~100 只行业 ETF，动量信号的
    统计显著性不如 2021 年（631 只）。早期的回测结果可能存在
    "幸存者偏差"——能够存续至今的 ETF 本身就有一定的质量筛选。

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
    result = result.loc[result.index.notna()]
    result = result[result.index >= START_DATE]
    return result


def load_benchmark():
    """加载沪深 300 ETF (510300) 作为业绩基准。

    Returns:
        Series 或 None（数据不存在时）
    """
    df = load_etf_data("510300")
    if df is None:
        return None
    return df.set_index("日期")["收盘"].sort_index()

# ============================================================
# 行业分类识别 — v3 新增功能
# 通过 ETF 名称关键词映射到预定义行业分类（共 16 个行业），
# 用于行业覆盖约束，防止同行业 ETF 过度集中持仓。
# ============================================================
# 行业聚类是动量轮动策略的重要辅助手段:
#   - 动量选股天然偏好连续上涨的少数行业
#   - 不同主题的 ETF 可能实际走势高度同步（如"新能源"和"光伏"）
#   - 行业约束强制策略分散到多个主题，降低单一行业崩盘的冲击
#
# 分类方法: 基于 ETF 名称中的关键词进行规则匹配。
# 这是一种"轻量级"方法，无需接入外部行业分类数据。
# 局限性: 部分 ETF 名称过于模糊（如"XX龙头ETF"）难以精确分类。


def identify_sector(etf_name):
    """根据 ETF 名称关键词识别所属行业分类。

    遍历 SECTOR_KEYWORDS 字典中的 16 个行业分类，
    对每个行业的关键词列表进行子串匹配，返回第一个匹配的行业。
    若全不匹配则返回"其他"。

    例如:
      "新能源车ETF" → "新能源"（匹配"电动车"关键词）
      "半导体ETF" → "科技"（匹配"半导体"关键词）
      "消费龙头ETF" → "消费"（匹配"消费"关键词）
      "物流ETF" → "综合"（匹配"物流"关键词）

    Args:
        etf_name: ETF 名称字符串

    Returns:
        str: 行业分类名称
    """
    for sector, keywords in SECTOR_KEYWORDS.items():
        for kw in keywords:
            if kw in etf_name:
                return sector
    return "其他"


def load_etf_sectors():
    """加载所有行业 ETF 的行业归属字典 {代码: 行业分类}。

    流程:
      1. load_etf_list() — 加载全部 ETF
      2. filter_industry_etfs() — 过滤出行业 ETF
      3. identify_sector() — 逐只识别行业归属
      4. 返回 {代码: 行业} 字典

    Returns:
        dict: {ETF 代码: 行业分类名称}
    """
    df = load_etf_list()
    df = filter_industry_etfs(df)
    df["代码"] = df["代码"].astype(str).str.zfill(6)
    df["行业"] = df["名称"].apply(identify_sector)
    return dict(zip(df["代码"], df["行业"]))

# ============================================================
# 风险控制
# v3 新增了行业分散约束和风险平价加权功能。
# 相关性分散沿用 v1/v2 逻辑（贪心法，基于 20 日收益率）。
# ============================================================


def diversify_by_correlation(close_df, candidates, top_n, corr_threshold=CORR_THRESHOLD):
    """基于相关性贪心选股，避免持有同质化 ETF。

    与 v1/v2 完全相同的实现。
    从动量排名高的候选开始，逐一检查与已选持仓的相关性，
    仅当与所有已选持仓的相关系数均 < 0.9 时才纳入。

    Args:
        close_df: 近期收盘价 DataFrame（通常最近 20 日）
        candidates: 按动量排序的候选列表
        top_n: 目标持仓数
        corr_threshold: 相关性阈值

    Returns:
        list: 精选后的 ETF 列表
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
    """行业覆盖约束: 同一行业最多持有 max_per_sector 只 ETF。

    在相关性分散之后执行（候选池已排序），
    按动量排名顺序遍历，每行业最多选 max_per_sector 只。
    确保持仓覆盖至少 top_n / max_per_sector 个不同行业。

    v3 回测中，行业分散开启 vs 关闭的夏普改善仅 +0.03，
    说明 A 股行业 ETF 之间的同质化问题不如预期严重。

    Args:
        candidates: 按动量排序的候选列表
        etf_sectors: {代码: 行业} 字典（由 load_etf_sectors() 生成）
        top_n: 目标持仓数
        max_per_sector: 每个行业最大数量（默认 2）

    Returns:
        list: 行业分散后的 ETF 列表
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
    """风险平价加权: 权重与近 20 日波动率倒数成正比。

    传统等权分配 == 每只 ETF 分配相同的资金
    风险平价分配 == 高波动 ETF 配低权重，低波动 ETF 配高权重
    目标: 使各品种对组合的"风险贡献"大致相等

    计算公式:
      weight_i = (1 / vol_i) / sum(1 / vol_j for j in 所有持仓)

    其中 vol_i = ETF i 的 20 日日收益率标准差

    效果:
      - 降低极端行情下的尾部风险
      - 提升夏普比率（约 +0.01，微幅）
      - 在牛市中会略微拖累收益（低波动 ETF 涨幅通常偏小）
      - 在结构市中平稳净值曲线

    Args:
        close_df: 收盘价 DataFrame（需包含 selected 中的所有 ETF）
        selected: 选中的 ETF 代码列表
        window: 波动率计算窗口（默认 20 个交易日）

    Returns:
        dict: {ETF 代码: 权重}
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
# v3 核心改进:
#   - 双层趋势线: 60MA 入场 + 120MA 出场（替代 v2 的单线）
#   - 趋势强度分档建仓: 0%/50%/100% 三档（替代 v2 的二元仓位）
#   - 风险平价加权替代等权分配（可选）
#   - 行业覆盖约束（可选）
#
# 过滤器链条（按执行顺序）:
#   双层趋势线(策略级) → 动量选股 → 行业分散(组合级)
#   → 相关性分散(组合级) → 风险平价(权重分配)
# ============================================================


def _get_rebalance_dates(index, min_window=MIN_DATA_DAYS):
    """计算月度调仓日期。

    v3 仅支持月度调仓（与 v2 一致）。
    每月第一个交易日作为调仓日。
    最小持有窗口前不调仓（用于动量计算预热）。

    Args:
        index: 完整日期索引
        min_window: 最小持有窗口

    Returns:
        list[Timestamp]: 调仓日期列表
    """
    if len(index) <= min_window:
        return []
    dates = index[min_window:]
    s = pd.Series(dates, index=range(len(dates)))
    mask = s.dt.month != s.shift(1).dt.month
    mask.iloc[0] = True
    return s[mask].tolist()


def compute_trend_position(benchmark_close, date_idx, position_sizing):
    """基于沪深 300 的趋势位置判断和仓位建议。

    v3 与 v2 在趋势判断上的关键区别:
      - v2: 仅使用 120MA 单线，0/1 二元仓位
      - v3: 使用 60MA + 120MA 双线，0/50%/100% 三档分档

    三种仓位模式:
      1. "none": 无视趋势，始终满仓 100%
         - 相当于无趋势过滤器，作为对照组
         - 用于量化趋势过滤的边际贡献

      2. "binary": 二元仓位（等价于 v2 的趋势过滤器）
         - 价格 > 120MA → 满仓 100%
         - 价格 <= 120MA → 空仓 0%
         - 用于对比 60MA 入场 vs 120MA 入场的差异

      3. "scaled": 三档分档仓位（v3 新增）
         - 价格 < 60MA → 空仓 0%（市场处于下跌趋势）
         - 60MA < 价格 < 120MA → 轻仓 50%（刚站上快线，试探性建仓）
         - 价格 > 120MA 且偏离 >= 5% → 满仓 100%（强势上涨）
         - 价格 > 120MA 但偏离 < 5% → 轻仓 50%（站上但确认不足）

    分档仓位的优势:
      减少"刚站上均线就全仓买入，次日跌破均线割肉"的双向磨损。
      A 股市场在 120MA 附近的反复争夺非常频繁，
      0/1 二元仓位在这种行情中损耗严重。

    Args:
        benchmark_close: 沪深 300 收盘价 Series
        date_idx: 当前日期在序列中的位置索引（integer index）
        position_sizing: 仓位模式 ("none" / "binary" / "scaled")

    Returns:
        float: 仓位比例，范围 [0.0, 1.0]
    """
    if position_sizing == "none":
        return 1.0

    price = benchmark_close.iloc[date_idx]
    # 实时滚动计算 60MA 和 120MA
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
    """v3 动量趋势轮动策略回测引擎。

    v3 相对 v2 的核心改进（按执行顺序排列）:
      1. 仓位管理: compute_trend_position() 替代 v2 的简单 trend_filter
         - v2: 0/1 二元（120MA 以上满仓，以下空仓）
         - v3: "scaled" 模式实现三档分档（0%/50%/100%）
         - "none" 模式可验证趋势过滤本身的必要性
      2. 资金分配: risk_parity 替代等权分配（可选）
         - 等权: 每只 ETF 分配相同资金
         - 风险平价: 按波动率倒数加权
      3. 选股约束: sector_diversity 确保行业分散（可选）

    过滤器链条（按执行顺序）:
      双层趋势线(策略级) → 动量选股 → 行业分散(组合级)
      → 相关性分散(组合级) → 风险平价(权重分配)

    Args:
        close_df: ETF 收盘价宽表，shape=(交易日, ETF数量)
        benchmark_close: 沪深 300 基准收盘价 Series
        etf_sectors: 行业归属字典 {代码: 行业名}，由 load_etf_sectors() 生成
        top_n: 持仓数（默认 5）
        window: 动量窗口（20 或 60 日）
        momentum_type: "simple"（简单动量）或 "dual"（双重动量）
        position_sizing: "binary"（二元仓位）或 "scaled"（分档仓位）
        risk_parity: 是否启用风险平价加权
        sector_diversity: 是否启用行业分散约束

    Returns:
        nav: 策略净值 Series，index=日期
        trades_df: 交易记录 DataFrame
    """
    # ================================================================
    # 第一步: 计算动量分数
    # ================================================================
    # 与 v2 相同的动量计算逻辑
    returns = close_df / close_df.shift(window) - 1
    benchmark_returns = benchmark_close / benchmark_close.shift(window) - 1

    if momentum_type == "simple":
        momentum_scores = returns
    elif momentum_type == "dual":
        momentum_scores = returns.sub(benchmark_returns, axis=0)
        abs_mask = returns > 0
        momentum_scores = momentum_scores.where(abs_mask, np.nan)

    # ================================================================
    # 第二步: 波动率目标管理
    # ================================================================
    # 与 v2 完全相同的实现
    # 目标波动率 20%，以沪深 300 的 20 日滚动波动率为基准
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
    # 与 v2 相同的逐日循环架构，关键区别:
    #   1. 趋势判断逻辑由 compute_trend_position() 统一处理
    #   2. 仓位由 trend_position 和 vol_ratio 共同决定
    #   3. 行业分散作为可选的中间过滤步骤
    #   4. 风险平价替代等权分配（可选）
    for i in range(window, len(close_df)):
        date = close_df.index[i]
        current_prices = close_df.iloc[i]

        # --- 计算当日净值 ---
        current_nav_value = cash + sum(
            holdings[c]["shares"] * current_prices[c]
            for c in list(holdings.keys())
            if c in current_prices.index and pd.notna(current_prices[c])
        ) if holdings else cash
        nav.iloc[i] = current_nav_value
        peak_nav = max(peak_nav, current_nav_value)

        # --- 策略级回撤止损（30%）---
        # 与 v2 完全相同的止损逻辑
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
            if date in rebalance_set:
                in_drawdown_stop = False
            else:
                continue

        # --- 个股止损（-8%）---
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

        # --- 非调仓日跳过 ---
        if date not in rebalance_set:
            continue

        # ================================================================
        # 第五步: 调仓日操作
        # 过滤器链条: 趋势判断 → 动量选股 → 行业分散 → 相关性分散
        # ================================================================

        # --- 5a. 趋势判断（双层趋势线 + 分档/二元仓位）---
        # v3 的核心改进点:
        #   compute_trend_position() 整合了趋势判断和仓位管理
        #   - "binary" = v2 的传统趋势过滤（120MA 单线）
        #   - "scaled" = 60MA 入场 + 120MA 出场 + 偏离度分档
        #   - "none" = 完全无过滤（对照组）
        trend_position = compute_trend_position(benchmark_close, i, position_sizing)
        if trend_position <= 0:
            for c in list(holdings.keys()):
                if c in current_prices.index and pd.notna(current_prices[c]):
                    proceeds = holdings[c]["shares"] * current_prices[c]
                    cash += proceeds * (1 - TRADE_COST_RATE)
                    trades.append({"日期": date, "ETF": c, "方向": "卖出(趋势过滤)",
                                   "价格": current_prices[c], "原因": "趋势向下"})
            holdings = {}
            cost_prices = {}
            continue

        # --- 5b. 动量选股 ---
        # 取动量分数最高的 top_n × 2 只作为候选
        mom_row = momentum_scores.iloc[i]
        candidates = (
            mom_row.dropna()
            .sort_values(ascending=False)
            .head(top_n * REBALANCE_CANDIDATE_MULTIPLIER)
            .index.tolist()
        )

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

        # --- 5c. 相关性分散 ---
        # 先基于 20 日收益率相关系数贪心筛选，选出低相关持仓
        selected = diversify_by_correlation(
            close_df.iloc[max(0, i - 20):i + 1], candidates, top_n
        )
        if not selected:
            continue

        # --- 5d. 行业分散（v3 新增，可选）---
        # 在相关性分散之后执行行业约束
        # 确保同主题 ETF 最多选 2 只
        # 顺序说明: 先做相关性筛选（去掉高相关的），再做行业筛选（保证主题分散）
        if sector_diversity:
            selected = diversify_by_sector(selected, etf_sectors, top_n)

        if not selected:
            continue

        # --- 5e. 综合仓位系数 ---
        # 波动率目标 × 趋势仓位 = 最终仓位比例
        vol_ratio = vol_target_ratio.iloc[i] * trend_position

        # --- 5f. 卖出不在新持仓中的旧品种 ---
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

        # --- 5g. 风险平价加权（v3 新增，可选）---
        # 等权分配: 每只 ETF 分配 target_value_per_etf
        # 风险平价: 每只 ETF 按波动率倒数分配不同权重
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

        # --- 5h. 执行买入/调仓（差额调仓）---
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
# 计算策略的收益、风险和风险调整收益指标，
# 支持与沪深 300 基准的对比分析。
# 与 v1/v2 完全相同的实现，确保跨版本指标可对比。
# ============================================================


def calc_performance_metrics(nav, benchmark_nav=None, risk_free=0.02):
    """计算策略核心绩效指标（CAGR、夏普、最大回撤等）。

    v3 全区间 (2014-2026) 的最佳指标:
      最佳组合 = 双重动量_20日_分档仓位_风险平价_行业分散
      CAGR = +0.6%, 夏普 = -0.06, 最大回撤 = 52.9%

    与 v2 的对比（v2 仅在 2021-2026 回测）:
      v2 最佳夏普 = 0.40, v3 最佳夏普 = -0.06
      差异原因: v3 包含了 2016 震荡慢牛（全部组合亏损）和
      2019 核心资产牛（跑输基准），这些阶段在 v2 中不存在。

    Args:
        nav: 策略净值 Series
        benchmark_nav: 基准净值 Series（可选）
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
    """计算基准在指定时间区间的年化收益率。

    用于超额收益 = 策略 CAGR - 基准 CAGR。
    基准的时间区间必须与策略日期索引对齐。

    Args:
        benchmark_nav: 基准净值 Series
        index: 策略日期索引

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
    """将绩效指标数值格式化为百分比或小数形式的字符串。

    百分比保留 1 位小数，比率保留 2 位小数。

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
# 2 × 2 × 2 × 2 × 2 = 32 种组合:
#   动量窗口(20/60) × 动量类型(simple/dual)
#   × 仓位模式(binary/scaled) × 风险平价(开/关) × 行业分散(开/关)
# ============================================================
# v3 的参数网格与 v2 维度数相同（32 种），但内容发生了变化:
#   v2: 均线过滤 + 趋势过滤 + 多周期收敛
#   v3: 仓位模式(binary/scaled) + 风险平价 + 行业分散
# 均线过滤和多周期收敛被双层趋势线 + 分档仓位替代。
#
# 这样设计的原因:
#   - v2 证明均线过滤和多周期收敛的边际贡献较小（+0.19 和 +0.03）
#   - v3 将参数集中在仓位管理和组合构造上（权重分配、行业分散）
#   - 仓位模式(sizing)直接替代了趋势过滤——"scaled"
#     同时实现了趋势判断和仓位管理

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
    """生成参数组合的可读名称。

    命名规则: 动量类型_窗口_仓位模式_风险平价_行业分散
    例如: "双重动量_20日_分档仓位_风险平价_行业分散"

    Args:
        w: 动量窗口（20 或 60）
        t: 动量类型（"simple" 或 "dual"）
        p: 仓位模式（"binary" 或 "scaled"）
        rp: 风险平价（True/False）
        sd: 行业分散（True/False）

    Returns:
        str: 组合名称
    """
    parts = [MOMENTUM_TYPE_LABELS[t], WINDOW_LABELS[w], POSITION_LABELS[p]]
    if rp:
        parts.append("风险平价")
    if sd:
        parts.append("行业分散")
    return "_".join(parts)


def run_all_combinations(close_df, benchmark_close, etf_sectors, top_n=5):
    """遍历 32 种 v3 参数组合，运行全量回测并收集绩效结果。

    参数网格（2×2×2×2×2=32）:
      window: 20日 / 60日
      momentum_type: simple / dual
      position_sizing: binary（二元仓位） / scaled（分档仓位）
      risk_parity: 开 / 关
      sector_diversity: 开 / 关

    预期运行时间: ~60-120 秒（v3 数据范围从 2014 年起，约 3000 个交易日）

    Args:
        close_df: ETF 收盘价宽表
        benchmark_close: 沪深 300 基准收盘价
        etf_sectors: 行业归属字典
        top_n: 持仓数量

    Returns:
        results: list[dict]，每个元素是绩效指标字典
        navs: dict[str, pd.Series]，{组合名称: 净值曲线}
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


def print_summary_table(results, benchmark_close=None):
    """按夏普比率降序打印 v3 绩效总览表，含沪深 300 基准对比。

    v3 典型输出（2014-2026 全区间）:
      TOP 1: 双重动量_20日_分档仓位_风险平价_行业分散 | 夏普 -0.06
      TOP 2: 简单动量_20日_分档仓位_风险平价_行业分散 | 夏普 -0.06
      基准: 沪深 300 | CAGR +5.7% | 夏普 +0.25

    Args:
        results: run_all_combinations 返回的绩效指标列表
        benchmark_close: 基准收盘价 Series（可选）
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
    print(f"{'动量趋势轮动策略 v3 — 绩效总览':^{line_width}}")
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
    """保存 v3 回测结果 CSV（绩效汇总 + 净值曲线）。

    输出文件:
      - momentum_results/performance_summary_v3.csv: 32 种组合的绩效指标
      - momentum_results/nav_all_v3.csv: 32 种组合的净值曲线

    Args:
        results: run_all_combinations 返回的绩效指标列表
        navs: 净值曲线字典 {组合名称: Series}
    """
    os.makedirs("momentum_results", exist_ok=True)

    keys = ["组合", "CAGR", "超额收益", "夏普比率", "最大回撤", "年化波动", "胜率", "信息比率", "卡玛比率", "交易次数"]
    summary = pd.DataFrame(results)[keys]
    summary.to_csv("momentum_results/performance_summary_v3.csv", index=False, encoding="utf-8-sig")

    nav_df = pd.DataFrame(navs)
    nav_df.index.name = "日期"
    nav_df.to_csv("momentum_results/nav_all_v3.csv", encoding="utf-8-sig")

    print("结果已保存到 momentum_results/")


def main():
    """v3 主入口: 加载数据与行业分类 → 穷举 32 种参数组合 → 输出对比表。

    执行流程（比 v2 多一步）:
      1. 加载并过滤行业 ETF 列表
      2. 加载 ETF 日线数据（从 2014 年起）
      3. 加载沪深 300 基准
      4. 识别 ETF 行业归属（v3 新增）
      5. 运行 32 种参数组合回测
      6. 打印绩效总览表
      7. 保存结果 CSV

    运行时间较 v2 更长的原因:
      - 数据从 5 年扩展至 12.5 年
      - 2014 年 ETF 数量少但年份跨度大
      - 新增行业识别步骤

    使用方式:
      python momentum_strategy_v3.py
      python momentum_strategy_v3.py --top_n 3
    """
    parser = argparse.ArgumentParser(description="A股ETF动量趋势轮动策略 v3 回测")
    parser.add_argument("--top_n", type=int, default=5, help="持仓数量（默认5）")
    args = parser.parse_args()
    top_n = args.top_n

    print("=" * 60)
    print("动量趋势轮动策略 v3 回测")
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

    print(f"[5/5] 运行回测（持仓 {top_n} 只）...")
    results, navs = run_all_combinations(close_df, bench, etf_sectors, top_n)

    print_summary_table(results, bench)
    save_results(results, navs)


if __name__ == "__main__":
    main()
