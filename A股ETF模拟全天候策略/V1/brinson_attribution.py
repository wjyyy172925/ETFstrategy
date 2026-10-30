"""
Brinson归因分析
==============
基于Carino多期连接方法，对风险平价策略进行Brinson绩效归因。

归因框架：
  - 基准：经典全天候（30%权益 / 55%债券 / 7.5%黄金 / 7.5%商品）
  - 资产大类（sector）：equity, long_bond, mid_bond, gold, commodity
  - 归因周期：月频
  - 多期连接：Carino (1999) 对数连接系数

Brinson-Fachler分解：
  total_excess = allocation + selection + interaction

  allocation  = Σ (w_p_i - w_b_i) * (r_b_i - r_b)
  selection   = Σ  w_b_i       * (r_p_i - r_b_i)
  interaction = Σ (w_p_i - w_b_i) * (r_p_i - r_b_i)

Carino多期连接：
  每期计算 k_t = ln(1+R_p_t) - ln(1+R_b_t) / (R_p_t - R_b_t)
  权重 K_t = k_t / Σ k_t
  多期效应 = Σ (K_t * 单期效应)

数据来源：
  - weights.csv: 每日组合权重
  - nav.csv: 每日组合净值
  - data/etf/{code}.csv: ETF每日净值
"""

import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ETF_CODES = ["510300", "510500", "511260", "511010", "518880", "159985"]

ETF_NAMES = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "511260": "十年国债ETF",
    "511010": "国债ETF",
    "518880": "黄金ETF",
    "159985": "豆粕ETF",
}

ASSET_CLASS = {
    "510300": "equity",
    "510500": "equity",
    "511260": "long_bond",
    "511010": "mid_bond",
    "518880": "gold",
    "159985": "commodity",
}

CLASS_LABEL = {
    "equity": "权益",
    "long_bond": "长久期利率债",
    "mid_bond": "中短债+信用债",
    "gold": "黄金",
    "commodity": "商品",
}

CLASS_ORDER = ["equity", "long_bond", "mid_bond", "gold", "commodity"]

BENCHMARK_CLASS_WEIGHTS = {
    "equity": 0.30,
    "long_bond": 0.275,
    "mid_bond": 0.275,
    "gold": 0.075,
    "commodity": 0.075,
}

CSV_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "etf"
)
CSV_DIR = os.path.normpath(CSV_DIR)
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res", "run_001")

INITIAL_CAPITAL = 100_000_000       # 初始资金（与回测一致）
TOTAL_COST_PER_SIDE = 0.0008        # 万8单边交易成本（与回测一致）
MIN_TRADE_AMOUNT = 5000             # 最小交易金额（与回测一致）


def load_weights(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = df[ETF_CODES]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def load_etf_prices(codes: list) -> pd.DataFrame:
    prices = {}
    for code in codes:
        csv_path = os.path.join(CSV_DIR, f"{code}.csv")
        if not os.path.exists(csv_path):
            print(f"[brinson] 警告: {code}.csv 不存在于 {csv_path}")
            continue
        raw = pd.read_csv(csv_path, encoding="utf-8-sig")
        cols_lower = {c.strip().lower(): c for c in raw.columns}
        date_col = cols_lower.get("date", cols_lower.get("日期", "date"))
        close_col = cols_lower.get("close", cols_lower.get("收盘", "close"))
        df = raw[[date_col, close_col]].copy()
        df.columns = ["date", code]
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
        df = df.set_index("date")
        prices[code] = df[code]
    price_df = pd.DataFrame(prices).sort_index()
    return price_df


def aggregate_weights_to_classes(etf_weights: pd.DataFrame) -> pd.DataFrame:
    class_weights = pd.DataFrame(index=etf_weights.index)
    for cls in CLASS_ORDER:
        etfs_in_class = [c for c in ETF_CODES if ASSET_CLASS[c] == cls]
        class_weights[cls] = etf_weights[etfs_in_class].sum(axis=1)
    return class_weights


def compute_returns_from_prices(price_df: pd.DataFrame) -> pd.DataFrame:
    return price_df.pct_change().dropna()


def simulate_benchmark_daily(
    price_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    return_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    模拟经典全天候基准的每日持仓，口径与ERC回测完全一致：

      - 月初再平衡至目标权重（30/27.5/27.5/7.5/7.5）
      - 100股整手取整，余款变现金
      - 每次换仓收取 万8 单边成本
      - 现金不产生收益

    Returns:
        DataFrame, index=日期, columns=ETF代码+nav
        其中 ETF 权重为 holdings_value / portfolio_value（含现金）
    """
    codes = ETF_CODES
    n_assets = len(codes)

    # Include the first weight date because it is the prior close needed for
    # the first return date.
    common_dates = price_df.index.intersection(weight_df.index.union(return_df.index)).sort_values()
    price_df = price_df.loc[common_dates[0]:common_dates[-1], codes]
    price_df = price_df.astype(float)
    prices = price_df.values
    dates = price_df.index
    n_days = len(dates)

    target_etf_weights = np.zeros(n_assets, dtype=float)
    for i, code in enumerate(codes):
        cls = ASSET_CLASS[code]
        etfs_in_class = [c for c in codes if ASSET_CLASS[c] == cls]
        target_etf_weights[i] = float(BENCHMARK_CLASS_WEIGHTS[cls]) / len(etfs_in_class)

    cash = float(INITIAL_CAPITAL)
    shares = np.zeros(n_assets, dtype=float)
    for i in range(n_assets):
        target_value = float(INITIAL_CAPITAL) * target_etf_weights[i]
        raw_shares = target_value / float(prices[0, i])
        shares[i] = int(raw_shares / 100) * 100
        cash -= shares[i] * float(prices[0, i])
    cash -= float(INITIAL_CAPITAL) * TOTAL_COST_PER_SIDE

    equity_curve = np.zeros(n_days, dtype=float)
    weight_history = np.zeros((n_days, n_assets), dtype=float)
    equity_curve[0] = np.sum(shares * prices[0]) + cash
    if equity_curve[0] > 0:
        weight_history[0, :] = shares * prices[0] / equity_curve[0]
    else:
        weight_history[0, :] = target_etf_weights

    common_dates = price_df.index.intersection(weight_df.index.union(return_df.index)).sort_values()
    trade_dates = pd.Series(common_dates)
    monthly_groups = trade_dates.groupby([trade_dates.dt.year, trade_dates.dt.month])
    rebalance_set = set()
    for _, days_in_month in monthly_groups:
        rebalance_set.add(days_in_month.iloc[0])

    for t in range(1, n_days):
        date = dates[t]
        etf_values = shares * prices[t]
        holdings_value = np.sum(etf_values)
        portfolio_value = holdings_value + cash

        equity_curve[t] = portfolio_value
        if portfolio_value > 0:
            weight_history[t, :] = etf_values / portfolio_value

        if date in rebalance_set and portfolio_value > 0:
            target_values = portfolio_value * target_etf_weights
            turnover = np.sum(np.abs(target_values - etf_values))
            cost = turnover * TOTAL_COST_PER_SIDE

            available = portfolio_value - cost
            new_shares = np.zeros(n_assets)
            new_cash = available
            for i in range(n_assets):
                tv = available * target_etf_weights[i]
                if tv >= MIN_TRADE_AMOUNT:
                    raw = tv / prices[t, i]
                    new_shares[i] = int(raw / 100) * 100
                    new_cash -= new_shares[i] * prices[t, i]

            shares = new_shares
            cash = new_cash

    result = pd.DataFrame(weight_history, index=dates, columns=codes)
    result["nav"] = equity_curve
    return result


def build_monthly_data(
    weight_df: pd.DataFrame,
    return_df: pd.DataFrame,
    benchmark_weight_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    构建月度Brinson归因所需数据。
    每月取月初组合权重，月度ETF收益率，聚合到资产大类。

    Args:
        weight_df: ERC组合每日ETF权重
        return_df: ETF每日收益率
        benchmark_weight_df: 基准每日ETF权重（模拟口径），若为None则使用理论权重
    """
    weight_df = weight_df.sort_index()
    return_df = return_df.sort_index()
    if weight_df.index.has_duplicates:
        raise ValueError("weight_df contains duplicate dates")
    if return_df.index.has_duplicates:
        raise ValueError("return_df contains duplicate dates")

    # return_df[t] is close[t] / close[t-1] - 1.  Match each return date to
    # the latest strictly earlier weight date (never the same-day close).
    def align_prior_weights(source: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
        source = source.sort_index()
        left = pd.DataFrame({"date": dates})
        right = source.reset_index().rename(columns={source.index.name or "index": "weight_date"})
        merged = pd.merge_asof(
            left.sort_values("date"), right.sort_values("weight_date"),
            left_on="date", right_on="weight_date", direction="backward",
            allow_exact_matches=False,
        ).set_index("date")
        if merged.isna().any().any():
            raise ValueError("Missing prior-day portfolio weights for one or more return dates")
        return merged.drop(columns=["weight_date"])

    common_dates = weight_df.index.intersection(return_df.index).sort_values()
    return_df = return_df.loc[common_dates]
    weight_df = align_prior_weights(weight_df[ETF_CODES], common_dates)

    if benchmark_weight_df is not None:
        benchmark_weight_df = benchmark_weight_df.sort_index()
        if benchmark_weight_df.index.has_duplicates:
            raise ValueError("benchmark_weight_df contains duplicate dates")
        bm_common = benchmark_weight_df.index.intersection(common_dates).sort_values()
        benchmark_weight_df = align_prior_weights(benchmark_weight_df[ETF_CODES], bm_common)
        common_dates = bm_common
        weight_df = weight_df.loc[common_dates]
        return_df = return_df.loc[common_dates]

    trade_dates = pd.Series(common_dates)
    monthly_groups = trade_dates.groupby([trade_dates.dt.year, trade_dates.dt.month])

    rows = []
    for (year, month), days_in_month in monthly_groups:
        first_day = days_in_month.iloc[0]
        last_day = days_in_month.iloc[-1]

        if first_day == last_day:
            continue

        w_etf = weight_df.loc[first_day]
        r_etf = (1 + return_df.loc[first_day:last_day]).prod() - 1

        w_class = {}
        r_class_p = {}
        r_class_b = {}
        w_bm_class = {}
        for cls in CLASS_ORDER:
            etfs = [c for c in ETF_CODES if ASSET_CLASS[c] == cls]
            w_cls = sum(w_etf.get(c, 0) for c in etfs)
            w_class[cls] = w_cls

            if w_cls > 0:
                r_p = sum(w_etf.get(c, 0) * r_etf.get(c, 0) for c in etfs) / w_cls
            else:
                r_p = sum(r_etf.get(c, 0) for c in etfs) / len(etfs) if etfs else 0
            r_class_p[cls] = r_p

            n_bm = len(etfs)
            r_class_b[cls] = sum(r_etf.get(c, 0) for c in etfs) / n_bm if n_bm > 0 else 0

            if benchmark_weight_df is not None:
                w_bm_cls = sum(benchmark_weight_df.loc[first_day].get(c, 0) for c in etfs)
                w_bm_class[cls] = w_bm_cls
            else:
                w_bm_class[cls] = BENCHMARK_CLASS_WEIGHTS[cls]

        r_p_total = sum(w_class[cls] * r_class_p[cls] for cls in CLASS_ORDER)
        if benchmark_weight_df is not None:
            r_b_total = sum(w_bm_class[cls] * r_class_b[cls] for cls in CLASS_ORDER)
        else:
            r_b_total = sum(BENCHMARK_CLASS_WEIGHTS[cls] * r_class_b[cls] for cls in CLASS_ORDER)

        rows.append({
            "year": year,
            "month": month,
            "first_date": first_day,
            "last_date": last_day,
            "r_p": r_p_total,
            "r_b": r_b_total,
            **{f"w_p_{cls}": w_class[cls] for cls in CLASS_ORDER},
            **{f"w_b_{cls}": w_bm_class[cls] for cls in CLASS_ORDER},
            **{f"r_p_{cls}": r_class_p[cls] for cls in CLASS_ORDER},
            **{f"r_b_{cls}": r_class_b[cls] for cls in CLASS_ORDER},
        })

    monthly = pd.DataFrame(rows)
    monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    return monthly


def compute_single_period_brinson(row) -> Dict[str, float]:
    r_p = row["r_p"]
    r_b = row["r_b"]
    excess = r_p - r_b

    allocation = 0.0
    selection = 0.0
    interaction = 0.0

    alloc_by_class = {}
    select_by_class = {}
    interact_by_class = {}

    for cls in CLASS_ORDER:
        w_p = row[f"w_p_{cls}"]
        w_b = row[f"w_b_{cls}"]
        r_p_i = row[f"r_p_{cls}"]
        r_b_i = row[f"r_b_{cls}"]

        a = (w_p - w_b) * (r_b_i - r_b)
        s = w_b * (r_p_i - r_b_i)
        i = (w_p - w_b) * (r_p_i - r_b_i)

        allocation += a
        selection += s
        interaction += i

        alloc_by_class[cls] = a
        select_by_class[cls] = s
        interact_by_class[cls] = i

    result = {
        "r_p": r_p,
        "r_b": r_b,
        "excess": excess,
        "allocation": allocation,
        "selection": selection,
        "interaction": interaction,
        "alloc_sum_check": allocation + selection + interaction,
    }
    for cls in CLASS_ORDER:
        result[f"alloc_{cls}"] = alloc_by_class[cls]
        result[f"select_{cls}"] = select_by_class[cls]
        result[f"interact_{cls}"] = interact_by_class[cls]

    return result


def carino_linking(monthly_df: pd.DataFrame, single_periods: pd.DataFrame) -> dict:
    """
    Carino (1999) 多期连接方法。

    核心思想：
      1. 用对数连接系数 k_t 将每期效应从算术空间映射到对数空间
         k_t = [ln(1+r_p_t) - ln(1+r_b_t)] / (r_p_t - r_b_t)
         满足: ln(1+r_p_t) - ln(1+r_b_t) = k_t * (r_p_t - r_b_t)

      2. 对数空间效应直接可加：
         ln(1+R_p) - ln(1+R_b) = Σ k_t * (r_p_t - r_b_t) = Σ k_t * (A_t + S_t + I_t)

      3. 再缩放回算术空间，保证链接效应精确等于算术超额收益：
         E_arithmetic = E_log * (R_p - R_b) / [ln(1+R_p) - ln(1+R_b)]
    """
    n = len(monthly_df)
    k = np.zeros(n)

    for t in range(n):
        r_p = monthly_df["r_p"].iloc[t]
        r_b = monthly_df["r_b"].iloc[t]
        if abs(r_p - r_b) < 1e-12:
            k[t] = 1.0 / (1.0 + r_p)
        else:
            k[t] = (np.log(1 + r_p) - np.log(1 + r_b)) / (r_p - r_b)

    R_p_total = (1 + monthly_df["r_p"]).prod() - 1
    R_b_total = (1 + monthly_df["r_b"]).prod() - 1
    total_excess = R_p_total - R_b_total

    log_R_p = np.log(1 + R_p_total)
    log_R_b = np.log(1 + R_b_total)
    log_excess = log_R_p - log_R_b

    scale = total_excess / log_excess if abs(log_excess) > 1e-12 else 1.0

    linked = {}
    for col in ["allocation", "selection", "interaction"]:
        linked[col] = np.sum(k * single_periods[col].values) * scale

    linked["excess"] = total_excess
    linked["excess_from_linking"] = linked["allocation"] + linked["selection"] + linked["interaction"]
    linked["R_p_cum"] = R_p_total
    linked["R_b_cum"] = R_b_total

    for cls in CLASS_ORDER:
        linked[f"alloc_{cls}"] = np.sum(k * single_periods[f"alloc_{cls}"].values) * scale
        linked[f"select_{cls}"] = np.sum(k * single_periods[f"select_{cls}"].values) * scale
        linked[f"interact_{cls}"] = np.sum(k * single_periods[f"interact_{cls}"].values) * scale

    linked["linking_error"] = abs(linked["excess"] - linked["excess_from_linking"])
    return linked


def run_brinson(
    weights_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    if weights_path is None:
        weights_path = os.path.join(RES_DIR, "weights.csv")
    # nav_path is no longer used; portfolio returns are computed from weights and ETF returns

    print("=" * 70)
    print("Brinson 归因分析 (Carino 多期连接)")
    print("=" * 70)

    print("\n[1/5] 加载数据...")
    weight_df = load_weights(weights_path)
    price_df = load_etf_prices(ETF_CODES)
    print(f"  权重数据: {weight_df.index[0].date()} ~ {weight_df.index[-1].date()}, {len(weight_df)}天")
    print(f"  ETF价格:  {price_df.index[0].date()} ~ {price_df.index[-1].date()}, {price_df.columns.tolist()}")

    print("\n[2/5] 计算ETF日收益率...")
    return_df = compute_returns_from_prices(price_df)

    common_start = max(weight_df.index[0], return_df.index[0])
    common_end = min(weight_df.index[-1], return_df.index[-1])
    # Keep the prior trading day's weight; build_monthly_data shifts it onto
    # the first return date to prevent same-day look-ahead.
    weight_df = weight_df.loc[:common_end]
    return_df = return_df.loc[common_start:common_end]
    print(f"  对齐后范围: {common_start.date()} ~ {common_end.date()}")

    print("\n[3/5] 模拟基准组合（含交易成本、整手取整、现金拖累）...")
    benchmark_weight_df = simulate_benchmark_daily(price_df, weight_df, return_df)
    bm_nav_end = benchmark_weight_df["nav"].iloc[-1]
    bm_total_return = bm_nav_end / INITIAL_CAPITAL - 1
    print(f"  基准累计收益(模拟口径): {bm_total_return:.4%}")

    print("\n[4/5] 构建月度归因数据...")
    monthly_df = build_monthly_data(weight_df, return_df, benchmark_weight_df)
    print(f"  月度周期数: {len(monthly_df)}")

    print("\n[5/5] 计算单期Brinson归因 + Carino连接...")
    single_results = monthly_df.apply(compute_single_period_brinson, axis=1, result_type="expand")
    linked = carino_linking(monthly_df, single_results)

    n_years = len(monthly_df) / 12.0

    print(f"  回测年数: {n_years:.2f}")
    print(f"  累计组合收益: {linked['R_p_cum']:.4%}")
    print(f"  累计基准收益: {linked['R_b_cum']:.4%}")
    print(f"  累计超额收益: {linked['excess']:.4%}")
    print(f"  连接误差:     {linked['linking_error']:.10f}")

    linked["n_years"] = n_years
    return monthly_df, single_results, linked


def print_report(monthly_df: pd.DataFrame, single_results: pd.DataFrame, linked: dict):
    n_years = len(monthly_df) / 12.0

    print("\n" + "=" * 70)
    print("Brinson 归因结果 (Carino 多期连接)")
    print("=" * 70)

    print(f"\n回测期间: {monthly_df['first_date'].iloc[0].date()} ~ {monthly_df['last_date'].iloc[-1].date()}")
    print(f"月度周期: {len(monthly_df)} 个月 ({n_years:.1f} 年)")
    print(f"基准策略: 经典全天候 (30%权益 / 27.5%长久期债 / 27.5%中短债 / 7.5%黄金 / 7.5%商品)")

    print(f"\n{'─' * 70}")
    print(f"{'累计收益汇总':^70}")
    print(f"{'─' * 70}")
    print(f"  组合累计收益:  {linked['R_p_cum']:>10.4%}")
    print(f"  基准累计收益:  {linked['R_b_cum']:>10.4%}")
    print(f"  超额收益:      {linked['excess']:>10.4%}")
    print(f"  连接误差:      {linked['linking_error']:>10.8f}")

    print(f"\n{'─' * 70}")
    print(f"{'归因分解 (累计)':^70}")
    print(f"{'─' * 70}")
    print(f"  {'效应':<16}{'累计贡献':>12}{'占比':>10}")
    print(f"  {'─' * 40}")
    alloc_val = linked["allocation"]
    select_val = linked["selection"]
    interact_val = linked["interaction"]
    total = alloc_val + select_val + interact_val
    for name, val in [("配置效应", alloc_val), ("选股效应", select_val), ("交互效应", interact_val)]:
        pct = val / total * 100 if abs(total) > 1e-10 else 0
        print(f"  {name:<16}{val:>+11.4%}{pct:>9.1f}%")
    print(f"  {'─' * 40}")
    print(f"  {'合计':<16}{total:>+11.4%}{100:>9.1f}%")
    print(f"  组合超额(累计): {linked['excess']:+.4%}")

    print(f"\n{'─' * 70}")
    print(f"{'归因分解 (年化)':^70}")
    print(f"{'─' * 70}")
    print(f"  {'效应':<16}{'年化贡献':>12}{'占比':>10}")
    print(f"  {'─' * 40}")
    alloc_ann = alloc_val / n_years
    select_ann = select_val / n_years
    interact_ann = interact_val / n_years
    excess_ann = linked["excess"] / n_years
    total_ann = alloc_ann + select_ann + interact_ann
    for name, val in [("配置效应", alloc_ann), ("选股效应", select_ann), ("交互效应", interact_ann)]:
        pct = val / total_ann * 100 if abs(total_ann) > 1e-10 else 0
        print(f"  {name:<16}{val:>+11.4%}{pct:>9.1f}%")
    print(f"  {'─' * 40}")
    print(f"  {'合计':<16}{total_ann:>+11.4%}{100:>9.1f}%")
    print(f"  组合超额(年化): {excess_ann:+.4%}")

    print(f"\n{'─' * 70}")
    print(f"{'配置效应分解 (年化)':^70}")
    print(f"{'─' * 70}")
    print(f"  {'资产大类':<16}{'年化配置效应':>14}")
    print(f"  {'─' * 40}")
    for cls in CLASS_ORDER:
        val = linked.get(f"alloc_{cls}", 0) / n_years
        print(f"  {CLASS_LABEL[cls]:<16}{val:>+13.4%}")

    print(f"\n{'─' * 70}")
    print(f"{'选股效应分解 (年化)':^70}")
    print(f"{'─' * 70}")
    print(f"  {'资产大类':<16}{'年化选股效应':>14}")
    print(f"  {'─' * 40}")
    for cls in CLASS_ORDER:
        val = linked.get(f"select_{cls}", 0) / n_years
        print(f"  {CLASS_LABEL[cls]:<16}{val:>+13.4%}")

    print(f"\n{'─' * 70}")
    print(f"{'交互效应分解 (年化)':^70}")
    print(f"{'─' * 70}")
    print(f"  {'资产大类':<16}{'年化交互效应':>14}")
    print(f"  {'─' * 40}")
    for cls in CLASS_ORDER:
        val = linked.get(f"interact_{cls}", 0) / n_years
        print(f"  {CLASS_LABEL[cls]:<16}{val:>+13.4%}")

    print(f"\n{'─' * 70}")
    print(f"{'历年超额收益':^70}")
    print(f"{'─' * 70}")
    yearly = monthly_df.groupby("year").agg(
        r_p=("r_p", lambda x: (1 + x).prod() - 1),
        r_b=("r_b", lambda x: (1 + x).prod() - 1),
    )
    yearly["excess"] = yearly["r_p"] - yearly["r_b"]
    print(f"  {'年份':<8}{'组合收益':>10}{'基准收益':>10}{'超额':>10}")
    print(f"  {'─' * 40}")
    for yr, row in yearly.iterrows():
        print(f"  {int(yr):<8}{row['r_p']:>9.2%}  {row['r_b']:>9.2%}  {row['excess']:>+9.2%}")

    print("\n" + "=" * 70)


def save_results(
    monthly_df: pd.DataFrame,
    single_results: pd.DataFrame,
    linked: dict,
    output_dir: str = None,
):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res", "run_001")

    n_years = len(monthly_df) / 12.0

    out_cols = [
        "period", "first_date", "last_date", "r_p", "r_b",
        "allocation", "selection", "interaction", "excess",
    ]
    out_cols += [f"w_p_{cls}" for cls in CLASS_ORDER]
    out_cols += [f"w_b_{cls}" for cls in CLASS_ORDER]
    out_cols += [f"r_p_{cls}" for cls in CLASS_ORDER]
    out_cols += [f"r_b_{cls}" for cls in CLASS_ORDER]

    detail = monthly_df.copy()
    for col in ["allocation", "selection", "interaction", "excess", "alloc_sum_check"]:
        detail[col] = single_results[col].values
    for cls in CLASS_ORDER:
        for prefix in ["alloc", "select", "interact"]:
            detail[f"{prefix}_{cls}"] = single_results[f"{prefix}_{cls}"].values

    detail_path = os.path.join(output_dir, "brinson_monthly.csv")
    detail[out_cols].to_csv(detail_path, index=False, encoding="utf-8-sig", float_format="%.10f")
    print(f"\n月度归因明细已保存: {detail_path}")

    alloc_val = linked["allocation"]
    select_val = linked["selection"]
    interact_val = linked["interaction"]
    total = alloc_val + select_val + interact_val

    summary_rows = []

    def add_section(title):
        summary_rows.append({"类别": f"【{title}】", "指标": "", "数值（累计）": "", "数值（年化）": ""})

    def add_row(cat, name, cum, ann=None):
        if ann is None:
            ann = cum / n_years
        summary_rows.append({"类别": cat, "指标": name, "数值（累计）": f"{cum:.6f}", "数值（年化）": f"{ann:.6f}"})

    add_section("收益概况")
    add_row("收益", "组合收益", linked["R_p_cum"])
    add_row("收益", "基准收益", linked["R_b_cum"])
    add_row("收益", "超额收益", linked["excess"])
    add_row("收益", "连接误差", linked["linking_error"], linked["linking_error"])

    add_section("Brinson归因 (Arithmetic)")
    add_row("归因", "配置效应", alloc_val)
    add_row("归因", "选股效应", select_val)
    add_row("归因", "交互效应", interact_val)
    add_row("归因", "合计", total)

    add_section("配置效应分解")
    for cls in CLASS_ORDER:
        val = linked.get(f"alloc_{cls}", 0)
        add_row("配置", CLASS_LABEL[cls], val)

    add_section("选股效应分解")
    for cls in CLASS_ORDER:
        val = linked.get(f"select_{cls}", 0)
        add_row("选股", CLASS_LABEL[cls], val)

    add_section("交互效应分解")
    for cls in CLASS_ORDER:
        val = linked.get(f"interact_{cls}", 0)
        add_row("交互", CLASS_LABEL[cls], val)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "brinson_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"归因汇总已保存: {summary_path}")


if __name__ == "__main__":
    monthly_df, single_results, linked = run_brinson()
    print_report(monthly_df, single_results, linked)
    save_results(monthly_df, single_results, linked)
