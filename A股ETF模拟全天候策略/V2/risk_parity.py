"""
风险平价权重求解器
==================
实现等风险贡献（Equal Risk Contribution, ERC）优化算法。

核心数学原理：
  风险平价的目标是让每类资产对组合总风险的贡献度相等。
  对于权重向量 w，协方差矩阵 Σ，组合波动率 σ(w)=√(w'Σw)，
  第i个资产的边际风险贡献 MRC_i = ∂σ/∂w_i = (Σw)_i / σ(w)，
  风险贡献 RC_i = w_i * MRC_i。

  ERC目标：RC_1 = RC_2 = ... = RC_N
  即在约束 Σw_i = 1, w_i ≥ 0 下，最小化 Σ(RC_i - RC_j)²

实现细节：
  - 使用scipy.optimize.minimize进行SLSQP优化
  - 提供多级降级策略：如果最优解不收敛，放宽约束后重试
  - 支持基于资产大类的风险预算分配
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ASSET_CLASS_MAP,
    ETF_POOL,
    WEIGHT_MIN,
    WEIGHT_MAX,
    SINGLE_ETF_MAX,
    VOL_WINDOW,
)
import os


# ============================================================================
# 协方差矩阵估计
# ============================================================================
def estimate_covariance(
    price_df: pd.DataFrame,
    window: int = VOL_WINDOW,
) -> np.ndarray:
    """
    根据历史价格计算收益率的协方差矩阵。

    计算步骤：
      1. 对价格取对数，计算对数日收益率
      2. 用最近window个交易日的数据估计协方差
      3. 使用无偏样本协方差

    Args:
        price_df: ETF价格矩阵，index=日期，columns=ETF代码
        window: 使用多少天的历史数据（默认60个交易日）

    Returns:
        N×N 协方差矩阵
    """
    returns = np.log(price_df).diff().dropna()
    recent = returns.iloc[-window:] if len(returns) >= window else returns
    return recent.cov().values


# ============================================================================
# 风险贡献计算
# ============================================================================
def risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    计算每类资产对组合总风险的贡献度。

    公式：
      组合波动率 σ = √(w'Σw)
      边际风险贡献 MRC = (Σw) / σ
      风险贡献 RC = w * MRC

    Args:
        weights: 权重向量 [w1, w2, ..., wN]
        cov: 协方差矩阵 N×N

    Returns:
        各资产的风险贡献向量 [RC1, RC2, ..., RCN]
    """
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    if portfolio_vol < 1e-10:
        return np.zeros_like(weights)
    marginal_contrib = np.dot(cov, weights) / portfolio_vol
    return weights * marginal_contrib


# ============================================================================
# ERC目标函数
# ============================================================================
def _erc_objective(
    weights: np.ndarray,
    cov: np.ndarray,
    risk_budgets: np.ndarray,
) -> float:
    """
    ERC优化目标函数——最小化各资产风险贡献与目标预算的偏离。

    损失函数：Σ (RC_i - budget_i * σ_total)²
    即让每个资产的风险贡献尽可能接近其预算比例乘以总风险。

    当所有资产使用等风险预算时，就是标准ERC。

    Args:
        weights: 候选权重向量
        cov: 协方差矩阵
        risk_budgets: 各资产的目标风险预算比例（和为1）

    Returns:
        偏离度的平方和（标量）
    """
    rc = risk_contribution(weights, cov)
    total_risk = rc.sum()
    if total_risk < 1e-10:
        return 0.0
    # 目标：各RC占总风险的比例 = 目标预算比例
    target = risk_budgets * total_risk
    return np.sum((rc - target) ** 2)


# ============================================================================
# ERC权重求解
# ============================================================================
def solve_erc_weights(
    price_df: pd.DataFrame,
    codes: list,
    asset_class_map: Optional[Dict[str, dict]] = None,
    window: int = VOL_WINDOW,
    initial_guess: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    求解等风险贡献（ERC）最优权重。

    多级求解策略：
      1. 先尝试默认约束（WEIGHT_MIN ~ WEIGHT_MAX，且单ETF≤30%）
      2. 如果不收敛，放宽边界（WEIGHT_MIN/2 ~ WEIGHT_MAX*1.5）
      3. 仍不收敛，使用等权重回退

    资产大类支持：
      如果asset_class_map提供了risk_budget，则在大类间分配预算，
      大类内部ETF再等分。例如equity类25%预算，沪深300和中证500各12.5%。

    Args:
        price_df: ETF价格矩阵
        codes: ETF代码列表（与price_df列对应）
        asset_class_map: 资产类别映射，格式 {class: {"risk_budget": float}}
        window: 协方差估计窗口
        initial_guess: 初始权重猜测（None则用等权重）

    Returns:
        (weights, info)
        weights: 最优权重向量（满足约束）
        info: dict，包含求解状态、风险贡献等元信息
    """
    n = len(codes)
    cov = estimate_covariance(price_df, window)

    # 构建 code → ETF_POOL 条目的映射（仅取实际存在的codes）
    code_to_etf = {etf["code"]: etf for etf in ETF_POOL}
    active_etfs = [code_to_etf[c] for c in codes if c in code_to_etf]
    if len(active_etfs) != n:
        print(f"[risk_parity] 警告: 只有 {len(active_etfs)}/{len(ETF_POOL)} 只ETF有数据")

    # ================================================================
    # 1. 构建资产大类风险预算
    # ================================================================
    if asset_class_map is None:
        risk_budgets = np.ones(n) / n
    else:
        # 统计实际可用ETF的大类数量
        class_budget = {}
        for etf in active_etfs:
            ac = etf["asset_class"]
            class_budget[ac] = class_budget.get(ac, 0) + 1

        risk_budgets = np.zeros(n)
        for i, etf in enumerate(active_etfs):
            ac = etf["asset_class"]
            budget = asset_class_map.get(ac, {}).get("risk_budget", 1.0 / len(asset_class_map))
            risk_budgets[i] = budget / class_budget[ac]

    # ================================================================
    # 2. 设置权重约束
    # ================================================================
    bounds = []
    for i in range(n):
        lower = WEIGHT_MIN
        upper = min(WEIGHT_MAX, SINGLE_ETF_MAX)
        bounds.append((lower, upper))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # 初始猜测
    if initial_guess is None:
        w0 = np.ones(n) / n
        # 将初始猜测clip到边界内
        for i in range(n):
            low, high = bounds[i]
            w0[i] = np.clip(w0[i], low, high)
        w0 = w0 / w0.sum()
    else:
        w0 = initial_guess

    # ================================================================
    # 3. 执行优化（多级降级策略）
    # ================================================================
    result = None
    used_bounds = "tight"

    for attempt in range(3):
        result = minimize(
            _erc_objective,
            w0,
            args=(cov, risk_budgets),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
        )

        if result.success and _check_convergence(result.x, cov, risk_budgets):
            break

        # 降级策略：逐步放宽约束
        if attempt == 0:
            # 第一次降级：放宽上限
            bounds = [(max(WEIGHT_MIN * 0.5, 0.01), min(WEIGHT_MAX * 1.5, 0.60)) for _ in range(n)]
            used_bounds = "loose"
        elif attempt == 1:
            # 第二次降级：完全放开到 [0, 1]
            bounds = [(0.0, 1.0) for _ in range(n)]
            used_bounds = "free"
            w0 = np.ones(n) / n  # 重置初始猜测

    # ================================================================
    # 4. 最终回退
    # ================================================================
    if result is None or not result.success:
        print(f"[risk_parity] 优化未收敛, 使用等权重回退")
        final_weights = np.ones(n) / n
        solved = False
    else:
        final_weights = result.x
        solved = True

    # ================================================================
    # 5. 计算风险贡献和元信息
    # ================================================================
    rc = risk_contribution(final_weights, cov)
    port_vol = np.sqrt(np.dot(final_weights.T, np.dot(cov, final_weights)))

    info = {
        "success": solved,
        "bounds_level": used_bounds,
        "portfolio_volatility": port_vol,
        "risk_contributions": rc,
        "risk_contribution_pct": rc / (rc.sum() + 1e-10),
        "target_budgets": risk_budgets,
        "optimizer_result": result,
    }

    return final_weights, info


def _check_convergence(
    weights: np.ndarray,
    cov: np.ndarray,
    risk_budgets: np.ndarray,
    tolerance: float = 0.05,
) -> bool:
    """
    检查ERC优化是否收敛：
    各资产的实际风险贡献比例与目标预算的偏差是否在容忍范围内。

    Args:
        weights: 求解出的权重
        cov: 协方差矩阵
        risk_budgets: 目标风险预算
        tolerance: 容忍偏差（默认5%）

    Returns:
        True=已收敛, False=偏差过大
    """
    rc = risk_contribution(weights, cov)
    rc_total = rc.sum()
    if rc_total < 1e-10:
        return False
    rc_pct = rc / rc_total
    max_dev = np.max(np.abs(rc_pct - risk_budgets))
    return max_dev < tolerance


# ============================================================================
# HRP 接口
# ============================================================================
def solve_hrp_from_cov(
    cov: np.ndarray,
    codes: list,
) -> Tuple[np.ndarray, dict]:
    """
    使用预计算协方差矩阵求解 HRP 权重。

    直接接收协方差矩阵（通常来自 HMM 状态或 Ledoit-Wolf 收缩），
    不依赖 price_df，适用于 HMM 状态感知场景。

    Args:
        cov: N×N 协方差矩阵
        codes: ETF 代码列表

    Returns:
        (weights, info)
    """
    from hrp import get_hrp_weights
    return get_hrp_weights(cov=cov, codes=codes)


# ============================================================================
# 统一入口
# ============================================================================
def get_target_weights(
    price_df: Optional[pd.DataFrame] = None,
    codes: Optional[list] = None,
    cov: Optional[np.ndarray] = None,
    last_weights: Optional[np.ndarray] = None,
    mode: str = "hrp",
) -> Tuple[np.ndarray, dict]:
    """
    对外暴露的权重求解统一接口。

    支持两种模式：
      - "hrp": 分层风险平价（默认），基于层次聚类 + 递归二分法
      - "erc": 等风险贡献，基于 SLSQP 优化（旧版回退）

    HRP 模式下：
      如果提供 cov 参数，直接使用该协方差矩阵（由 HMM 状态提供）；
      否则从 price_df 计算 Ledoit-Wolf 收缩协方差。

    ERC 模式下：
      使用 price_df 计算样本协方差，执行 SLSQP 优化。
      使用上一次权重作为初始猜测，有助于提高收敛速度。

    Args:
        price_df: ETF 价格矩阵
        codes: ETF 代码列表
        cov: 预计算协方差矩阵（HRP 模式优先使用）
        last_weights: 上次权重（仅 ERC 模式使用）
        mode: "hrp" 或 "erc"

    Returns:
        (weights, info)
    """
    if codes is None:
        codes = [etf["code"] for etf in ETF_POOL]

    if mode == "hrp":
        # ─── HRP 模式 ───
        if cov is None and price_df is not None:
            from hrp import ledoit_wolf_cov
            returns = np.log(price_df).diff().dropna().values
            cov, delta = ledoit_wolf_cov(returns)
        elif cov is None:
            cov = estimate_covariance(price_df) if price_df is not None else np.eye(len(codes))

        weights, info = solve_hrp_from_cov(cov=cov, codes=codes)
        info["mode"] = "hrp"
        return weights, info

    else:
        # ─── ERC 模式（旧版回退）───
        return solve_erc_weights(
            price_df=price_df,
            codes=codes,
            asset_class_map=ASSET_CLASS_MAP,
            initial_guess=last_weights,
        )
