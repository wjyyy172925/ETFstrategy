import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calc_cov_matrix(returns: pd.DataFrame, window: int = 60) -> np.ndarray:
    return returns.tail(window).cov().values * 252


def _risk_contribution(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    portfolio_var = weights @ cov @ weights
    marginal_contrib = cov @ weights
    risk_contrib = weights * marginal_contrib / np.sqrt(max(portfolio_var, 1e-16))
    return risk_contrib


def _erc_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    rc = _risk_contribution(weights, cov)
    target = rc.mean()
    return float(np.sum((rc - target) ** 2))


def solve_erc_weights(returns: pd.DataFrame,
                      w_min: float = 0.05,
                      w_max: float = 0.40) -> np.ndarray:
    n = returns.shape[1]
    cov = calc_cov_matrix(returns)
    bounds = [(w_min, w_max)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    x0 = np.array([1.0 / n] * n)
    result = minimize(_erc_objective, x0, args=(cov,),
                      method="SLSQP", bounds=bounds, constraints=constraints)
    if not result.success:
        bounds_loose = [(0.02, 0.60)] * n
        result = minimize(_erc_objective, x0, args=(cov,),
                          method="SLSQP", bounds=bounds_loose, constraints=constraints)
    if not result.success:
        raise RuntimeError("ERC optimization failed to converge")
    return result.x / result.x.sum()


def calc_etf_weights(return_df: pd.DataFrame, etf_pool: list[dict],
                     w_min: float = 0.05, w_max: float = 0.40) -> pd.Series:
    asset_classes = list(dict.fromkeys(e["asset_class"] for e in etf_pool))
    class_returns = pd.DataFrame()
    for ac in asset_classes:
        codes = [e["code"] for e in etf_pool if e["asset_class"] == ac]
        cols = [c for c in codes if c in return_df.columns]
        if not cols:
            continue
        class_returns[ac] = return_df[cols].mean(axis=1)
    if class_returns.empty:
        raise ValueError("No valid asset classes found in return_df columns")
    class_weights = solve_erc_weights(class_returns, w_min, w_max)
    etf_weights = pd.Series(0.0, index=return_df.columns)
    for ac, cw in zip(class_returns.columns, class_weights):
        codes = [e["code"] for e in etf_pool if e["asset_class"] == ac]
        cols = [c for c in codes if c in return_df.columns]
        for col in cols:
            etf_weights[col] = cw / len(cols)
    return etf_weights
