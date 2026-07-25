"""
HRP 分层风险平价 + Ledoit-Wolf 收缩协方差
=========================================

分层风险平价（Hierarchical Risk Parity）
  Lopez de Prado (2016) 提出的基于层次聚类的资产配置方法。
  
  核心步骤：
    1. 计算资产间相关系数 → 距离矩阵
    2. 层次聚类（单链法），构建聚类树
    3. 矩阵序列化（Optimal Leaf Ordering），重排协方差矩阵
    4. 递归二分法分配权重

  优点：
    - 不需要优化器，不存在收敛问题
    - 高相关资产自动聚簇，簇内按逆方差分配
    - 权重稳定，不会出现极端配置或震荡

Ledoit-Wolf 收缩协方差
  Ledoit & Wolf (2004) 的收缩估计量。
  
  公式：Σ_lw = δ * F + (1-δ) * S
    - S = 样本协方差
    - F = 常量相关目标矩阵
    - δ = 最优收缩强度

  优点：
    - 极端行情时 δ 自适应增大，抑制噪声
    - 比滚窗样本协方差更稳定，降低策略换手率
"""

import numpy as np
from typing import List, Optional, Tuple


# ============================================================================
# Ledoit-Wolf 收缩协方差估计
# ============================================================================
def ledoit_wolf_cov(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ledoit-Wolf 收缩协方差估计。

    目标矩阵 F：使用常量相关系数模型
        F_ii = S_ii（方差与样本一致）
        F_ij = r_bar * sqrt(S_ii * S_jj)（相关系数取平均）

    收缩强度 δ 由数据自动确定，在 [0, 1] 内。

    Args:
        returns: T×N 收益率矩阵（每列一个资产）

    Returns:
        (lw_cov, shrinkage)
        lw_cov: N×N 收缩协方差矩阵
        shrinkage: 收缩强度 δ
    """
    T, N = returns.shape

    if T < 2:
        return np.cov(returns, rowvar=False), 0.0

    # 样本协方差（无偏）
    demeaned = returns - returns.mean(axis=0)
    S = (demeaned.T @ demeaned) / (T - 1)

    # 方差和标准差
    var = np.diag(S)
    std = np.sqrt(var)

    # 样本相关系数矩阵
    std_outer = np.outer(std, std)
    R = S / std_outer

    # 平均相关系数（取上三角非对角线的均值）
    r_bar = (np.sum(R) - N) / (N * (N - 1))

    # 目标矩阵 F（常量相关系数模型）
    F = r_bar * std_outer
    np.fill_diagonal(F, var)

    # ---- 计算最优收缩强度 δ ----

    # pi = sum_{i,j} Var(S_ij) 的估计
    pi_sum = 0.0
    for i in range(N):
        for j in range(N):
            y = demeaned[:, i] * demeaned[:, j]
            y_bar = y.mean()
            var_ij = np.sum((y - y_bar) ** 2) / T
            pi_sum += var_ij

    # rho = sum_{i,j} Cov(S_ij, F_ij) 的估计
    rho_sum = 0.0
    for i in range(N):
        for j in range(N):
            y = demeaned[:, i] * demeaned[:, j]
            y_bar = y.mean()
            var_ij = np.sum((y - y_bar) ** 2) / T
            theta_ij = F[i, j]
            if abs(theta_ij) > 1e-12:
                rho_sum += var_ij * theta_ij

    # gamma = ||F - S||²_F（Frobenius 范数平方）
    gamma = np.sum((F - S) ** 2)

    # 收缩强度
    if gamma < 1e-10:
        delta = 0.0
    else:
        delta = max(0.0, min(1.0, (pi_sum - rho_sum) / gamma))

    lw_cov = delta * F + (1 - delta) * S
    return lw_cov, delta


# ============================================================================
# HRP：层次聚类 + 递归二分
# ============================================================================
def _corr_to_dist(corr: np.ndarray) -> np.ndarray:
    """相关系数矩阵 → 距离矩阵。"""
    return np.sqrt(2.0 * (1.0 - np.clip(corr, -1.0, 1.0)))


def _get_quasi_diag(link: np.ndarray) -> List[int]:
    """
    矩阵序列化（Quasi-Diagonalization）。

    对层次聚类树进行后序遍历，按合并顺序展开叶子节点，
    使相近的簇在序列中相邻。

    聚类树的节点编号规则（scipy）：
      - 原始资产: 0 ~ N-1
      - 合并节点: N + i（第 i 次合并产生的新节点）
      - 根节点: N + (N-2)

    Args:
        link: scipy.cluster.hierarchy.linkage 输出的 (N-1)×4 矩阵

    Returns:
        排序后的资产索引列表
    """
    N = link.shape[0] + 1

    # 构建子节点映射 {父节点: [左子, 右子]}
    children = {}
    for i in range(N - 1):
        parent = N + i
        children[parent] = [int(link[i, 0]), int(link[i, 1])]

    def _leaves(cluster: int) -> List[int]:
        if cluster < N:
            return [cluster]
        left, right = children[cluster]
        return _leaves(left) + _leaves(right)

    # 根节点
    root = N + (N - 2)
    return _leaves(root)


def _get_cluster_var(cov: np.ndarray, idx: List[int]) -> float:
    """
    计算聚类簇的方差。

    将簇内资产视为一个迷你组合（等权），计算其方差。

    Args:
        cov: 协方差矩阵
        idx: 簇内资产的索引列表

    Returns:
        簇方差
    """
    w = np.ones(len(idx)) / len(idx)
    return w @ cov[np.ix_(idx, idx)] @ w


def _recursive_bisection(
    cov: np.ndarray, sort_idx: List[int]
) -> np.ndarray:
    """
    递归二分法分配权重。

    从完整资产集开始，沿序列化后的顺序递归二分：
      1. 将当前序列分成左右两个子簇
      2. 计算两个子簇的方差
      3. 按逆方差分配权重：左簇权重 ∝ 1/左方差，右簇权重 ∝ 1/右方差
      4. 递归到每个子簇，直到单个资产

    Args:
        cov: N×N 协方差矩阵
        sort_idx: 序列化后的资产索引

    Returns:
        N×1 权重向量
    """
    weights = np.ones(len(sort_idx))

    def _bisect(w_idx: List[int], w: np.ndarray):
        if len(w_idx) <= 1:
            return

        # 二分
        mid = len(w_idx) // 2
        left_idx = w_idx[:mid]
        right_idx = w_idx[mid:]

        left_var = _get_cluster_var(cov, left_idx)
        right_var = _get_cluster_var(cov, right_idx)

        # 逆方差分配
        inv_left = 1.0 / left_var if left_var > 1e-10 else 1.0
        inv_right = 1.0 / right_var if right_var > 1e-10 else 1.0
        alpha = inv_left / (inv_left + inv_right)

        # 分配到对应资产
        for i in left_idx:
            w[i] *= alpha
        for i in right_idx:
            w[i] *= (1.0 - alpha)

        _bisect(left_idx, w)
        _bisect(right_idx, w)

    _bisect(sort_idx, weights)
    return weights / weights.sum()


# ============================================================================
# 外部接口
# ============================================================================
def solve_hrp_weights(
    cov: np.ndarray,
    codes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    HRP 权重求解主函数。

    Args:
        cov: N×N 协方差矩阵（建议先经 Ledoit-Wolf 收缩）
        codes: 资产代码列表（仅用于返回信息）

    Returns:
        (weights, info)
        weights: 权重向量（和为1）
        info: dict 包含聚类信息
    """
    N = cov.shape[0]

    if N == 0:
        return np.array([]), {"success": True, "mode": "hrp", "note": "empty"}

    if N == 1:
        return np.array([1.0]), {"success": True, "mode": "hrp", "note": "single_asset"}

    # 1. 相关系数矩阵
    std = np.sqrt(np.diag(cov))
    std_outer = np.outer(std, std)
    corr = cov / std_outer

    # 2. 距离矩阵
    dist = _corr_to_dist(corr)

    # 3. 层次聚类
    from scipy.cluster.hierarchy import linkage
    link = linkage(dist, method='single')

    # 4. 序列化
    sort_idx = _get_quasi_diag(link)

    # 5. 递归二分赋权
    weights = _recursive_bisection(cov, sort_idx)

    # 6. 构建信息
    cluster_info = {
        f"asset_{i}": {"idx": sort_idx[i], "weight": weights[i]}
        for i in range(N)
    }

    if codes:
        code_list = [codes[i] for i in sort_idx]
    else:
        code_list = None

    info = {
        "success": True,
        "sort_idx": sort_idx,
        "sorted_codes": code_list,
        "linkage": link,
        "cluster_info": cluster_info,
    }

    return weights, info


# ============================================================================
# 统一配置入口
# ============================================================================
def get_hrp_weights(
    cov: np.ndarray,
    codes: Optional[List[str]] = None,
) -> Tuple[np.ndarray, dict]:
    """
    对外暴露的 HRP 求解接口。
    使用预计算的协方差矩阵（通常来自 HMM 状态或 Ledoit-Wolf 收缩）。

    Args:
        cov: N×N 协方差矩阵
        codes: ETF 代码列表

    Returns:
        (weights, info)
    """
    return solve_hrp_weights(cov=cov, codes=codes)
