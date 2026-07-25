"""
HMM 隐马尔可夫模型市场状态识别
==============================

使用 Gaussian HMM 将市场划分为 3 个状态：
  - 状态 0: 高波动熊市（低收益率、高波动）
  - 状态 1: 低波动震荡（收益率≈0、低波动）
  - 状态 2: 复苏牛市（高收益率、中波动）

每个状态维护独立的 Ledoit-Wolf 收缩协方差矩阵，
在再平衡日识别当前状态后，使用对应状态协方差做 HRP。

实现细节：
  - 使用 hmmlearn.GaussianHMM，3 状态
  - 训练窗口 252 个交易日（约 1 年）
  - 每季度重新训练一次
  - Viterbi 解码当前状态
  - 状态切换时协方差平滑过渡，防止权重跳变
"""

import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ETF_POOL

from hrp import ledoit_wolf_cov


# ============================================================================
# HMM 市场状态检测器
# ============================================================================
class HMMRegimeDetector:
    """
    HMM 市场状态检测器。

    使用 Gaussian HMM 将市场划分为多个状态，每个状态
    维护独立的 Ledoit-Wolf 收缩协方差。

    Attributes:
        n_states: 状态数量（默认 3）
        window: 训练窗口（交易日）
        retrain_freq: 重新训练频率（交易日）
        model: hmmlearn GaussianHMM 实例
        state_covs: dict {state_idx: LW_cov_matrix}
        current_state: 最新预测状态
        state_history: list 状态历史
        state_labels: 状态中文标签
    """

    def __init__(
        self,
        n_states: int = 3,
        window: int = 252,
        retrain_freq: int = 63,
    ):
        self.n_states = n_states
        self.window = window
        self.retrain_freq = retrain_freq
        self.model = None
        self.state_covs: Dict[int, np.ndarray] = {}
        self.current_state = 0
        self.state_history: List[int] = []
        self._last_train_t = -999
        self._trained_once = False

        # 状态释义（运行时根据均值方差自动标注）
        self.state_labels: Dict[int, str] = {}

    def _auto_label_states(self, means: np.ndarray, covs: np.ndarray):
        """
        根据 HMM 的均值和方差自动标注状态。

        规则：
          - 均值最低 → 熊市
          - 均值最高 → 牛市
          - 其余 → 震荡

        Args:
            means: (n_states, n_features) 均值矩阵
            covs: (n_states, n_features, n_features) 协方差矩阵
        """
        n_features = means.shape[1]
        state_means = means.mean(axis=1)
        mean_order = np.argsort(state_means)

        self.state_labels[int(mean_order[0])] = "熊市"
        self.state_labels[int(mean_order[-1])] = "牛市"

        for i in range(self.n_states):
            if i not in [int(mean_order[0]), int(mean_order[-1])]:
                self.state_labels[i] = "震荡"

    def fit(self, returns: np.ndarray):
        """
        训练 HMM 模型。

        使用指定窗口的历史收益率训练 Gaussian HMM，然后：
          1. 解码每个时间点最可能的状态
          2. 为每个状态计算 Ledoit-Wolf 收缩协方差
          3. 自动标注状态含义

        Args:
            returns: T×N 收益率矩阵
        """
        T = returns.shape[0]
        if T < self.window:
            train_returns = returns
        else:
            train_returns = returns[-self.window:]

        from hmmlearn import hmm

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            tol=1e-4,
            n_iter=1000,
            random_state=42,
        )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not converging.*")
                self.model.fit(train_returns)
        except Exception as e:
            print(f"[hmm] 训练失败: {e}，使用回退模式")
            self._trained_once = False
            return

        self._trained_once = True

        # 解码状态
        states = self.model.predict(train_returns)
        self.state_history = list(states)

        # 每个状态的 LW 协方差
        N = returns.shape[1]
        for s in range(self.n_states):
            mask = states == s
            if mask.sum() >= 2 * N:
                state_returns = train_returns[mask]
                cov_lw, delta = ledoit_wolf_cov(state_returns)
                self.state_covs[s] = cov_lw
            else:
                self.state_covs[s] = np.cov(train_returns, rowvar=False) + np.eye(N) * 1e-6

        # 自动标注
        self._auto_label_states(self.model.means_, self.model.covars_)

        current_state = states[-1]
        self.current_state = current_state
        self._last_train_t = T

        print(f"[hmm] 训练完成 | 状态: {self.state_labels} | 当前: {self.state_labels.get(current_state, '?')}")

    def predict(self, returns: np.ndarray) -> int:
        """
        预测最新时间点的市场状态。

        Args:
            returns: T×N 收益率矩阵

        Returns:
            当前状态编号（0 ~ n_states-1）
        """
        if self.model is None or not self._trained_once:
            return 0

        T = returns.shape[0]
        if T < 10:
            return self.current_state

        try:
            states = self.model.predict(returns)
            self.state_history = list(states)
            self.current_state = states[-1]
        except Exception:
            pass

        return self.current_state

    def get_regime_covariance(
        self, price_df: pd.DataFrame, current_state: int
    ) -> np.ndarray:
        """
        获取指定市场状态的 Ledoit-Wolf 收缩协方差。

        如果该状态的协方差不可用（数据不足），回退到全局 LW。

        Args:
            price_df: ETF 价格矩阵
            current_state: 当前状态编号

        Returns:
            N×N 协方差矩阵
        """
        if current_state in self.state_covs:
            return self.state_covs[current_state].copy()

        # 回退：全局 Ledoit-Wolf
        returns = np.log(price_df).diff().dropna().values
        cov_lw, _ = ledoit_wolf_cov(returns)
        return cov_lw

    def smooth_transition(
        self, cov_new: np.ndarray, cov_old: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        状态切换时协方差平滑过渡。

        使用指数加权：cov = alpha * cov_new + (1-alpha) * cov_old

        Args:
            cov_new: 新状态的协方差
            cov_old: 旧状态的协方差
            alpha: 新协方差的权重（默认 0.5）

        Returns:
            平滑后的协方差
        """
        return alpha * cov_new + (1.0 - alpha) * cov_old

    def get_state_label(self, state: int) -> str:
        """获取状态的中文标签。"""
        return self.state_labels.get(state, f"状态{state}")

    def summary(self) -> dict:
        """返回当前状态摘要。"""
        return {
            "current_state": self.current_state,
            "current_label": self.get_state_label(self.current_state),
            "state_labels": self.state_labels,
            "trained": self._trained_once,
            "n_states": self.n_states,
        }


# ============================================================================
# 便捷函数
# ============================================================================
def get_global_lw_cov(price_df: pd.DataFrame) -> np.ndarray:
    """
    计算全局 Ledoit-Wolf 收缩协方差（不依赖 HMM 状态）。

    作为 HMM 不可用时的回退方案。

    Args:
        price_df: ETF 价格矩阵

    Returns:
        N×N 收缩协方差矩阵
    """
    returns = np.log(price_df).diff().dropna().values
    cov_lw, delta = ledoit_wolf_cov(returns)
    return cov_lw
