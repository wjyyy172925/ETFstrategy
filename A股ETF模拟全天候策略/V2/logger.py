"""
日志模块
========
使用Python标准logging模块，配置RotatingFileHandler实现日志轮转。

日志文件:
  - backtest.log: 主回测日志（含再平衡、风控、交易、月报等）
  - 按大小轮转：每个文件10MB，最多保留5个历史文件

日志级别:
  - INFO: 再平衡、风控、月报、交易等关键事件
  - WARNING: 风控触发（回撤预警、CVaR预警、紧急退出）
  - DEBUG: 每日状态（每20天输出一次）
  - ERROR: 数据缺失、优化失败等异常（暂未使用，预留）

日志格式:
  [YYYY-MM-DD HH:MM:SS,mmm] LEVEL | 模块 | 消息
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR


# ============================================================================
# 日志格式定义
# ============================================================================
_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_ENCODING = "utf-8"


def setup_logger(
    name: str = "backtest",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    配置并返回日志记录器。

    创建规则：
      1. 如果已经存在该名称的logger，直接返回（避免重复handler）
      2. 创建RotatingFileHandler写入文件，同时添加StreamHandler输出到控制台
      3. 文件handler级别=传入level，控制台handler级别=INFO

    Args:
        name: logger名称，用于区分不同模块的日志
        log_dir: 日志文件保存目录（默认使用config.OUTPUT_DIR下的logs子目录）
        level: 日志级别，默认INFO

    Returns:
        配置好的logging.Logger实例
    """
    # 获取或创建logger
    logger = logging.getLogger(name)

    # 如果已经配置过handler，直接返回
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # ================================================================
    # 日志目录
    # ================================================================
    if log_dir is None:
        log_dir = os.path.join(OUTPUT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{name}.log")

    # ================================================================
    # 文件Handler：RotatingFileHandler（按大小轮转）
    # ================================================================
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,               # 保留5个备份
        encoding=_LOG_ENCODING,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(file_handler)

    # ================================================================
    # 控制台Handler：输出到stdout
    # ================================================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台只输出INFO及以上
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(console_handler)

    logger.info(f"日志初始化完成 | 文件: {log_path} | 级别: {logging.getLevelName(level)}")

    return logger


# ============================================================================
# 月报生成（预留）
# ============================================================================
def monthly_report(
    logger: logging.Logger,
    date,
    metrics: dict,
    weights: dict,
):
    """
    输出月度绩效摘要到日志。

    包含：组合净值、月度收益、年化收益、波动率、当前权重分布。

    Args:
        logger: 日志记录器
        date: 报告日期
        metrics: 绩效指标dict
        weights: 当前权重dict {ETF代码: 权重}
    """
    logger.info(
        f"[月报] {date} | "
        f"净值:{metrics.get('nav', 0):.4f} | "
        f"月收益:{metrics.get('monthly_return', 0):.2%} | "
        f"年化收益:{metrics.get('annual_return', 0):.2%} | "
        f"年化波动:{metrics.get('annual_vol', 0):.2%} | "
        f"权重:{weights}"
    )
