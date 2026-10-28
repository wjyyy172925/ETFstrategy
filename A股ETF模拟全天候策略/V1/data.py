"""
数据加载与下载模块
==================
负责从本地CSV加载ETF净值数据、多ETF日期对齐、以及从akshare下载历史数据。

数据来源说明：
  - 本地CSV文件位于 data/etf/，文件名格式为 {code}.csv
  - CSV必须包含date和close两列（date为日期，close为复权净值）
  - 如果本地文件不足，自动调用akshare从东方财富下载补充
"""

import os
import sys
import glob
import traceback
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# 将项目根目录加入path，以便导入config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ETF_POOL, BACKTEST_START, BACKTEST_END

warnings.filterwarnings("ignore")


# ============================================================================
# 配置
# ============================================================================
CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "etf")
CSV_DIR = os.path.normpath(CSV_DIR)

# 中英文列名映射表（支持 data/etf/ 下的中文格式和 akshare 原始返回格式）
CN_TO_EN_COLUMNS = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "振幅": "amplitude",
    "涨跌幅": "change_pct",
    "涨跌额": "change",
    "换手率": "turnover_rate",
    "代码": "code",
    "名称": "name",
}


# ============================================================================
# 辅助函数
# ============================================================================
def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一列名：strip → lowercase → 中文转英文映射（如果列名是中文）。
    确保后续逻辑总能按英文列名（date / close 等）访问。
    """
    new_cols = []
    for c in df.columns:
        c_clean = c.strip().lower()
        if c_clean in CN_TO_EN_COLUMNS:
            new_cols.append(CN_TO_EN_COLUMNS[c_clean])
        else:
            new_cols.append(c_clean)
    df.columns = new_cols
    return df


def _find_csv_file(code: str) -> Optional[str]:
    """
    在CSV目录下查找某个ETF代码对应的CSV文件。
    支持 .csv 和 .CSV 后缀。

    Args:
        code: ETF代码（如"510300"）

    Returns:
        完整文件路径（找到了）或 None（没找到）
    """
    for ext in [".csv", ".CSV"]:
        path = os.path.join(CSV_DIR, f"{code}{ext}")
        if os.path.isfile(path):
            return path
    # 也尝试搜 {code}_*.csv（某些下载器会加后缀）
    pattern = os.path.join(CSV_DIR, f"{code}_*.csv")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def _validate_df(df: pd.DataFrame, code: str) -> bool:
    """
    验证DataFrame是否包含必要列且数据量足够回测。

    Args:
        df: 待验证的DataFrame
        code: ETF代码（仅用于日志）

    Returns:
        True=验证通过, False=数据不足
    """
    required = {"date", "close"}
    if not required.issubset(df.columns):
        print(f"[data] {code} 缺少必要列，需要 {required}，现有 {set(df.columns)}")
        return False
    if len(df) < 10:
        print(f"[data] {code} 数据量太少 ({len(df)}行)")
        return False
    return True


# ============================================================================
# 核心数据加载函数
# ============================================================================
def load_etf_data(
    codes: list,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    加载多个ETF的净值数据，优先从CSV读取，不足时从akshare下载。

    Args:
        codes: ETF代码列表（如 ["510300", "511260"]）
        start: 起始日期 "YYYYMMDD"
        end:   截止日期 "YYYYMMDD"
        force_download: 是否强制重新下载（忽略本地缓存）

    Returns:
        dict: {code: DataFrame(date列, close列)}，date已排序且无重复
    """
    data = {}
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    for code in codes:
        csv_path = _find_csv_file(code)

        # 尝试从CSV加载
        if csv_path and not force_download:
            try:
                df = pd.read_csv(csv_path, encoding="utf-8-sig")
                df = _normalize_columns(df)
                if _validate_df(df, code):
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.dropna(subset=["date"])
                    df = df.sort_values("date").drop_duplicates(subset="date")
                    print(f"[data] 从CSV读取 {code} ({csv_path}): {len(df)}行")
                else:
                    df = None
            except Exception as e:
                print(f"[data] 读取 {code} CSV失败: {e}")
                df = None
        else:
            df = None

        # CSV不存在或数据不足时，从akshare下载
        if df is None:
            try:
                df = _download_etf_akshare(code, start, end)
            except ImportError:
                print("[data] akshare未安装，使用akshare下载需要安装: pip install akshare")
                continue
            except Exception as e:
                print(f"[data] 下载 {code} 失败: {e}")
                continue

        # 截取回测时间范围
        df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)

        if len(df) == 0:
            print(f"[data] 警告: {code} 在 {start}~{end} 范围内无数据")
            continue

        data[code] = df
        print(f"[data] {code} 最终数据: {df['date'].min()} ~ {df['date'].max()}, {len(df)}行")

    return data


def _download_etf_akshare(code: str, start: str, end: str) -> pd.DataFrame:
    """
    使用akshare从新浪/东方财富下载ETF历史净值。

    首先尝试 fund_etf_hist_sina（新浪ETF日线），
    回退到 stock_zh_a_hist（东方财富A股日线，部分ETF支持）。

    Args:
        code: ETF代码
        start: 起始日期 "YYYYMMDD"
        end:   截止日期 "YYYYMMDD"

    Returns:
        包含date和close列的DataFrame

    Raises:
        ImportError: akshare未安装
        ValueError: 下载后数据为空
    """
    import akshare as ak

    # 格式化日期
    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_fmt = f"{end[:4]}-{end[4:6]}-{end[6:]}"

    print(f"[data] 从akshare下载 {code} ...")
    df = None

    # 尝试1：fund_etf_hist_sina（新浪ETF日线，稳定可靠）
    try:
        raw = ak.fund_etf_hist_sina(symbol=f"sh{code}")
        if raw is not None and not raw.empty:
            raw = _normalize_columns(raw)
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw = raw.dropna(subset=["date"])
            # 按日期范围过滤
            raw = raw[(raw["date"] >= start_fmt) & (raw["date"] <= end_fmt)]
            if not raw.empty:
                df = raw[["date", "close"]].copy()
                df["close"] = df["close"].astype(float)
                df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
                return df
    except Exception as e:
        print(f"[data] fund_etf_hist_sina {code} 失败: {e}")
        traceback.print_exc()

    # 尝试2：fund_etf_hist_em（东方财富ETF日线）
    try:
        raw = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start_fmt, end_date=end_fmt, adjust="qfq")
        if raw is not None and not raw.empty:
            raw = _normalize_columns(raw)
            date_col = "date" if "date" in raw.columns else None
            close_col = "close" if "close" in raw.columns else None
            if date_col and close_col:
                df = pd.DataFrame({
                    "date": pd.to_datetime(raw[date_col], errors="coerce"),
                    "close": raw[close_col].astype(float),
                })
                df = df.dropna(subset=["date"])
                df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
                return df
    except Exception as e:
        print(f"[data] fund_etf_hist_em {code} 失败: {e}")
        traceback.print_exc()

    # 尝试3：stock_zh_a_hist（东方财富A股日线，回退方案）
    try:
        raw = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_fmt, end_date=end_fmt, adjust="qfq")
        if raw is not None and not raw.empty:
            raw = _normalize_columns(raw)
            date_col = "date" if "date" in raw.columns else None
            close_col = "close" if "close" in raw.columns else None
            if date_col and close_col:
                df = pd.DataFrame({
                    "date": pd.to_datetime(raw[date_col], errors="coerce"),
                    "close": raw[close_col].astype(float),
                })
                df = df.dropna(subset=["date"])
                df = df.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
                return df
    except Exception as e:
        print(f"[data] stock_zh_a_hist {code} 失败: {e}")
        traceback.print_exc()

    raise ValueError(f"所有下载方式均失败: {code}")


# ============================================================================
# 多ETF日期对齐
# ============================================================================
def align_etf_data(data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, list]:
    """
    将多个ETF的DataFrame按日期对齐，确保所有ETF在同一天都有数据。

    对齐方法：
      1. 取所有ETF日期的交集（共同交易日）
      2. 每个ETF只保留共同交易日的close列
      3. 合并为一个宽表

    Args:
        data_dict: {code: DataFrame(date, close)}

    Returns:
        (aligned_df, codes)
        aligned_df: DataFrame，index为日期，每列为一个ETF的close
        codes: ETF代码列表（按原始顺序）
    """
    if not data_dict:
        raise ValueError("data_dict 为空，无可对齐的数据")

    codes = list(data_dict.keys())
    # 将所有ETF的close列按日期对齐
    series_list = []
    for code in codes:
        df = data_dict[code]
        s = df.set_index("date")["close"]
        s.name = code
        series_list.append(s)

    # 合并为宽表，用内连接取日期交集。
    # 注意：inner join 会丢弃所有 ETF 中任一缺少数据的日期。
    # 若某 ETF 上市较晚（如 159649 于 2022 年上市），
    # 其之前的历史区间将被截断。
    combined = pd.concat(series_list, axis=1, join="inner")
    combined.index = pd.to_datetime(combined.index)
    combined = combined.sort_index()

    print(f"[data] 日期对齐后: {len(combined)}个交易日, {len(codes)}只ETF")
    for code in codes:
        print(f"  {code}: 均值{combined[code].mean():.3f}, "
              f"最新{combined[code].iloc[-1]:.3f}, "
              f"缺失率{(1 - combined[code].notna().mean())*100:.1f}%")

    return combined, codes


# ============================================================================
# 便捷加载
# ============================================================================
def load_and_align(
    codes: list = None,
    start: str = None,
    end: str = None,
) -> Tuple[pd.DataFrame, list]:
    """
    一键加载→对齐→返回。

    典型用法：
        price_df, codes = load_and_align()
        # price_df: index=date, columns=ETF代码

    Args:
        codes: ETF代码列表，默认从config的ETF_POOL提取
        start: 起始日期，默认BACKTEST_START
        end:   截止日期，默认BACKTEST_END

    Returns:
        (price_df, codes) 同align_etf_data
    """
    if codes is None:
        codes = [etf["code"] for etf in ETF_POOL]
    if start is None:
        start = BACKTEST_START
    if end is None:
        end = BACKTEST_END

    raw = load_etf_data(codes, start, end)
    return align_etf_data(raw)


# ============================================================================
# 入口测试
# ============================================================================
if __name__ == "__main__":
    price_df, codes = load_and_align()
    print(f"\n对齐后的价格矩阵形状: {price_df.shape}")
    print(price_df.head())