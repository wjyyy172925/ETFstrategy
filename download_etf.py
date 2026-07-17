#!/usr/bin/env python3
"""A股ETF数据下载工具 - 将ETF历史日线数据下载到本地CSV

功能:
  1. 从 akshare 获取全市场ETF实时列表（代码、名称等）
  2. 根据列表逐只下载历史日线行情（复权因子、开高低收、成交量等）
  3. 支持增量更新（仅下载缺失日期）、强制覆盖、指定ETF子集
  4. 本地以 CSV 形式存储，每只ETF一个文件，文件名为 "代码.csv"

数据来源:
  - ETF 列表: akshare 的 fund_etf_spot_em()，即东方财富 ETF 实时行情
  - 历史日线: akshare 的 fund_etf_hist_em()，即东方财富 ETF 历史行情
  - 如果东方财富失败，尝试新浪数据源作为降级
"""

import os
import sys
import time
import random
import argparse
from datetime import datetime, timedelta

import pandas as pd
import akshare as ak

# 交易所代码前缀映射（用于新浪数据源）
SH_CODES = {"51", "56", "58"}  # 上海ETF代码前缀
SZ_CODES = {"15", "16"}        # 深圳ETF代码前缀


def code_to_exchange(code):
    """根据ETF代码前2位判断交易所，返回新浪API所需的前缀。

    Args:
        code: 6位ETF代码字符串。

    Returns:
        str: "sh" 或 "sz"。
    """
    prefix = code[:2]
    if prefix in SH_CODES:
        return "sh"
    return "sz"

# ============================================================
# 全局配置常量
# ============================================================

DATA_DIR = "data/etf"              # ETF历史数据存放目录，每只ETF存为一个 CSV 文件
ETF_LIST_FILE = "data/etf_list.csv"  # ETF列表缓存文件，避免每次运行都重新拉取
SLEEP_SECONDS = 1.5                # 两次请求之间的间隔（秒），避免触发 API 频率限制


# ============================================================
# 工具函数
# ============================================================

def ensure_dir(path):
    """确保指定目录存在，如果不存在则递归创建。

    Args:
        path: 目录路径。
    """
    os.makedirs(path, exist_ok=True)


# ============================================================
# ETF 列表管理
# ============================================================

def download_etf_list():
    """从东方财富（akshare）拉取全市场ETF实时行情列表。

    数据包含: 代码、名称、最新价、涨跌幅、成交量、成交额等字段。
    结果保存到本地 CSV（ETF_LIST_FILE），并同时返回 DataFrame。

    Returns:
        pd.DataFrame: ETF 列表，至少包含 "代码" 和 "名称" 两列。
    """
    df = ak.fund_etf_spot_em()
    df.to_csv(ETF_LIST_FILE, index=False, encoding="utf-8-sig")
    print(f"ETF列表已保存: {ETF_LIST_FILE} ({len(df)} 只)")
    return df


def load_etf_list():
    """加载本地缓存的ETF列表；如果本地文件不存在，则触发 download_etf_list()。

    Returns:
        pd.DataFrame: ETF 列表 DataFrame。
    """
    if os.path.exists(ETF_LIST_FILE):
        df = pd.read_csv(ETF_LIST_FILE)
        print(f"加载ETF列表: {len(df)} 只")
        return df
    return download_etf_list()


# ============================================================
# 增量更新判断
# ============================================================

def get_latest_date(code):
    """读取指定ETF本地 CSV 中已有数据的最大（最新）日期。

    Args:
        code: ETF 代码字符串，例如 "510050"。

    Returns:
        datetime 或 None: 如果本地文件存在且有数据则返回最大日期，否则返回 None。
    """
    path = os.path.join(DATA_DIR, f"{code}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        return pd.to_datetime(df["日期"]).max()
    return None


# ============================================================
# 单只 ETF 历史数据下载
# ============================================================

def download_etf_history(code, name, start_date, end_date, force=False, delay=1.5, max_retries=3):
    """下载单只ETF在指定日期范围内的历史日线数据，并保存到本地 CSV。

    工作逻辑:
       1. 如果 force=False 且本地已有数据，则从最新日期+1天开始增量下载。
       2. 调用 akshare 获取东方财富的日线行情，包含日期、开高低收、成交量、成交额、振幅、换手率等。
       3. 在原始字段前插入 "代码" 和 "名称" 两列，方便多文件合并时追溯。
       4. 如果已有本地文件且非强制覆盖，则与旧数据合并、去重、排序后保存。
       5. 如果是首次下载或强制覆盖，则直接覆盖写入。
       6. 遇到连接错误时自动重试（指数退避）。

    Args:
        code:        ETF 代码，6位数字字符串（如 "510050"）。
        name:        ETF 名称（如 "上证50ETF"），仅用于打印日志与 CSV 标记。
        start_date:  起始日期字符串，格式 "YYYYMMDD"。
        end_date:    结束日期字符串，格式 "YYYYMMDD"。
        force:       是否强制覆盖 —— True 则重新下载全部并覆盖本地文件。
        delay:       重试之间的基础等待时间（秒）。
        max_retries: 最大重试次数。

    Returns:
        pd.DataFrame 或 None:
            - 若有新数据返回包含新数据的 DataFrame。
            - 如果已是最新（无需更新）或请求无数据，返回 None。

    Raises:
        Exception: 所有重试耗尽后仍失败时向上抛出异常。
    """
    path = os.path.join(DATA_DIR, f"{code}.csv")

    # ---- 增量模式下判断是否需要下载 ----
    if not force:
        latest = get_latest_date(code)
        if latest is not None:
            # 从已有数据最新日期的次日开始请求
            start = (latest + timedelta(days=1)).strftime("%Y%m%d")
            if start >= end_date:
                # 起始日期已超过结束日期，说明本地数据已经包含最新数据
                return None

    # ---- 带重试的 API 调用 ----
    last_error = None
    for attempt in range(max_retries):
        try:
            df = ak.fund_etf_hist_em(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
            )
            if df.empty:
                return None

            # 插入标识列，方便后续多文件合并时区分来源
            df.insert(0, "代码", code)
            df.insert(1, "名称", name)

            # ---- 写入/合并到本地 CSV ----
            if os.path.exists(path) and not force:
                # 增量模式: 读取旧数据，合并新数据，去重后写回
                old = pd.read_csv(path)
                combined = pd.concat([old, df], ignore_index=True)
                # 以 (日期, 代码) 为唯一键去重，后出现的保留（keep="last"）
                combined.drop_duplicates(subset=["日期", "代码"], keep="last", inplace=True)
                combined.sort_values("日期", inplace=True)
                combined.to_csv(path, index=False, encoding="utf-8-sig")
            else:
                # 首次下载或强制覆盖模式
                df.sort_values("日期", inplace=True)
                df.to_csv(path, index=False, encoding="utf-8-sig")
            return df

        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                time.sleep(wait)
                continue

    # ---- 东方财富失败，尝试新浪数据源作为降级 ----
    try:
        sina_symbol = f"{code_to_exchange(code)}{code}"
        df = ak.fund_etf_hist_sina(symbol=sina_symbol)
        if df.empty:
            return None

        # 统一列名为东方财富格式
        df = df.rename(columns={
            "date": "日期",
            "open": "开盘",
            "close": "收盘",
            "high": "最高",
            "low": "最低",
            "volume": "成交量",
            "amount": "成交额",
        })
        # 补齐缺少的列（东方财富有但新浪没有的）
        for col in ["振幅", "涨跌幅", "涨跌额", "换手率"]:
            df[col] = None

        df = df[["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"]]
        df.insert(0, "代码", code)
        df.insert(1, "名称", name)
        df["日期"] = pd.to_datetime(df["日期"]).dt.strftime("%Y-%m-%d")

        # ---- 写入/合并到本地 CSV ----
        if os.path.exists(path) and not force:
            old = pd.read_csv(path)
            combined = pd.concat([old, df], ignore_index=True)
            combined.drop_duplicates(subset=["日期", "代码"], keep="last", inplace=True)
            combined.sort_values("日期", inplace=True)
            combined.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            df.sort_values("日期", inplace=True)
            df.to_csv(path, index=False, encoding="utf-8-sig")
        return df

    except Exception as e:
        raise RuntimeError(f"东方财富: {last_error}; 新浪: {e}")



# ============================================================
# 主入口（CLI）
# ============================================================

def main():
    """命令行入口：解析参数 → 获取/加载ETF列表 → 逐只下载历史数据。

    支持的参数:
      --start      起始日期，默认10年前
      --end        结束日期，默认今天
      --force      强制重新下载（覆盖已有数据）
      --delay      两次请求间隔（秒），默认 1.5
      --codes      指定 ETF 代码列表（空格分隔），仅下载这些 ETF
      --update-list 强制重新拉取 ETF 列表（忽略本地缓存）
    """
    parser = argparse.ArgumentParser(description="A股ETF数据下载工具")
    parser.add_argument(
        "--start",
        default=(datetime.now() - timedelta(days=3650)).strftime("%Y%m%d"),
        help="起始日期 YYYYMMDD",
    )
    parser.add_argument(
        "--end",
        default=datetime.now().strftime("%Y%m%d"),
        help="结束日期 YYYYMMDD",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载全部数据（覆盖已有）",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=SLEEP_SECONDS,
        help="请求间隔秒数（默认 %.1f）" % SLEEP_SECONDS,
    )
    parser.add_argument(
        "--codes",
        nargs="+",
        help="指定ETF代码列表（空格分隔），如 --codes 510050 159915",
    )
    parser.add_argument(
        "--update-list",
        action="store_true",
        help="强制更新ETF列表（忽略本地缓存）",
    )
    args = parser.parse_args()

    # 确保数据存放目录存在
    ensure_dir(DATA_DIR)

    # ---- 加载/下载 ETF 列表 ----
    if args.update_list or not os.path.exists(ETF_LIST_FILE):
        etf_df = download_etf_list()
    else:
        etf_df = load_etf_list()

    # ---- 若指定了 --codes，只处理指定的 ETF ----
    if args.codes:
        etf_df = etf_df[etf_df["代码"].astype(str).isin(args.codes)]

    total = len(etf_df)
    print(f"准备下载 {total} 只ETF数据，区间 {args.start} ~ {args.end}")

    # ---- 统计计数器 ----
    success = 0   # 成功下载到新数据的 ETF 数量
    skipped = 0   # 已是最新 / 无数据的 ETF 数量
    fail = 0      # 下载失败的 ETF 数量

    # ---- 逐只下载 ----
    for i, (_, row) in enumerate(etf_df.iterrows(), 1):
        code = str(row["代码"]).zfill(6)  # 补齐6位，如 "510050"
        name = row["名称"]
        print(f"[{i}/{total}] {code} {name} ... ", end="", flush=True)

        try:
            df = download_etf_history(
                code, name, args.start, args.end,
                force=args.force, delay=args.delay,
            )
            if df is None:
                print("已是最新")
                skipped += 1
            else:
                print(f"{len(df)} 条")
                success += 1
        except Exception as e:
            print(f"  [失败] {code} {name}: {e}")
            fail += 1

        # 请求间隔（随机 jitter），避免触发 API 频率限制
        if i < total:
            time.sleep(random.uniform(args.delay * 0.5, args.delay * 1.5))

    # ---- 最终报告 ----
    print(f"\n完成！成功: {success}, 跳过: {skipped}, 失败: {fail}")
    print(f"数据目录: {os.path.abspath(DATA_DIR)}")


if __name__ == "__main__":
    main()
