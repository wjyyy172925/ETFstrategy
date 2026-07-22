import os
import pandas as pd
import sys

DATA_DIR = "data/etf"


def get_etf_data_path(code: str) -> str:
    return os.path.join(DATA_DIR, f"{code}.csv")


def load_single_etf(code: str) -> pd.DataFrame:
    path = get_etf_data_path(code)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").set_index("日期")
    return df


def load_multi_etf(codes: list[str]) -> pd.DataFrame:
    close_df = pd.DataFrame()
    for code in codes:
        df = load_single_etf(code)
        close_df[code] = df["收盘"]
    close_df = close_df.dropna()
    return close_df


def download_all_data():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from download_etf import download_etf_history, ensure_dir
    from config import BACKTEST_START, BACKTEST_END, ETF_POOL, BENCHMARK_CODES

    ensure_dir(DATA_DIR)
    all_codes = set()
    for etf in ETF_POOL:
        all_codes.add(etf["code"])
    for code in BENCHMARK_CODES:
        all_codes.add(code)
    for code in sorted(all_codes):
        try:
            name = next((e["name"] for e in ETF_POOL if e["code"] == code), "")
            download_etf_history(code, name, BACKTEST_START, BACKTEST_END, force=False)
        except Exception as e:
            print(f"下载 {code} 失败: {e}")
