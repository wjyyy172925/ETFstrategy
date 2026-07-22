ETF_POOL = [
    {"code": "510300", "name": "沪深300ETF",  "asset_class": "equity"},
    {"code": "510500", "name": "中证500ETF",  "asset_class": "equity"},
    {"code": "511260", "name": "十年国债ETF", "asset_class": "long_bond"},
    {"code": "511010", "name": "国债ETF",     "asset_class": "mid_bond"},
    {"code": "159649", "name": "国开债ETF",   "asset_class": "mid_bond"},
    {"code": "518880", "name": "黄金ETF",     "asset_class": "gold"},
    {"code": "159985", "name": "豆粕ETF",     "asset_class": "commodity"},
]

BENCHMARK_CODES = ["510300"]

ASSET_CLASS_MAP = {
    "equity":    {"risk_budget": 0.25, "label": "权益"},
    "long_bond": {"risk_budget": 0.25, "label": "长久期利率债"},
    "mid_bond":  {"risk_budget": 0.25, "label": "中短债+信用债"},
    "gold":      {"risk_budget": 0.25, "label": "黄金"},
    "commodity": {"risk_budget": 0.25, "label": "商品"},
}

WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.40
SINGLE_ETF_MAX = 0.30
VOL_WINDOW = 60

REBALANCE_FREQ = "ME"
THRESHOLD_PCT = 0.08
VOL_CHANGE_THRESHOLD = 0.30
EXECUTION_RATIO = 0.80

TARGET_VOL = 0.10
VOL_BAND = (0.08, 0.12)
CVAR_THRESHOLD = 0.15
DRAWDOWN_WARN = 0.10
DRAWDOWN_MAX = 0.20
LIQUIDITY_MIN_DAILY_VOL = 50_000_000
DURATION_LIMIT = 7.0

COMMISSION_RATE = 0.00003
SLIPPAGE_RATE = 0.0005
TOTAL_COST_PER_SIDE = COMMISSION_RATE + SLIPPAGE_RATE

RISK_FREE_RATE = 0.025

INITIAL_CAPITAL = 100_000_000

BACKTEST_START = "20150101"
BACKTEST_END = "20260722"

CLASSIC_AA_WEIGHTS = [0.30, 0.55, 0.075, 0.075]
EQUAL_WEIGHT = [0.25, 0.25, 0.25, 0.25]
