"""
settings.py — 全域設定檔
========================
所有模組的唯一設定來源 (Single Source of Truth)。
修改這裡即可調整整個專案的行為，不需要逐一修改各模組。
"""

import pathlib

# ── 專案路徑 ────────────────────────────────────────────────────────────────
# BASE_DIR 指向專案根目錄（settings.py 往上兩層：config/ → 根目錄）
BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH  = DATA_DIR / "superlotto.db"
EXPORTS_DIR = DATA_DIR / "exports"

# 確保 data/ 資料夾存在（首次 import 時自動建立）
DATA_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)

# ── CA Lottery API 設定 ──────────────────────────────────────────────────────
# 官方公開的 JSON API，無需登入或 API key
API_BASE_URL  = "https://www.calottery.com/api/DrawGameApi/DrawGamePastDrawResults"
GAME_ID       = 8      # SuperLotto Plus 的遊戲 ID（可至官網確認）
PAGE_SIZE     = 50     # 每頁資料筆數（API 最大支援 50，超過會回傳 null）

# 爬蟲禮儀設定
REQUEST_DELAY   = 3.0  # 每次 HTTP 請求之間的等待秒數（避免觸發速率限制）
REQUEST_TIMEOUT = 30   # 單次請求的逾時秒數

# ── SuperLotto Plus 號碼規則 ────────────────────────────────────────────────
WHITE_BALL_MIN = 1
WHITE_BALL_MAX = 47    # 白球號碼範圍：1–47
MEGA_BALL_MIN  = 1
MEGA_BALL_MAX  = 27    # Mega 球號碼範圍：1–27

# 高低區分界點（用於特徵工程）
# 1–LOW_HIGH_THRESHOLD 為低區，(LOW_HIGH_THRESHOLD+1)–47 為高區
LOW_HIGH_THRESHOLD = 23

# ── 特徵工程參數 ─────────────────────────────────────────────────────────────
# 短期滾動視窗：計算「近期熱號」時採用的期數範圍
SHORT_WINDOW = 20
# 中期滾動視窗
MID_WINDOW   = 50
# 長期滾動視窗（近似全部歷史）
LONG_WINDOW  = 200

# ── ML 模型參數 ──────────────────────────────────────────────────────────────
# 滑動視窗大小：用前幾期的特徵來預測下一期
LOOKBACK_PERIODS = 10

# 訓練 / 測試比例（時序切分，前 80% 訓練，後 20% 測試）
# 注意：彩票資料為時間序列，絕對不能隨機 shuffle
TRAIN_RATIO = 0.80

# Random Forest 超參數
RF_N_ESTIMATORS = 300
RF_RANDOM_STATE = 42

# ── 日誌設定 ─────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"   # DEBUG / INFO / WARNING / ERROR
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
