-- ============================================================
-- schema.sql — SuperLotto Plus 資料庫綱要定義
-- ============================================================
-- 設計原則：
--   • Raw 資料表（draws, draw_prizes）永遠只新增，不修改，保護原始資料完整性
--   • Derived 資料表（*_stats, draw_features, number_gap_history）可隨時重算
--   • 擴充欄位（machine_id, temperature 等）預留但允許 NULL，供未來補充
-- ============================================================

PRAGMA foreign_keys = ON;  -- 啟用外鍵約束（SQLite 預設關閉）
PRAGMA journal_mode = WAL; -- Write-Ahead Logging：改善並發寫入效能


-- ============================================================
-- 資料表 1：原始開獎紀錄（Raw draw results）
-- 每一期開獎對應一筆資料，是整個資料庫的核心事實表
-- ============================================================
CREATE TABLE IF NOT EXISTS draws (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_number     INTEGER UNIQUE NOT NULL,     -- 官方期號（全域唯一，用於跨表關聯）
    draw_date       DATE    NOT NULL,            -- 開獎日期 (YYYY-MM-DD)

    -- 5 顆白球（儲存時已排序由小到大，方便查詢比對）
    n1  INTEGER NOT NULL CHECK(n1 BETWEEN 1 AND 47),
    n2  INTEGER NOT NULL CHECK(n2 BETWEEN 1 AND 47),
    n3  INTEGER NOT NULL CHECK(n3 BETWEEN 1 AND 47),
    n4  INTEGER NOT NULL CHECK(n4 BETWEEN 1 AND 47),
    n5  INTEGER NOT NULL CHECK(n5 BETWEEN 1 AND 47),

    mega_number     INTEGER NOT NULL CHECK(mega_number BETWEEN 1 AND 27),
    jackpot_amount  REAL,    -- 頭獎金額（可能為 NULL，API 未必每期都有）

    -- ── 物理偏差擴充欄位（目前留空，待未來補充）────────────────
    -- 理論上，搖獎機磨損、彩球重量偏差、溫濕度都可能造成微小的統計偏差
    machine_id      TEXT,    -- 搖獎機編號（例如 "M-01"）
    ball_set_id     TEXT,    -- 彩球組編號（例如 "Set-A"）
    temperature     REAL,    -- 開獎現場溫度（°F）
    humidity        REAL,    -- 相對濕度（%，0.0–100.0）

    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 常用查詢索引
CREATE INDEX IF NOT EXISTS idx_draws_date        ON draws(draw_date);
CREATE INDEX IF NOT EXISTS idx_draws_draw_number ON draws(draw_number);


-- ============================================================
-- 資料表 2：獎項明細（Prize breakdown per draw）
-- 每期有多個獎項等級，與 draws 是「一對多」關係，必須獨立成表
-- ============================================================
CREATE TABLE IF NOT EXISTS draw_prizes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_number       INTEGER NOT NULL REFERENCES draws(draw_number) ON DELETE CASCADE,
    prize_description TEXT    NOT NULL,  -- 例如 "5 + Mega", "4 Numbers", "3 + Mega"
    winner_count      INTEGER,           -- 該獎項得獎人數（可能為 0）
    prize_amount      REAL,              -- 每人得獎金額（美元）

    UNIQUE(draw_number, prize_description)  -- 同一期同一獎項只能有一筆
);

CREATE INDEX IF NOT EXISTS idx_prizes_draw_number ON draw_prizes(draw_number);


-- ============================================================
-- 資料表 3：白球號碼頻率彙總（Hot/Cold analysis for white balls）
-- 每個號碼（1–47）一筆，記錄全域統計資訊
-- 此表為 derived，可由 draws 表隨時重算
-- ============================================================
CREATE TABLE IF NOT EXISTS white_ball_stats (
    number              INTEGER PRIMARY KEY CHECK(number BETWEEN 1 AND 47),
    total_appearances   INTEGER DEFAULT 0,  -- 歷史出現總次數
    last_seen_draw      INTEGER,            -- 最後出現的期號
    last_seen_date      DATE,              -- 最後出現日期
    current_gap         INTEGER DEFAULT 0, -- 距最新一期已有幾期未出現（冷號指標）
    avg_gap             REAL,              -- 平均出現間隔（期數）
    max_gap             INTEGER,           -- 歷史最長連續缺席期數
    min_gap             INTEGER,           -- 歷史最短連續出現間隔

    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- 資料表 4：Mega 球頻率彙總（Hot/Cold analysis for mega ball）
-- 結構與 white_ball_stats 相同，但號碼範圍是 1–27
-- ============================================================
CREATE TABLE IF NOT EXISTS mega_ball_stats (
    number              INTEGER PRIMARY KEY CHECK(number BETWEEN 1 AND 27),
    total_appearances   INTEGER DEFAULT 0,
    last_seen_draw      INTEGER,
    last_seen_date      DATE,
    current_gap         INTEGER DEFAULT 0,
    avg_gap             REAL,
    max_gap             INTEGER,
    min_gap             INTEGER,

    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- 資料表 5：每期衍生特徵（Per-draw engineered features）
-- 與 draws 是「一對一」關係（每期一筆），分開存放是為了：
--   1. 保持原始資料的純淨性
--   2. 特徵計算錯誤時可重算，不影響原始資料
-- ============================================================
CREATE TABLE IF NOT EXISTS draw_features (
    draw_number         INTEGER PRIMARY KEY REFERENCES draws(draw_number) ON DELETE CASCADE,

    -- 總和統計（常態分佈中間值附近的組合更常見）
    white_sum           INTEGER,  -- 5 顆白球號碼總和
    white_mean          REAL,     -- 5 顆白球號碼平均值

    -- 奇偶比例（0–5）
    odd_count           INTEGER,  -- 奇數球個數
    even_count          INTEGER,  -- 偶數球個數

    -- 高低區比例（低區：1–23；高區：24–47）
    low_count           INTEGER,  -- 低區號碼個數
    high_count          INTEGER,  -- 高區號碼個數

    -- 連號統計（連續相鄰號碼的對數，例如 [7,8,12] → 1 對連號）
    consecutive_pairs   INTEGER,

    -- 號碼跨度（最大號碼 − 最小號碼，跨度小表示號碼集中）
    number_range        INTEGER,

    -- 個位數多樣性（0–9 中用到幾種個位數，越高越分散）
    unique_last_digits  INTEGER,

    -- Mega 球特徵
    mega_is_odd         INTEGER,  -- Mega 球是否為奇數（0 或 1）

    computed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
-- 資料表 6：號碼間隔歷史序列（Gap series for time-series analysis）
-- 記錄每個號碼「每次出現時，距上次出現間隔了幾期」
-- 用途：
--   • 分析號碼的間隔分佈（正態？泊松？）
--   • 作為 LSTM 等時序模型的輸入序列
-- ============================================================
CREATE TABLE IF NOT EXISTS number_gap_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ball_type   TEXT    NOT NULL CHECK(ball_type IN ('white', 'mega')),
    number      INTEGER NOT NULL,
    draw_number INTEGER NOT NULL REFERENCES draws(draw_number) ON DELETE CASCADE,
    gap_periods INTEGER NOT NULL,  -- 距上次出現間隔幾期（首次出現時為 NULL）

    UNIQUE(ball_type, number, draw_number)  -- 每個號碼每期只記錄一次
);

CREATE INDEX IF NOT EXISTS idx_gap_history_number ON number_gap_history(ball_type, number);
