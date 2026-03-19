"""
repository.py — 資料庫 CRUD 操作封裝
=======================================
Repository Pattern：把所有 SQL 操作集中在這個模組，
其他模組只需呼叫語義清晰的方法（upsert_draws、get_all_draws 等），
不需要知道底層 SQL 細節。

好處：
  • 未來若從 SQLite 遷移到 PostgreSQL，只需修改這個檔案
  • SQL 邏輯集中，容易測試和維護
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.database.connection import get_connection

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DrawRepository:
    """
    SuperLotto Plus 開獎資料的資料存取物件（DAO）。
    所有 SQL 操作都透過 get_connection() context manager 執行。
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    # ────────────────────────────────────────────────────────────────
    # WRITE 操作
    # ────────────────────────────────────────────────────────────────

    def upsert_draws(self, draws: list[dict]) -> int:
        """
        批量新增或更新開獎資料（UPSERT）。

        使用 INSERT OR REPLACE：
          • 若 draw_number 不存在 → INSERT（新增）
          • 若 draw_number 已存在 → REPLACE（整筆覆蓋更新）

        為何用 executemany（批量）而不是逐筆 execute？
          executemany 只需一次 Python↔SQLite 的上下文切換，
          速度比迴圈快 10-50 倍（對 2,691 筆資料意義重大）。

        Args:
            draws: parse_draw() 回傳的 dict 清單

        Returns:
            int: 實際影響的資料列數（新增 + 更新的總和）
        """
        if not draws:
            return 0

        sql = """
        INSERT OR REPLACE INTO draws
            (draw_number, draw_date, n1, n2, n3, n4, n5, mega_number, jackpot_amount)
        VALUES
            (:draw_number, :draw_date, :n1, :n2, :n3, :n4, :n5, :mega_number, :jackpot_amount)
        """

        with get_connection(self.db_path) as conn:
            cursor = conn.executemany(sql, draws)
            affected = cursor.rowcount

        logger.debug(f"upsert_draws: {affected} 筆受影響")
        return affected

    def upsert_prizes(self, prizes: list[dict]) -> int:
        """
        批量新增或更新獎項明細。

        Args:
            prizes: parse_prizes() 回傳的 dict 清單

        Returns:
            int: 實際影響的資料列數
        """
        if not prizes:
            return 0

        sql = """
        INSERT OR REPLACE INTO draw_prizes
            (draw_number, prize_description, winner_count, prize_amount)
        VALUES
            (:draw_number, :prize_description, :winner_count, :prize_amount)
        """

        with get_connection(self.db_path) as conn:
            cursor = conn.executemany(sql, prizes)
            return cursor.rowcount

    def upsert_draw_features(self, features: list[dict]) -> int:
        """
        批量新增或更新每期衍生特徵。

        Args:
            features: feature_builder 計算後的 dict 清單

        Returns:
            int: 實際影響的資料列數
        """
        if not features:
            return 0

        sql = """
        INSERT OR REPLACE INTO draw_features (
            draw_number, white_sum, white_mean,
            odd_count, even_count, low_count, high_count,
            consecutive_pairs, number_range, unique_last_digits, mega_is_odd
        ) VALUES (
            :draw_number, :white_sum, :white_mean,
            :odd_count, :even_count, :low_count, :high_count,
            :consecutive_pairs, :number_range, :unique_last_digits, :mega_is_odd
        )
        """

        with get_connection(self.db_path) as conn:
            cursor = conn.executemany(sql, features)
            return cursor.rowcount

    # ────────────────────────────────────────────────────────────────
    # READ 操作
    # ────────────────────────────────────────────────────────────────

    def get_latest_draw_number(self) -> int:
        """
        取得資料庫中最新（最大）期號。
        若資料庫為空，回傳 0（表示沒有任何資料）。

        Returns:
            int: 最新期號
        """
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT MAX(draw_number) FROM draws").fetchone()
            return row[0] if row[0] is not None else 0

    def get_draw_count(self) -> int:
        """取得資料庫中總開獎筆數"""
        with get_connection(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM draws").fetchone()
            return row[0]

    def get_all_draws(self) -> pd.DataFrame:
        """
        讀取所有開獎資料，回傳 pandas DataFrame。

        為何回傳 DataFrame 而不是 list of dicts？
          特徵工程和 ML 模型都使用 pandas 操作，
          直接回傳 DataFrame 避免不必要的格式轉換。

        Returns:
            pd.DataFrame: 欄位包含 draw_number, draw_date, n1-n5, mega_number, jackpot_amount
                          按 draw_number 升序排列（時序分析必須保持時間順序）
        """
        sql = """
        SELECT draw_number, draw_date, n1, n2, n3, n4, n5, mega_number, jackpot_amount
        FROM draws
        ORDER BY draw_number ASC
        """
        with get_connection(self.db_path) as conn:
            df = pd.read_sql_query(sql, conn, parse_dates=["draw_date"])

        logger.debug(f"get_all_draws: 讀取 {len(df)} 筆資料")
        return df

    def get_draws_with_features(self) -> pd.DataFrame:
        """
        讀取開獎資料並 JOIN 衍生特徵，回傳寬表 DataFrame。
        用於 ML 模型訓練。

        Returns:
            pd.DataFrame: draws + draw_features 的合併結果
        """
        sql = """
        SELECT
            d.draw_number, d.draw_date,
            d.n1, d.n2, d.n3, d.n4, d.n5, d.mega_number,
            f.white_sum, f.white_mean,
            f.odd_count, f.even_count,
            f.low_count, f.high_count,
            f.consecutive_pairs, f.number_range,
            f.unique_last_digits, f.mega_is_odd
        FROM draws d
        LEFT JOIN draw_features f USING (draw_number)
        ORDER BY d.draw_number ASC
        """
        with get_connection(self.db_path) as conn:
            df = pd.read_sql_query(sql, conn, parse_dates=["draw_date"])

        logger.debug(f"get_draws_with_features: 讀取 {len(df)} 筆資料")
        return df

    def get_white_ball_stats(self) -> pd.DataFrame:
        """取得白球頻率統計表"""
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM white_ball_stats ORDER BY number", conn
            )

    def get_mega_ball_stats(self) -> pd.DataFrame:
        """取得 Mega 球頻率統計表"""
        with get_connection(self.db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM mega_ball_stats ORDER BY number", conn
            )
