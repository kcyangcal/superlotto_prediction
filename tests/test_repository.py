"""
test_repository.py — 資料庫操作單元測試
=========================================
使用 SQLite in-memory 資料庫進行測試，不依賴實際的 DB 檔案。
每個測試函式都有獨立的乾淨資料庫，避免測試間互相干擾。

執行方式：
    pytest tests/test_repository.py -v
"""

import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.database.connection import initialize_database, get_connection
from src.database.repository import DrawRepository

# in-memory DB 路徑（:memory: 讓 SQLite 建立一個不落地的暫時資料庫）
IN_MEMORY = Path(":memory:")


@pytest.fixture
def repo(tmp_path):
    """
    每個測試函式都會得到一個全新的空白資料庫。
    使用 tmp_path（pytest 內建的暫存目錄）而非 :memory:，
    因為不同的 connection 呼叫會共用同一個檔案。
    """
    db_path = tmp_path / "test.db"
    initialize_database(db_path)
    return DrawRepository(db_path)


SAMPLE_DRAWS = [
    dict(draw_number=1001, draw_date="2024-01-01",
         n1=3, n2=7, n3=12, n4=25, n5=36, mega_number=5, jackpot_amount=None),
    dict(draw_number=1002, draw_date="2024-01-04",
         n1=1, n2=10, n3=22, n4=33, n5=44, mega_number=15, jackpot_amount=5000000.0),
    dict(draw_number=1003, draw_date="2024-01-07",
         n1=2, n2=4, n3=8,  n4=16, n5=32, mega_number=10, jackpot_amount=None),
]


class TestUpsertDraws:
    """upsert_draws 方法測試"""

    def test_insert_new_draws(self, repo):
        """新資料應成功插入"""
        saved = repo.upsert_draws(SAMPLE_DRAWS)
        assert saved == 3

    def test_count_after_insert(self, repo):
        """插入後 get_draw_count 應回傳正確數量"""
        repo.upsert_draws(SAMPLE_DRAWS)
        assert repo.get_draw_count() == 3

    def test_upsert_no_duplicate(self, repo):
        """重複插入相同資料不應產生重複筆數"""
        repo.upsert_draws(SAMPLE_DRAWS)
        repo.upsert_draws(SAMPLE_DRAWS)  # 再插入一次
        assert repo.get_draw_count() == 3  # 應仍是 3，而不是 6

    def test_empty_list(self, repo):
        """空清單不應報錯，回傳 0"""
        saved = repo.upsert_draws([])
        assert saved == 0


class TestGetLatestDrawNumber:
    """get_latest_draw_number 方法測試"""

    def test_empty_db_returns_zero(self, repo):
        """空資料庫應回傳 0"""
        assert repo.get_latest_draw_number() == 0

    def test_returns_max_draw_number(self, repo):
        """應回傳最大期號"""
        repo.upsert_draws(SAMPLE_DRAWS)
        assert repo.get_latest_draw_number() == 1003


class TestGetAllDraws:
    """get_all_draws 方法測試"""

    def test_returns_dataframe(self, repo):
        """應回傳 pandas DataFrame"""
        import pandas as pd
        repo.upsert_draws(SAMPLE_DRAWS)
        df = repo.get_all_draws()
        assert hasattr(df, "columns"), "應回傳 DataFrame"

    def test_correct_row_count(self, repo):
        """DataFrame 行數應與插入資料筆數一致"""
        repo.upsert_draws(SAMPLE_DRAWS)
        df = repo.get_all_draws()
        assert len(df) == 3

    def test_ordered_by_draw_number(self, repo):
        """回傳資料應按 draw_number 升序排列"""
        # 以相反順序插入
        repo.upsert_draws(list(reversed(SAMPLE_DRAWS)))
        df = repo.get_all_draws()
        draw_numbers = df["draw_number"].tolist()
        assert draw_numbers == sorted(draw_numbers), "應按期號升序排列"

    def test_columns_exist(self, repo):
        """必要欄位應都存在"""
        repo.upsert_draws(SAMPLE_DRAWS)
        df = repo.get_all_draws()
        for col in ["draw_number", "draw_date", "n1", "n2", "n3", "n4", "n5", "mega_number"]:
            assert col in df.columns, f"欄位 {col} 不存在"


class TestDatabaseInitialization:
    """資料庫初始化測試"""

    def test_all_tables_created(self, tmp_path):
        """初始化後應建立所有 6 張資料表"""
        db_path = tmp_path / "init_test.db"
        initialize_database(db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0] for row in
            conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        conn.close()

        expected = {
            "draws", "draw_prizes", "white_ball_stats",
            "mega_ball_stats", "draw_features", "number_gap_history"
        }
        missing = expected - tables
        assert not missing, f"以下資料表未建立：{missing}"

    def test_idempotent(self, tmp_path):
        """重複初始化不應報錯（IF NOT EXISTS）"""
        db_path = tmp_path / "idempotent_test.db"
        initialize_database(db_path)
        initialize_database(db_path)  # 第二次應不報錯
