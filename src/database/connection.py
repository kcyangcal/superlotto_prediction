"""
connection.py — SQLite 資料庫連線管理
======================================
提供 context manager 方式管理 SQLite 連線，確保：
  • 每次操作後自動 commit（或在例外時 rollback）
  • 連線使用完畢後自動關閉，不會發生資源洩漏
  • 啟用 foreign key 約束（SQLite 預設是關閉的！）
  • 回傳 Row 物件，支援以欄位名稱存取（像 dict 一樣方便）
"""

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path

# 從全域設定讀取 DB 路徑，避免硬編碼
from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT

# 設定這個模組的日誌器
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def _get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """
    建立並回傳一個 SQLite 連線，套用推薦設定。

    Args:
        db_path: 資料庫檔案路徑，預設使用 settings.DB_PATH

    Returns:
        已設定好的 sqlite3.Connection 物件
    """
    conn = sqlite3.connect(str(db_path))

    # row_factory 讓查詢結果可以用欄位名稱存取
    # 例如：row['draw_number'] 而不是 row[0]
    conn.row_factory = sqlite3.Row

    # 啟用外鍵約束（SQLite 每次連線都要重新啟用）
    conn.execute("PRAGMA foreign_keys = ON")

    # WAL 模式：讀取不阻塞寫入，適合多個腳本同時讀寫的場景
    conn.execute("PRAGMA journal_mode = WAL")

    return conn


@contextmanager
def get_connection(db_path: Path = DB_PATH):
    """
    提供 SQLite 連線的 context manager。

    使用範例：
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM draws").fetchall()

    設計說明：
      • 正常離開 with 區塊 → 自動 commit
      • 發生例外 → 自動 rollback 並重新拋出例外
      • 無論如何 → 自動關閉連線

    Args:
        db_path: 資料庫路徑（測試時可傳入 :memory: 或暫存路徑）

    Yields:
        sqlite3.Connection
    """
    conn = _get_connection(db_path)
    try:
        yield conn
        conn.commit()
        logger.debug("Transaction committed successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Transaction rolled back due to error: {e}")
        raise
    finally:
        conn.close()
        logger.debug("Database connection closed.")


def initialize_database(db_path: Path = DB_PATH, schema_path: Path = None) -> None:
    """
    執行 schema.sql，建立所有資料表（若尚未存在）。

    這個函式只應在 scripts/init_db.py 呼叫一次，
    或是在測試環境初始化 in-memory DB 時使用。

    Args:
        db_path:     目標資料庫路徑
        schema_path: schema.sql 的路徑（預設自動尋找）
    """
    if schema_path is None:
        # 預設路徑：與 connection.py 同目錄下的 schema.sql
        schema_path = Path(__file__).parent / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"找不到 schema 檔案：{schema_path}")

    schema_sql = schema_path.read_text(encoding="utf-8")

    with get_connection(db_path) as conn:
        # executescript 支援一次執行多個 SQL 陳述式（以分號分隔）
        conn.executescript(schema_sql)

    logger.info(f"資料庫初始化完成：{db_path}")
