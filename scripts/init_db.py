"""
init_db.py — 資料庫初始化腳本
===============================
功能：執行 schema.sql 建立所有資料表（若已存在則略過）。
執行時機：首次設置專案時執行一次即可。

使用方式（在專案根目錄執行）：
    python scripts/init_db.py

如需清空重建（危險！所有資料將遺失）：
    python scripts/init_db.py --reset
"""

import argparse
import logging
import sys
from pathlib import Path

# 將專案根目錄加入 Python path，使得 import 能正確找到 src/ 和 config/
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.database.connection import initialize_database

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="初始化 SuperLotto Plus 資料庫")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="⚠️  刪除現有資料庫並重新建立（所有資料將遺失！）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 危險操作確認 ──────────────────────────────────────────────────────────
    if args.reset and DB_PATH.exists():
        confirm = input(
            f"\n⚠️  即將刪除 {DB_PATH} 並重新建立資料庫！\n"
            "所有歷史資料將永久遺失！\n"
            "輸入 'yes' 確認繼續：\n> "
        ).strip()
        if confirm != "yes":
            logger.info("操作已取消。")
            return
        DB_PATH.unlink()
        logger.warning(f"已刪除現有資料庫：{DB_PATH}")

    # ── 建立資料庫 ────────────────────────────────────────────────────────────
    if DB_PATH.exists():
        logger.info(f"資料庫已存在：{DB_PATH}（若需重建請加 --reset 參數）")
    else:
        logger.info(f"正在建立資料庫：{DB_PATH}")

    initialize_database(DB_PATH)

    # ── 驗證：確認所有資料表都已建立 ──────────────────────────────────────────
    import sqlite3

    conn = sqlite3.connect(str(DB_PATH))
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()

    table_names = [t[0] for t in tables]
    expected_tables = {
        "draws",
        "draw_prizes",
        "white_ball_stats",
        "mega_ball_stats",
        "draw_features",
        "number_gap_history",
    }

    missing = expected_tables - set(table_names)
    if missing:
        logger.error(f"以下資料表建立失敗：{missing}")
        sys.exit(1)

    logger.info(f"✅ 資料庫初始化成功！共建立 {len(table_names)} 張資料表：")
    for name in sorted(table_names):
        logger.info(f"   • {name}")


if __name__ == "__main__":
    main()
