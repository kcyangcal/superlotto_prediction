"""
build_features.py — 重建特徵資料表
=====================================
讀取 draws 表中的所有開獎資料，重新計算所有衍生特徵，
並將結果寫入 draw_features 資料表。

何時執行：
  • 首次抓完全部歷史資料後
  • 每次 scrape_latest.py 新增資料後
  • 修改特徵工程邏輯後

使用方式：
  python scripts/build_features.py
"""

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.database.repository import DrawRepository
from src.features.feature_builder import build_draw_features

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    if not DB_PATH.exists():
        logger.error("找不到資料庫！請先執行 init_db.py 和 scrape_all.py")
        sys.exit(1)

    repo = DrawRepository(DB_PATH)
    draw_count = repo.get_draw_count()
    logger.info(f"資料庫共有 {draw_count} 筆開獎資料")

    if draw_count == 0:
        logger.error("資料庫為空！請先執行 scrape_all.py")
        sys.exit(1)

    # 讀取所有資料
    logger.info("正在讀取開獎資料...")
    df = repo.get_all_draws()

    # 計算衍生特徵
    features_df = build_draw_features(df)

    # 寫回資料庫
    logger.info("正在寫入 draw_features 資料表...")
    records = features_df.to_dict(orient="records")
    saved = repo.upsert_draw_features(records)
    logger.info(f"✅ 特徵計算完成，共寫入 {saved} 筆")


if __name__ == "__main__":
    main()
