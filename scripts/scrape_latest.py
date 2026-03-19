"""
scrape_latest.py — 增量更新最新開獎資料
=========================================
每週三/六開獎後執行，只抓取比資料庫最新期號更新的資料。
通常只新增 1–2 筆，執行非常快速。

使用方式：
  python scripts/scrape_latest.py

建議：設定為每週排程任務（Windows 工作排程器）：
  每週三/六晚上 8 點執行（加州開獎時間約為 7:57 PM PT）
"""

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.scraper.client import LotteryApiClient
from src.scraper.runner import scrape_latest
from src.database.repository import DrawRepository

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    if not DB_PATH.exists():
        logger.error("找不到資料庫！請先執行 init_db.py 和 scrape_all.py")
        sys.exit(1)

    repo = DrawRepository(DB_PATH)
    logger.info(f"DB 最新期號：{repo.get_latest_draw_number()}")

    with LotteryApiClient() as client:
        new_count = scrape_latest(client, repo)

    if new_count > 0:
        logger.info(f"新增 {new_count} 筆資料，建議重新執行 build_features.py 更新特徵")
    else:
        logger.info("無新資料，資料庫已是最新")


if __name__ == "__main__":
    main()
