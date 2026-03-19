"""
scrape_all.py — 全量抓取所有歷史開獎資料
==========================================
首次建立資料庫時執行，抓取 CA Lottery SuperLotto Plus 的全部歷史資料。
預計約 2,691 筆，分 27 頁，每頁間隔 1.5 秒，總計約 40 秒完成。

前提條件：
  請先執行 python scripts/init_db.py 建立資料庫

使用方式：
  python scripts/scrape_all.py
"""

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.scraper.client import LotteryApiClient
from src.scraper.runner import scrape_all
from src.database.repository import DrawRepository

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def main():
    if not DB_PATH.exists():
        logger.error("找不到資料庫檔案！請先執行：python scripts/init_db.py")
        sys.exit(1)

    logger.info(f"目標資料庫：{DB_PATH}")
    repo = DrawRepository(DB_PATH)
    before_count = repo.get_draw_count()
    logger.info(f"執行前：資料庫已有 {before_count} 筆資料")

    with LotteryApiClient() as client:
        total = scrape_all(client, repo)

    after_count = repo.get_draw_count()
    logger.info(f"執行後：資料庫共有 {after_count} 筆資料（新增/更新 {after_count - before_count} 筆）")


if __name__ == "__main__":
    main()
