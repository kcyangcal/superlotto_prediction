"""
runner.py — 爬蟲執行主流程
============================
控制兩種抓取模式：
  1. 全量抓取（scrape_all）：首次建立資料庫時使用，抓取所有歷史資料
  2. 增量更新（scrape_latest）：每週開獎後執行，只抓取比 DB 最新期號更新的資料

設計說明：
  兩種模式共用同一套 client + parser + repository，
  差別只在「要抓幾頁」的邏輯控制。
"""

import logging
import math

from config.settings import PAGE_SIZE, LOG_LEVEL, LOG_FORMAT
from src.scraper.client import LotteryApiClient
from src.scraper.parser import parse_page
from src.database.repository import DrawRepository

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def scrape_all(
    client: LotteryApiClient,
    repo: DrawRepository,
    page_size: int = PAGE_SIZE,
) -> int:
    """
    全量抓取：從第 1 頁到最後一頁，把所有歷史開獎資料存入資料庫。

    使用 UPSERT（INSERT OR REPLACE），所以重複執行是安全的——
    已存在的資料會被覆蓋更新，新資料會被新增。

    Args:
        client:    LotteryApiClient 實例
        repo:      DrawRepository 實例
        page_size: 每頁筆數

    Returns:
        int: 成功處理的開獎筆數
    """
    logger.info("▶ 開始全量抓取 SuperLotto Plus 歷史資料...")

    # 先抓第 1 頁，取得總資料筆數，計算總頁數
    first_page = client.fetch_page(page=1)
    total_draws = first_page.get("TotalDrawCount", 0)
    total_pages = math.ceil(total_draws / page_size)
    logger.info(f"  總計 {total_draws} 筆資料，分 {total_pages} 頁（每頁 {page_size} 筆）")

    total_saved = 0

    # 第 1 頁已經抓了，先處理它
    draws_list, prizes_list = parse_page(first_page)
    saved = repo.upsert_draws(draws_list)
    repo.upsert_prizes(prizes_list)
    total_saved += saved
    logger.info(f"  第 1/{total_pages} 頁：新增/更新 {saved} 筆")

    # 從第 2 頁開始繼續抓取
    for page in range(2, total_pages + 1):
        try:
            api_response = client.fetch_page(page=page)
            draws_list, prizes_list = parse_page(api_response)

            if not draws_list:
                logger.info(f"  第 {page} 頁回傳空資料，抓取完成")
                break

            saved = repo.upsert_draws(draws_list)
            repo.upsert_prizes(prizes_list)
            total_saved += saved
            logger.info(f"  第 {page}/{total_pages} 頁：新增/更新 {saved} 筆（累計 {total_saved} 筆）")

        except Exception as e:
            logger.error(f"  第 {page} 頁抓取失敗：{e}（跳過繼續）")

    logger.info(f"✅ 全量抓取完成，共處理 {total_saved} 筆資料")
    return total_saved


def scrape_latest(
    client: LotteryApiClient,
    repo: DrawRepository,
    check_pages: int = 2,
) -> int:
    """
    增量更新：只抓取比資料庫最新期號更新的資料。

    策略：
      1. 先查 DB 中最新的期號（latest_in_db）
      2. 抓取 API 前 check_pages 頁
      3. 只保留 draw_number > latest_in_db 的資料
      4. 存入資料庫

    Args:
        client:      LotteryApiClient 實例
        repo:        DrawRepository 實例
        check_pages: 要檢查幾頁（預設 2 頁 = 最近 200 期，遠超任何增量需求）

    Returns:
        int: 新增的開獎筆數（0 表示已是最新）
    """
    latest_in_db = repo.get_latest_draw_number()
    logger.info(f"▶ 開始增量更新（DB 最新期號：{latest_in_db}）")

    new_draws  = []
    new_prizes = []

    for page in range(1, check_pages + 1):
        api_response = client.fetch_page(page=page)
        draws_list, prizes_list = parse_page(api_response)

        # 只保留比 DB 更新的資料
        new_in_page = [d for d in draws_list if d["draw_number"] > latest_in_db]
        corresponding_prizes = [
            p for p in prizes_list
            if p["draw_number"] > latest_in_db
        ]

        new_draws.extend(new_in_page)
        new_prizes.extend(corresponding_prizes)

        # 如果這一頁的資料都比 DB 舊了，不需要繼續抓
        if len(new_in_page) < len(draws_list):
            break

    if not new_draws:
        logger.info("✅ 資料庫已是最新，無新資料")
        return 0

    saved = repo.upsert_draws(new_draws)
    repo.upsert_prizes(new_prizes)
    logger.info(f"✅ 增量更新完成，新增 {saved} 筆資料")
    return saved
