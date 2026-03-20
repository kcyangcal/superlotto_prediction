"""
scrape_history.py — 從 lotterycorner.com 補抓完整歷史資料
============================================================
CA Lottery 官方 API 只提供最近 106 筆，
此腳本從 lotterycorner.com 抓取 2000–2025 年所有 HTML 表格資料
並以 UPSERT 方式補進資料庫（不會覆蓋 CA Lottery 官方已有的資料）。

資料範圍：2000–2025（SuperLotto Plus 開始於 2000 年 6 月 7 日）
預計筆數：約 2,600 筆

使用方式：
    python scripts/scrape_history.py
"""

import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import requests
from bs4 import BeautifulSoup

from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT, REQUEST_DELAY
from src.database.repository import DrawRepository

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.lotterycorner.com/ca/superlotto-plus/{year}"
# SuperLotto Plus 正式開始日期
START_YEAR = 2000
END_YEAR   = 2025


def parse_year_page(html: str, year: int) -> list[dict]:
    """
    解析 lotterycorner.com 某年份頁面的 HTML 表格。

    表格欄位格式：
        Date       | Result                      | Jackpot
        "Dec 28"   | "6 14 22 28 35 24 Mega Ball" | "$11M"

    Result 欄位：白球1 白球2 白球3 白球4 白球5 Mega球 "Mega Ball"
    """
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        logger.warning(f"{year}: 找不到資料表格")
        return []

    rows = table.find_all("tr")[1:]  # 跳過 header 行
    draws = []

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue

        # ── 日期解析 ──────────────────────────────────────────────────────────
        date_text = cells[0].get_text(strip=True)
        if not date_text:
            continue
        try:
            draw_date = datetime.strptime(date_text, "%B %d, %Y").date()
        except ValueError:
            try:
                draw_date = datetime.strptime(f"{date_text} {year}", "%B %d %Y").date()
            except ValueError:
                logger.debug(f"無法解析日期：{date_text!r}，略過")
                continue

        # ── 號碼解析 ──────────────────────────────────────────────────────────
        result_text = cells[1].get_text(" ", strip=True)
        # 移除 "Mega Ball" 文字後，提取所有數字
        result_clean = result_text.replace("Mega Ball", "").strip()
        numbers = [int(n) for n in re.findall(r"\d+", result_clean)]

        if len(numbers) != 6:
            logger.debug(f"{draw_date}：號碼數量異常 {numbers}，略過")
            continue

        white_balls = sorted(numbers[:5])
        mega_ball   = numbers[5]

        # 基本範圍驗證
        if not all(1 <= n <= 47 for n in white_balls):
            logger.debug(f"{draw_date}：白球超出範圍 {white_balls}，略過")
            continue
        if not (1 <= mega_ball <= 27):
            logger.debug(f"{draw_date}：Mega 球超出範圍 {mega_ball}，略過")
            continue

        # ── 獎金解析（允許為 None）────────────────────────────────────────────
        jackpot = None
        if len(cells) >= 3:
            jackpot_text = cells[2].get_text(strip=True).replace("$", "").replace(",", "")
            # 處理 "10 Million" / "10M" 等格式
            m = re.search(r"([\d.]+)\s*(million|m|billion|b)?", jackpot_text, re.I)
            if m:
                val = float(m.group(1))
                unit = (m.group(2) or "").lower()
                if unit in ("million", "m"):
                    val *= 1_000_000
                elif unit in ("billion", "b"):
                    val *= 1_000_000_000
                jackpot = val

        draws.append({
            "draw_number":    None,      # lotterycorner 沒有期號，用 None
            "draw_date":      str(draw_date),
            "n1": white_balls[0], "n2": white_balls[1],
            "n3": white_balls[2], "n4": white_balls[3],
            "n5": white_balls[4],
            "mega_number":    mega_ball,
            "jackpot_amount": jackpot,
        })

    return draws


def assign_draw_numbers(draws: list[dict], repo: DrawRepository) -> list[dict]:
    """
    lotterycorner.com 沒有官方期號，
    此函式根據開獎日期與資料庫中已有的期號進行對應，
    對於資料庫中已存在的日期，跳過（避免覆蓋官方資料）。
    對於新日期，分配一個臨時負數期號（負數表示非官方來源）。

    注意：CA Lottery 官方資料（draw_number > 0）永遠優先，
          此處的負數期號只是佔位符，等 scrape_latest 取得官方期號後自動被 UPSERT 覆蓋。
    """
    import sqlite3
    conn = sqlite3.connect(str(DB_PATH))

    # 取得資料庫中已有的所有日期
    existing_dates = {
        row[0] for row in
        conn.execute("SELECT draw_date FROM draws").fetchall()
    }

    # 取得最小已有期號（用於分配負數期號起點）
    min_num = conn.execute("SELECT MIN(draw_number) FROM draws").fetchone()[0] or 0
    conn.close()

    temp_num = min(min_num, 0) - 1
    result = []
    for draw in reversed(draws):  # 反轉使最舊的先分配
        if draw["draw_date"] in existing_dates:
            continue   # 已有此日期，略過
        draw["draw_number"] = temp_num
        temp_num -= 1
        result.append(draw)

    return result


def main():
    if not DB_PATH.exists():
        logger.error("找不到資料庫！請先執行 init_db.py")
        sys.exit(1)

    repo   = DrawRepository(DB_PATH)
    before = repo.get_draw_count()
    logger.info(f"執行前：資料庫共 {before} 筆")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept":     "text/html,application/xhtml+xml,*/*",
        "Referer":    "https://www.lotterycorner.com",
    })

    all_draws = []

    for year in range(END_YEAR, START_YEAR - 1, -1):
        url = BASE_URL.format(year=year)
        try:
            time.sleep(REQUEST_DELAY)
            r = session.get(url, timeout=30)
            r.raise_for_status()
            year_draws = parse_year_page(r.text, year)
            all_draws.extend(year_draws)
            logger.info(f"  {year}：解析 {len(year_draws)} 筆")
        except Exception as e:
            logger.warning(f"  {year}：抓取失敗 ({e})，略過")

    logger.info(f"共解析 {len(all_draws)} 筆（含可能重複）")

    # 過濾掉資料庫中已有的日期
    new_draws = assign_draw_numbers(all_draws, repo)
    logger.info(f"排除已有日期後：{len(new_draws)} 筆待新增")

    if new_draws:
        saved = repo.upsert_draws(new_draws)
        logger.info(f"✅ 補充完成，新增 {saved} 筆歷史資料")
    else:
        logger.info("無新資料需要補充")

    after = repo.get_draw_count()
    logger.info(f"執行後：資料庫共 {after} 筆")


if __name__ == "__main__":
    main()
