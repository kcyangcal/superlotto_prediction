"""
parser.py — API 回應解析與資料清洗
=====================================
負責把 CA Lottery API 回傳的原始 JSON 轉換為標準化的 Python dict，
並進行資料驗證，確保進入資料庫的資料品質。

API 回應的 WinningNumbers 結構範例：
    {
      "WinningNumbers": {
        "0": {"Number": "3",  "IsSpecial": false},
        "1": {"Number": "12", "IsSpecial": false},
        "2": {"Number": "25", "IsSpecial": false},
        "3": {"Number": "35", "IsSpecial": false},
        "4": {"Number": "46", "IsSpecial": false},
        "5": {"Number": "7",  "IsSpecial": true}   ← Mega ball
      }
    }

IsSpecial=True 代表 Mega ball，這是識別的唯一依據。
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    MEGA_BALL_MIN, MEGA_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ParseError(Exception):
    """解析或驗證失敗時拋出的自定例外"""
    pass


def parse_draw(raw: dict) -> dict:
    """
    解析 API 中單筆開獎資料，回傳標準化 dict。

    Args:
        raw: API 回傳的單筆開獎 JSON dict

    Returns:
        標準化開獎資料 dict，欄位名稱與 draws 資料表一致：
        {
            draw_number:   int,
            draw_date:     str (YYYY-MM-DD),
            n1~n5:         int (排序後的白球，由小到大),
            mega_number:   int,
            jackpot_amount: float or None,
        }

    Raises:
        ParseError: 當資料無法解析或驗證失敗時
    """
    try:
        draw_number = int(raw["DrawNumber"])
    except (KeyError, ValueError, TypeError) as e:
        raise ParseError(f"無法解析期號：{e}，原始資料：{raw}")

    # ── 日期解析 ──────────────────────────────────────────────────────────────
    # API 的日期格式："2026-03-18T07:00:00" 或 "2026-03-18T00:00:00+00:00"
    try:
        raw_date = raw["DrawDate"]
        # 移除時區資訊後解析（只取日期部分）
        draw_date = datetime.fromisoformat(raw_date.split("T")[0]).date()
    except (KeyError, ValueError) as e:
        raise ParseError(f"期號 {draw_number}：日期解析失敗 ({e})")

    # ── 號碼解析 ──────────────────────────────────────────────────────────────
    try:
        winning_numbers = raw["WinningNumbers"]
    except KeyError:
        raise ParseError(f"期號 {draw_number}：找不到 WinningNumbers 欄位")

    white_balls = []
    mega_ball   = None

    for key, ball_data in winning_numbers.items():
        try:
            number     = int(ball_data["Number"])
            is_special = bool(ball_data["IsSpecial"])
        except (KeyError, ValueError, TypeError) as e:
            raise ParseError(f"期號 {draw_number}：號碼解析失敗 (key={key})：{e}")

        if is_special:
            # IsSpecial=True → Mega ball
            if mega_ball is not None:
                raise ParseError(f"期號 {draw_number}：發現多個 Mega ball！")
            mega_ball = number
        else:
            white_balls.append(number)

    # ── 資料驗證 ──────────────────────────────────────────────────────────────
    if len(white_balls) != 5:
        raise ParseError(f"期號 {draw_number}：白球數量異常（{len(white_balls)} 顆）")

    if mega_ball is None:
        raise ParseError(f"期號 {draw_number}：找不到 Mega ball")

    # 驗證號碼範圍
    for n in white_balls:
        if not (WHITE_BALL_MIN <= n <= WHITE_BALL_MAX):
            raise ParseError(f"期號 {draw_number}：白球 {n} 超出範圍 [{WHITE_BALL_MIN}, {WHITE_BALL_MAX}]")

    if not (MEGA_BALL_MIN <= mega_ball <= MEGA_BALL_MAX):
        raise ParseError(f"期號 {draw_number}：Mega ball {mega_ball} 超出範圍 [{MEGA_BALL_MIN}, {MEGA_BALL_MAX}]")

    # 白球排序（由小到大）：方便後續查詢與特徵計算
    white_balls.sort()

    # ── 獎金解析（允許為 None）────────────────────────────────────────────────
    jackpot_amount = _parse_jackpot(raw)

    return {
        "draw_number":    draw_number,
        "draw_date":      str(draw_date),
        "n1":             white_balls[0],
        "n2":             white_balls[1],
        "n3":             white_balls[2],
        "n4":             white_balls[3],
        "n5":             white_balls[4],
        "mega_number":    mega_ball,
        "jackpot_amount": jackpot_amount,
    }


def parse_prizes(raw: dict) -> list[dict]:
    """
    解析單期的獎項明細清單。

    Returns:
        list of dicts，每個 dict 對應 draw_prizes 資料表的一筆資料：
        [
            {draw_number, prize_description, winner_count, prize_amount},
            ...
        ]
    """
    draw_number = int(raw["DrawNumber"])
    prizes = []

    prize_list = raw.get("PrizePayoutDetails") or raw.get("Prizes") or []
    for prize in prize_list:
        try:
            prizes.append({
                "draw_number":       draw_number,
                "prize_description": str(prize.get("PrizeDescription", "")),
                "winner_count":      _safe_int(prize.get("NumberOfWinners")),
                "prize_amount":      _safe_float(prize.get("PrizeAmount")),
            })
        except Exception as e:
            logger.warning(f"期號 {draw_number}：獎項解析失敗：{e}，略過此獎項")

    return prizes


def parse_page(api_response: dict) -> tuple[list[dict], list[dict]]:
    """
    解析整頁 API 回應，回傳 (draws_list, prizes_list)。

    Args:
        api_response: fetch_page() 回傳的完整 JSON dict

    Returns:
        draws_list:  標準化開獎資料清單（可直接傳給 repository.upsert_draws）
        prizes_list: 獎項明細清單（可直接傳給 repository.upsert_prizes）
    """
    raw_draws = api_response.get("PreviousDraws", [])
    draws_list  = []
    prizes_list = []
    errors      = 0

    for raw in raw_draws:
        try:
            draws_list.append(parse_draw(raw))
            prizes_list.extend(parse_prizes(raw))
        except ParseError as e:
            logger.warning(f"解析失敗，略過此筆：{e}")
            errors += 1

    if errors:
        logger.warning(f"本頁共 {errors} 筆解析失敗（已略過）")

    return draws_list, prizes_list


# ── 私有輔助函式 ──────────────────────────────────────────────────────────────

def _parse_jackpot(raw: dict) -> Optional[float]:
    """嘗試從多個可能的欄位名稱中解析頭獎金額"""
    for key in ("JackpotAmount", "Jackpot", "jackpot_amount"):
        val = raw.get(key)
        if val is not None:
            return _safe_float(val)
    return None


def _safe_int(value) -> Optional[int]:
    """安全地將值轉為 int，失敗則回傳 None"""
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def _safe_float(value) -> Optional[float]:
    """安全地將值轉為 float，失敗則回傳 None"""
    try:
        # 處理帶逗號的金額字串，例如 "1,000,000"
        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "").strip()
        return float(value) if value not in (None, "", "N/A") else None
    except (ValueError, TypeError):
        return None
