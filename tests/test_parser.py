"""
test_parser.py — 解析器單元測試
================================
用已知的開獎資料驗證 parser.py 的正確性。
所有測試使用「手工構造」的資料，不依賴網路或資料庫。

執行方式：
    pytest tests/test_parser.py -v
"""

import sys
from pathlib import Path

# 確保 import 能找到 src/ 和 config/
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.scraper.parser import parse_draw, parse_prizes, ParseError


# ── 測試用固定資料 ────────────────────────────────────────────────────────────
# 模擬 API 回傳的單筆開獎 JSON
SAMPLE_RAW_DRAW = {
    "DrawNumber": 4065,
    "DrawDate":   "2026-03-18T07:00:00",
    "WinningNumbers": {
        "0": {"Number": "1",  "IsSpecial": False},
        "1": {"Number": "26", "IsSpecial": False},
        "2": {"Number": "35", "IsSpecial": False},
        "3": {"Number": "40", "IsSpecial": False},
        "4": {"Number": "46", "IsSpecial": False},
        "5": {"Number": "7",  "IsSpecial": True},   # Mega ball
    },
    "JackpotAmount": 5000000.0,
    "PrizePayoutDetails": [
        {"PrizeDescription": "5 + Mega", "NumberOfWinners": 0, "PrizeAmount": 5000000.0},
        {"PrizeDescription": "5 Numbers", "NumberOfWinners": 1, "PrizeAmount": 30000.0},
        {"PrizeDescription": "4 + Mega", "NumberOfWinners": 3, "PrizeAmount": 1500.0},
    ],
}


class TestParseDrawSuccess:
    """正常情況下的解析測試"""

    def test_draw_number(self):
        """期號應正確解析"""
        result = parse_draw(SAMPLE_RAW_DRAW)
        assert result["draw_number"] == 4065

    def test_draw_date(self):
        """日期應解析為 YYYY-MM-DD 字串"""
        result = parse_draw(SAMPLE_RAW_DRAW)
        assert result["draw_date"] == "2026-03-18"

    def test_white_balls_sorted(self):
        """5 顆白球應排序後依序存入 n1-n5"""
        result = parse_draw(SAMPLE_RAW_DRAW)
        assert result["n1"] == 1
        assert result["n2"] == 26
        assert result["n3"] == 35
        assert result["n4"] == 40
        assert result["n5"] == 46

    def test_mega_ball_identified(self):
        """IsSpecial=True 的號碼應識別為 Mega ball"""
        result = parse_draw(SAMPLE_RAW_DRAW)
        assert result["mega_number"] == 7

    def test_jackpot_amount(self):
        """頭獎金額應正確解析"""
        result = parse_draw(SAMPLE_RAW_DRAW)
        assert result["jackpot_amount"] == 5000000.0

    def test_white_balls_in_range(self):
        """所有白球應在 1–47 範圍內"""
        result = parse_draw(SAMPLE_RAW_DRAW)
        for key in ["n1", "n2", "n3", "n4", "n5"]:
            assert 1 <= result[key] <= 47, f"{key}={result[key]} 超出範圍"

    def test_mega_in_range(self):
        """Mega ball 應在 1–27 範圍內"""
        result = parse_draw(SAMPLE_RAW_DRAW)
        assert 1 <= result["mega_number"] <= 27


class TestParseDrawErrors:
    """異常資料應拋出 ParseError"""

    def test_missing_draw_number(self):
        """缺少期號應拋出 ParseError"""
        bad_data = {k: v for k, v in SAMPLE_RAW_DRAW.items() if k != "DrawNumber"}
        with pytest.raises(ParseError):
            parse_draw(bad_data)

    def test_white_ball_out_of_range(self):
        """白球號碼超出 1–47 應拋出 ParseError"""
        bad_data = {
            **SAMPLE_RAW_DRAW,
            "WinningNumbers": {
                "0": {"Number": "99", "IsSpecial": False},  # 超出範圍！
                "1": {"Number": "2",  "IsSpecial": False},
                "2": {"Number": "3",  "IsSpecial": False},
                "3": {"Number": "4",  "IsSpecial": False},
                "4": {"Number": "5",  "IsSpecial": False},
                "5": {"Number": "7",  "IsSpecial": True},
            }
        }
        with pytest.raises(ParseError):
            parse_draw(bad_data)

    def test_wrong_white_ball_count(self):
        """白球數量不等於 5 應拋出 ParseError"""
        bad_data = {
            **SAMPLE_RAW_DRAW,
            "WinningNumbers": {
                "0": {"Number": "1",  "IsSpecial": False},
                "1": {"Number": "2",  "IsSpecial": False},
                # 只有 2 顆白球
                "2": {"Number": "7",  "IsSpecial": True},
            }
        }
        with pytest.raises(ParseError):
            parse_draw(bad_data)

    def test_no_mega_ball(self):
        """沒有 Mega ball 應拋出 ParseError"""
        bad_data = {
            **SAMPLE_RAW_DRAW,
            "WinningNumbers": {
                str(i): {"Number": str(i+1), "IsSpecial": False}
                for i in range(5)
            }
        }
        with pytest.raises(ParseError):
            parse_draw(bad_data)


class TestParsePrizes:
    """獎項明細解析測試"""

    def test_prize_count(self):
        """應解析出正確數量的獎項"""
        prizes = parse_prizes(SAMPLE_RAW_DRAW)
        assert len(prizes) == 3

    def test_prize_draw_number(self):
        """所有獎項的 draw_number 應對應正確期號"""
        prizes = parse_prizes(SAMPLE_RAW_DRAW)
        assert all(p["draw_number"] == 4065 for p in prizes)

    def test_jackpot_prize(self):
        """頭獎資料應正確解析"""
        prizes = parse_prizes(SAMPLE_RAW_DRAW)
        jackpot = next(p for p in prizes if p["prize_description"] == "5 + Mega")
        assert jackpot["prize_amount"] == 5000000.0
        assert jackpot["winner_count"] == 0

    def test_no_prize_data(self):
        """無獎項資料時應回傳空清單"""
        raw = {**SAMPLE_RAW_DRAW}
        raw.pop("PrizePayoutDetails", None)
        prizes = parse_prizes(raw)
        assert prizes == []
