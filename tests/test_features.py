"""
test_features.py — 特徵工程單元測試
======================================
手算驗證特徵計算函式的正確性。
全部使用固定資料，不依賴資料庫。

執行方式：
    pytest tests/test_features.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from src.features.pattern_stats import add_pattern_features
from src.features.base_stats import (
    compute_white_ball_frequency,
    compute_gap_stats,
)


# ── 測試用 DataFrame ──────────────────────────────────────────────────────────
def make_df(draws: list[dict]) -> pd.DataFrame:
    """建立測試用 DataFrame"""
    return pd.DataFrame(draws)


SINGLE_ROW_CASES = [
    # (n1, n2, n3, n4, n5, mega, 預期 odd_count, even_count, low_count, high_count, white_sum)
    # [3, 7, 12, 25, 36], Mega=5
    # 奇數：3,7,25 → 3；偶數：12,36 → 2
    # 低區(≤23)：3,7,12 → 3；高區：25,36 → 2
    # 總和：3+7+12+25+36=83
    dict(n1=3, n2=7, n3=12, n4=25, n5=36, mega_number=5,
         expected_odd=3, expected_even=2, expected_low=3, expected_high=2, expected_sum=83),

    # [2, 4, 8, 16, 32], Mega=6
    # 全偶數 → odd=0, even=5
    # 低區：2,4,8,16 → 4；高區：32 → 1
    # 總和：62
    dict(n1=2, n2=4, n3=8, n4=16, n5=32, mega_number=6,
         expected_odd=0, expected_even=5, expected_low=4, expected_high=1, expected_sum=62),

    # [1, 3, 5, 7, 9], Mega=1
    # 全奇數 → odd=5, even=0
    # 全低區(≤23) → low=5, high=0
    # 總和：25
    dict(n1=1, n2=3, n3=5, n4=7, n5=9, mega_number=1,
         expected_odd=5, expected_even=0, expected_low=5, expected_high=0, expected_sum=25),
]


class TestPatternStatsOddEven:
    """奇偶比計算測試"""

    @pytest.mark.parametrize("case", SINGLE_ROW_CASES)
    def test_odd_count(self, case):
        df = make_df([{k: v for k, v in case.items() if not k.startswith("expected")}])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["odd_count"] == case["expected_odd"], (
            f"odd_count 預期 {case['expected_odd']}，得到 {result['odd_count']}"
        )

    @pytest.mark.parametrize("case", SINGLE_ROW_CASES)
    def test_even_count(self, case):
        df = make_df([{k: v for k, v in case.items() if not k.startswith("expected")}])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["even_count"] == case["expected_even"]

    @pytest.mark.parametrize("case", SINGLE_ROW_CASES)
    def test_odd_plus_even_equals_5(self, case):
        """奇數 + 偶數個數必須等於 5"""
        df = make_df([{k: v for k, v in case.items() if not k.startswith("expected")}])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["odd_count"] + result["even_count"] == 5


class TestPatternStatsHighLow:
    """高低區計算測試"""

    @pytest.mark.parametrize("case", SINGLE_ROW_CASES)
    def test_low_count(self, case):
        df = make_df([{k: v for k, v in case.items() if not k.startswith("expected")}])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["low_count"] == case["expected_low"]

    @pytest.mark.parametrize("case", SINGLE_ROW_CASES)
    def test_high_count(self, case):
        df = make_df([{k: v for k, v in case.items() if not k.startswith("expected")}])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["high_count"] == case["expected_high"]


class TestPatternStatsSum:
    """總和計算測試"""

    @pytest.mark.parametrize("case", SINGLE_ROW_CASES)
    def test_white_sum(self, case):
        df = make_df([{k: v for k, v in case.items() if not k.startswith("expected")}])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["white_sum"] == case["expected_sum"]


class TestPatternStatsConsecutive:
    """連號計算測試"""

    def test_no_consecutive(self):
        """[1, 3, 5, 7, 9] → 無連號"""
        df = make_df([dict(n1=1, n2=3, n3=5, n4=7, n5=9, mega_number=1)])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["consecutive_pairs"] == 0

    def test_one_consecutive(self):
        """[1, 2, 5, 7, 9] → 1 對連號 (1,2)"""
        df = make_df([dict(n1=1, n2=2, n3=5, n4=7, n5=9, mega_number=1)])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["consecutive_pairs"] == 1

    def test_two_consecutive(self):
        """[7, 8, 12, 13, 20] → 2 對連號"""
        df = make_df([dict(n1=7, n2=8, n3=12, n4=13, n5=20, mega_number=1)])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["consecutive_pairs"] == 2

    def test_all_consecutive(self):
        """[5, 6, 7, 8, 9] → 4 對連號"""
        df = make_df([dict(n1=5, n2=6, n3=7, n4=8, n5=9, mega_number=1)])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["consecutive_pairs"] == 4


class TestPatternStatsNumberRange:
    """號碼跨度計算測試"""

    def test_range_calculation(self):
        """[1, 10, 20, 30, 47] → range = 47 - 1 = 46"""
        df = make_df([dict(n1=1, n2=10, n3=20, n4=30, n5=47, mega_number=1)])
        df["draw_number"] = 1
        result = add_pattern_features(df).iloc[0]
        assert result["number_range"] == 46


class TestFrequencyAnalysis:
    """號碼頻率統計測試"""

    def test_basic_frequency(self):
        """號碼 1 出現 3 次，號碼 2 出現 1 次"""
        draws = [
            dict(draw_number=1, n1=1, n2=2, n3=10, n4=20, n5=30, mega_number=5),
            dict(draw_number=2, n1=1, n2=5,  n3=15, n4=25, n5=35, mega_number=3),
            dict(draw_number=3, n1=1, n2=10, n3=20, n4=30, n5=40, mega_number=7),
        ]
        df = pd.DataFrame(draws)
        freq = compute_white_ball_frequency(df)
        assert freq[1] == 3
        assert freq[2] == 1
        assert freq[47] == 0  # 未出現的號碼應為 0

    def test_total_frequency(self):
        """3 期 × 每期 5 顆 = 15 顆，頻率總和應為 15"""
        draws = [
            dict(draw_number=i, n1=i, n2=i+1, n3=i+2, n4=i+3, n5=i+4, mega_number=1)
            for i in range(1, 4)
        ]
        df = pd.DataFrame(draws)
        freq = compute_white_ball_frequency(df)
        assert freq.sum() == 15

    def test_windowed_frequency(self):
        """window=1 時只計算最後 1 期的頻率"""
        draws = [
            dict(draw_number=1, n1=1, n2=2, n3=3, n4=4, n5=5, mega_number=1),
            dict(draw_number=2, n1=10, n2=20, n3=30, n4=40, n5=47, mega_number=2),
        ]
        df = pd.DataFrame(draws)
        freq = compute_white_ball_frequency(df, window=1)
        # 只看最後 1 期：號碼 10,20,30,40,47 各出現 1 次
        assert freq[1]  == 0   # 第 1 期的號碼不應計入
        assert freq[10] == 1
        assert freq[47] == 1
