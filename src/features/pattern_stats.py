"""
pattern_stats.py — 模式特徵計算
=================================
計算每一期開獎結果的「模式特徵」，把原始號碼轉換成
數學統計量，讓機器學習模型能夠「理解」號碼組合的結構。

實作的特徵：
  1. 總和（white_sum）：5顆白球加總
  2. 平均值（white_mean）
  3. 奇偶比（odd_count / even_count）
  4. 高低比（low_count / high_count）：低區 1–23，高區 24–47
  5. 連號對數（consecutive_pairs）：相鄰差值為1的對數
  6. 號碼跨度（number_range）：最大值 − 最小值
  7. 個位數多樣性（unique_last_digits）：使用了幾種個位數

設計原則：
  全部使用 pandas 向量化操作，而非 .apply(lambda)，
  在大型 DataFrame 上速度快很多（10x–100x）。
"""

import logging
import numpy as np
import pandas as pd

from config.settings import LOW_HIGH_THRESHOLD, LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_COLS = ["n1", "n2", "n3", "n4", "n5"]


def add_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    對整個 DataFrame 批量計算模式特徵，新增欄位後回傳。
    這是本模組的主要入口函式。

    Args:
        df: 開獎資料 DataFrame，必須包含 n1-n5 和 mega_number 欄位

    Returns:
        新增特徵欄位後的 DataFrame（不修改原始 df，回傳副本）
    """
    df = df.copy()  # 避免修改原始 DataFrame（防禦性複製）

    white = df[WHITE_COLS]

    # ── 1. 總和與平均值 ──────────────────────────────────────────────────────
    # 觀察：歷史資料中，5個白球總和的分佈接近常態分佈
    # 極端高或低的總和組合在歷史上較少見
    df["white_sum"]  = white.sum(axis=1)
    df["white_mean"] = white.mean(axis=1).round(2)

    # ── 2. 奇偶比（各球是否為奇數：% 2 == 1）────────────────────────────────
    # 理論期望：5 顆球中約 2–3 顆為奇數（接近 50/50）
    # 向量化：對每個欄位分別做 % 2，然後 sum(axis=1) 得到奇數球個數
    odd_mask     = white.apply(lambda col: col % 2 == 1)
    df["odd_count"]  = odd_mask.sum(axis=1)
    df["even_count"] = 5 - df["odd_count"]

    # ── 3. 高低區比（1–LOW_HIGH_THRESHOLD 為低區，其餘為高區）────────────────
    low_mask      = white.apply(lambda col: col <= LOW_HIGH_THRESHOLD)
    df["low_count"]  = low_mask.sum(axis=1)
    df["high_count"] = 5 - df["low_count"]

    # ── 4. 連號對數 ──────────────────────────────────────────────────────────
    # 例如：[3, 7, 8, 12, 13] → 有 (7,8) 和 (12,13) 兩對連號 → consecutive_pairs = 2
    # 做法：對排序後相鄰欄位做差值，差值=1 代表連號
    # n1 ≤ n2 ≤ n3 ≤ n4 ≤ n5（parser.py 已排序），因此可以直接用相鄰欄位差
    df["consecutive_pairs"] = (
        ((df["n2"] - df["n1"]) == 1).astype(int)
        + ((df["n3"] - df["n2"]) == 1).astype(int)
        + ((df["n4"] - df["n3"]) == 1).astype(int)
        + ((df["n5"] - df["n4"]) == 1).astype(int)
    )

    # ── 5. 號碼跨度（最大 − 最小）────────────────────────────────────────────
    # n1 是最小，n5 是最大（因為已排序）
    df["number_range"] = df["n5"] - df["n1"]

    # ── 6. 個位數多樣性 ──────────────────────────────────────────────────────
    # 計算 5 顆球各自的個位數（% 10），看有幾種不同的個位數
    # 個位數全部一樣（例如 3,13,23,33,43）是極端情況
    # 使用 apply 是因為需要 set() 操作，這部分難以完全向量化
    last_digits = white.apply(lambda col: col % 10)
    df["unique_last_digits"] = last_digits.apply(
        lambda row: len(set(row)), axis=1
    )

    # ── 7. Mega 球奇偶 ───────────────────────────────────────────────────────
    df["mega_is_odd"] = (df["mega_number"] % 2 == 1).astype(int)

    logger.debug(f"add_pattern_features: 處理 {len(df)} 筆資料，新增 7 類特徵")
    return df


def compute_rolling_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    計算滾動視窗統計特徵（Rolling Window Features）。

    用途：在建構 ML 訓練資料時，以「前 window 期的統計量」作為特徵，
    預測下一期。這是時間序列預測的標準做法。

    例如：若 window=10，第 i 期的訓練特徵是第 i-10 到 i-1 期的統計摘要。

    Args:
        df:     已包含 pattern_features 的 DataFrame
        window: 滾動視窗大小（期數）

    Returns:
        新增滾動特徵欄位的 DataFrame
    """
    df = df.copy()

    roll_cols = ["white_sum", "odd_count", "low_count", "consecutive_pairs"]

    for col in roll_cols:
        if col not in df.columns:
            logger.warning(f"欄位 {col} 不存在，跳過滾動計算")
            continue
        # 滾動平均（用過去 window 期的平均值作為當期特徵）
        df[f"{col}_roll{window}_mean"] = (
            df[col].rolling(window=window, min_periods=window).mean()
        )
        # 滾動標準差（衡量近期的「變動性」）
        df[f"{col}_roll{window}_std"] = (
            df[col].rolling(window=window, min_periods=window).std()
        )

    logger.debug(f"compute_rolling_features: window={window}，新增 {len(roll_cols)*2} 個滾動特徵")
    return df
