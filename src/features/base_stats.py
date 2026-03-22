"""
base_stats.py — 基礎統計特徵
==============================
計算兩類基礎統計特徵：
  1. 號碼頻率（熱/冷號分析）
  2. Gap 間隔分析（每個號碼距上次出現的期數）

設計說明：
  所有函式接受 DataFrame 輸入，回傳 DataFrame 或 Series，
  方便鏈式呼叫與管線整合。
  盡量使用 pandas 向量化操作（.apply 在大資料時很慢，能避則避）。
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    MEGA_BALL_MIN, MEGA_BALL_MAX,
    SHORT_WINDOW, MID_WINDOW, LONG_WINDOW,
    LOG_LEVEL, LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# 白球與 Mega 球的完整號碼範圍
WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))  # 1–47
MEGA_NUMBERS  = list(range(MEGA_BALL_MIN,  MEGA_BALL_MAX  + 1))  # 1–27

WHITE_COLS = ["n1", "n2", "n3", "n4", "n5"]


def compute_white_ball_frequency(
    df: pd.DataFrame,
    window: Optional[int] = None,
) -> pd.Series:
    """
    計算每個白球號碼（1–47）的出現頻率。

    Args:
        df:     開獎資料 DataFrame（必須包含 n1-n5 欄位）
        window: 若指定，只計算最近 window 期的頻率（熱號分析）
                None 表示計算全部歷史

    Returns:
        pd.Series，index=號碼(1-47)，values=出現次數
        未出現的號碼填 0（reindex 確保所有號碼都有值）
    """
    data = df.tail(window) if window else df

    # 把5個白球欄位「堆疊」成一個扁平 Series，然後計算頻率
    # 這是 pandas 向量化操作，比 for 迴圈快很多
    all_balls = pd.concat([data[col] for col in WHITE_COLS], ignore_index=True)
    freq = all_balls.value_counts().reindex(WHITE_NUMBERS, fill_value=0)

    return freq.sort_index()


def compute_mega_ball_frequency(
    df: pd.DataFrame,
    window: Optional[int] = None,
) -> pd.Series:
    """
    計算每個 Mega 球號碼（1–27）的出現頻率。

    Args:
        df:     開獎資料 DataFrame（必須包含 mega_number 欄位）
        window: 最近幾期（None=全部歷史）

    Returns:
        pd.Series，index=號碼(1-27)，values=出現次數
    """
    data = df.tail(window) if window else df
    freq = data["mega_number"].value_counts().reindex(MEGA_NUMBERS, fill_value=0)
    return freq.sort_index()


def compute_gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算每個白球號碼的 Gap 間隔統計。

    Gap 的定義：
      某號碼在第 i 期出現，距上次（第 j 期）出現的間隔 = i - j 期

    演算法：
      遍歷所有期次（按時間順序），追蹤每個號碼最後出現的「列索引」。
      遇到號碼再次出現時，計算當前列索引 - 上次出現索引。

    Args:
        df: 開獎資料 DataFrame，必須已按 draw_number 升序排列

    Returns:
        pd.DataFrame，index=號碼(1-47)，欄位：
          avg_gap     — 平均間隔期數
          max_gap     — 最長連續缺席期數
          min_gap     — 最短出現間隔
          current_gap — 距最新一期已有幾期未出現
          appearances — 歷史出現總次數
    """
    df = df.sort_values("draw_number").reset_index(drop=True)

    # 使用 dict 追蹤：{號碼: 上次出現的列索引}
    last_seen: dict[int, int] = {}
    # 使用 dict 累積：{號碼: [gap1, gap2, ...]}
    gaps: dict[int, list[int]] = {n: [] for n in WHITE_NUMBERS}

    for idx, row in df.iterrows():
        for col in WHITE_COLS:
            n = row[col]
            if n in last_seen:
                # 計算間隔：當前索引 - 上次索引
                gaps[n].append(idx - last_seen[n])
            last_seen[n] = idx

    # 最新一期的列索引（用於計算 current_gap）
    max_idx = len(df) - 1

    records = []
    for n in WHITE_NUMBERS:
        gap_list = gaps[n]
        records.append({
            "number":      n,
            "avg_gap":     float(np.mean(gap_list)) if gap_list else None,
            "max_gap":     max(gap_list) if gap_list else None,
            "min_gap":     min(gap_list) if gap_list else None,
            "current_gap": max_idx - last_seen.get(n, -1),
            "appearances": len(gap_list) + (1 if n in last_seen else 0),
        })

    result = pd.DataFrame(records).set_index("number")
    return result


def compute_mega_gap_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算每個 Mega 球號碼（1–27）的 Gap 統計。
    邏輯與 compute_gap_stats 相同，但只處理 mega_number 欄位。

    Args:
        df: 開獎資料 DataFrame，已按 draw_number 升序排列

    Returns:
        pd.DataFrame，index=號碼(1-27)，欄位同 compute_gap_stats
    """
    df = df.sort_values("draw_number").reset_index(drop=True)

    last_seen: dict[int, int] = {}
    gaps: dict[int, list[int]] = {n: [] for n in MEGA_NUMBERS}

    for idx, row in df.iterrows():
        n = row["mega_number"]
        if n in last_seen:
            gaps[n].append(idx - last_seen[n])
        last_seen[n] = idx

    max_idx = len(df) - 1

    records = []
    for n in MEGA_NUMBERS:
        gap_list = gaps[n]
        records.append({
            "number":      n,
            "avg_gap":     float(np.mean(gap_list)) if gap_list else None,
            "max_gap":     max(gap_list) if gap_list else None,
            "min_gap":     min(gap_list) if gap_list else None,
            "current_gap": max_idx - last_seen.get(n, -1),
            "appearances": len(gap_list) + (1 if n in last_seen else 0),
        })

    return pd.DataFrame(records).set_index("number")


def build_number_current_gap(df: pd.DataFrame) -> dict[int, int]:
    """
    計算在最新一期之前，每個白球號碼的「當前缺席期數」。
    這個值直接作為 ML 特徵使用（而非只是統計摘要）。

    例如：號碼 7 最後一次出現是 3 期前 → current_gap[7] = 3

    Args:
        df: 開獎資料 DataFrame，已按 draw_number 升序排列

    Returns:
        dict[號碼, 缺席期數]，所有 1–47 號碼都有值（最大為總期數）
    """
    df = df.sort_values("draw_number").reset_index(drop=True)
    max_idx = len(df) - 1
    last_seen: dict[int, int] = {}

    for idx, row in df.iterrows():
        for col in WHITE_COLS:
            last_seen[row[col]] = idx

    return {
        n: max_idx - last_seen.get(n, -1)
        for n in WHITE_NUMBERS
    }


def build_number_gap_stats(df: pd.DataFrame) -> tuple[dict[int, int], dict[int, float]]:
    """
    單次遍歷同時計算每個白球號碼的 current_gap 和 avg_gap。
    比分別呼叫 build_number_current_gap + compute_gap_stats 快一倍。

    Args:
        df: 開獎資料 DataFrame，已按 draw_number 升序排列

    Returns:
        (current_gap_dict, avg_gap_dict)
        current_gap_dict: dict[號碼, 當前缺席期數]
        avg_gap_dict:     dict[號碼, 歷史平均間距]（無記錄的號碼用總期數代替）
    """
    df = df.sort_values("draw_number").reset_index(drop=True)
    n_total = len(df)
    last_seen: dict[int, int] = {}
    gap_sums:  dict[int, float] = {n: 0.0 for n in WHITE_NUMBERS}
    gap_counts: dict[int, int]  = {n: 0   for n in WHITE_NUMBERS}

    for idx in range(n_total):
        row = df.iloc[idx]
        for col in WHITE_COLS:
            ball = int(row[col])
            if ball in last_seen:
                gap = idx - last_seen[ball]
                gap_sums[ball]   += gap
                gap_counts[ball] += 1
            last_seen[ball] = idx

    max_idx = n_total - 1
    current_gap = {n: max_idx - last_seen.get(n, -1) for n in WHITE_NUMBERS}
    avg_gap = {
        n: gap_sums[n] / gap_counts[n] if gap_counts[n] > 0 else float(n_total)
        for n in WHITE_NUMBERS
    }
    return current_gap, avg_gap
