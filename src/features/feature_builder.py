"""
feature_builder.py — 特徵工程整合入口
========================================
整合 base_stats.py 和 pattern_stats.py 的所有特徵計算，
提供兩個主要功能：

  1. build_draw_features(df)：
     計算每期的衍生特徵，用於存回 draw_features 資料表

  2. build_ml_feature_matrix(df)：
     建構 ML 模型的訓練矩陣 (X, y)
     X：每一期的「前 lookback 期」特徵向量
     y：每一期每個號碼是否出現的 0/1 矩陣（47維）

這個模組是特徵工程的「最終出口」，確保 ML 模型只需面對
整理好的特徵矩陣，不需要知道底層的計算細節。
"""

import logging
import numpy as np
import pandas as pd

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    LOOKBACK_PERIODS, LOG_LEVEL, LOG_FORMAT,
    SHORT_WINDOW, MID_WINDOW,
)
from src.features.base_stats import (
    compute_white_ball_frequency,
    compute_gap_stats,
    build_number_current_gap,
)
from src.features.pattern_stats import add_pattern_features, compute_rolling_features

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))
WHITE_COLS    = ["n1", "n2", "n3", "n4", "n5"]


def build_draw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算每期的衍生特徵，準備寫入 draw_features 資料表。

    Args:
        df: 開獎資料 DataFrame（from DrawRepository.get_all_draws()）

    Returns:
        pd.DataFrame，欄位對應 draw_features 資料表，每期一筆
    """
    logger.info(f"開始計算 {len(df)} 期的衍生特徵...")

    # 新增模式特徵
    features_df = add_pattern_features(df)

    # 只保留 draw_features 資料表需要的欄位
    output_cols = [
        "draw_number", "white_sum", "white_mean",
        "odd_count", "even_count", "low_count", "high_count",
        "consecutive_pairs", "number_range", "unique_last_digits", "mega_is_odd",
    ]
    result = features_df[output_cols].copy()

    logger.info(f"特徵計算完成：{len(result)} 筆，{len(output_cols)} 個欄位")
    return result


def build_ml_feature_matrix(
    df: pd.DataFrame,
    lookback: int = LOOKBACK_PERIODS,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    建構 ML 模型的訓練矩陣。

    問題定義：
      給定「前 lookback 期」的各種特徵，預測「下一期」哪些號碼會出現。
      這是一個多標籤二元分類問題：47個號碼各自獨立預測（0或1）。

    特徵向量 X[i] 包含：
      ① 前 lookback 期的滾動統計（均值/標準差）
      ② 每個號碼目前已缺席的期數（47維）
      ③ 每個號碼在最近 SHORT_WINDOW 期的出現次數（47維）

    標籤向量 y[i] 包含：
      第 i 期每個號碼是否出現（0=未出現，1=出現），47維

    警告：
      ⚠️ 時間序列資料絕對不能隨機 shuffle！
         訓練/測試集必須按時間順序切分（前80%訓練，後20%測試）。
         若 shuffle，未來資訊會洩漏到訓練集（Look-ahead bias）。

    Args:
        df:       開獎資料 DataFrame，已按 draw_number 升序排列
        lookback: 往前看幾期（預設 10）

    Returns:
        X:       特徵矩陣，shape = (n_samples, n_features)
        y:       標籤矩陣，shape = (n_samples, 47)
        indices: 對應的 draw_number 索引，方便回溯追蹤
    """
    logger.info(f"建構 ML 特徵矩陣（lookback={lookback}，共 {len(df)} 期）...")

    # 先計算所有期的模式特徵和滾動特徵
    df_feat = add_pattern_features(df)
    df_feat = compute_rolling_features(df_feat, window=lookback)
    df_feat = df_feat.sort_values("draw_number").reset_index(drop=True)

    # 滾動特徵欄位（排除前 lookback 期的 NaN）
    roll_feature_cols = [
        c for c in df_feat.columns
        if "_roll" in c and ("_mean" in c or "_std" in c)
    ]

    X_list      = []
    y_list      = []
    index_list  = []

    for i in range(lookback, len(df_feat)):
        # ── 計算當前這一期（第 i 期）作為預測目標時的特徵 ────────────────────
        current_row = df_feat.iloc[i]

        # 特徵 ①：滾動統計（已在 df_feat 中計算好）
        roll_features = current_row[roll_feature_cols].values.tolist()

        # 特徵 ②：計算截至第 i-1 期，每個號碼的「當前缺席期數」
        # 注意：不能用第 i 期的資料，否則會有 look-ahead bias
        past_df = df_feat.iloc[:i]
        current_gap_dict = build_number_current_gap(past_df)
        gap_features = [current_gap_dict[n] for n in WHITE_NUMBERS]

        # 特徵 ③：最近 SHORT_WINDOW 期每個號碼的出現次數
        recent_df = past_df.tail(SHORT_WINDOW)
        freq = compute_white_ball_frequency(recent_df)
        freq_features = [int(freq.get(n, 0)) for n in WHITE_NUMBERS]

        # 合併所有特徵
        feature_vec = roll_features + gap_features + freq_features

        # ── 計算標籤（第 i 期哪些號碼出現）────────────────────────────────────
        appeared = set(current_row[WHITE_COLS].values.tolist())
        label_vec = [1 if n in appeared else 0 for n in WHITE_NUMBERS]

        X_list.append(feature_vec)
        y_list.append(label_vec)
        index_list.append(int(current_row["draw_number"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    indices = pd.Index(index_list, name="draw_number")

    logger.info(f"特徵矩陣建構完成：X shape={X.shape}，y shape={y.shape}")
    logger.info(
        f"特徵維度說明：{len(roll_feature_cols)} 滾動特徵 + "
        f"47 gap特徵 + 47 頻率特徵 = {X.shape[1]} 維"
    )

    return X, y, indices
