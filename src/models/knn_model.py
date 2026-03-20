"""
knn_model.py — KNN 最近鄰預測
================================
概念：
  在歷史所有開獎中，找到「最近似當前狀態」的 K 期，
  然後看這 K 期的「下一期」出現了哪些號碼，
  統計出現頻率來預測下一期。

當前狀態特徵向量：
  - 每個號碼的當前 gap（距上次出現幾期）
  - 最近 N 期每個號碼的出現頻率

KNN 距離度量：歐氏距離（Euclidean distance）

學習價值：
  理解 KNN 在時序資料中的應用，特徵向量設計，相似度度量
"""

import logging
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)
from src.features.base_stats import compute_white_ball_frequency, build_number_current_gap

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))
WHITE_COLS    = ["n1", "n2", "n3", "n4", "n5"]


class KNNPredictor:
    """
    K 最近鄰歷史相似期預測器。

    特徵向量 = [每個號碼的 current_gap (47維) + 最近20期頻率 (47維)] = 94維
    """

    def __init__(self, k: int = 10, freq_window: int = 20):
        """
        Args:
            k:           找幾個最近鄰（歷史相似期）
            freq_window: 計算近期頻率的視窗期數
        """
        self.k           = k
        self.freq_window = freq_window
        self._knn        = None
        self._X          = None   # 所有期的特徵向量
        self._df         = None   # 原始 DataFrame（用於查下一期號碼）
        self._is_fitted  = False

    def _build_feature_vector(self, df_upto: pd.DataFrame) -> np.ndarray:
        """
        為截至某期的歷史資料建立特徵向量。

        Args:
            df_upto: 截至某期（含）的歷史資料

        Returns:
            np.ndarray, shape=(94,)
        """
        # Gap 特徵（47維）
        gap_dict     = build_number_current_gap(df_upto)
        gap_features = np.array([gap_dict[n] for n in WHITE_NUMBERS], dtype=np.float32)

        # 近期頻率特徵（47維）
        freq         = compute_white_ball_frequency(df_upto, window=self.freq_window)
        freq_features = np.array([freq.get(n, 0) for n in WHITE_NUMBERS], dtype=np.float32)

        return np.concatenate([gap_features, freq_features])

    def fit(self, df: pd.DataFrame) -> "KNNPredictor":
        """
        建立所有歷史期次的特徵向量，準備 KNN 查詢。

        注意：特徵向量建構很耗時（每期都要計算一次 gap），
              資料量 2691 期約需 10–30 秒。

        Args:
            df: 開獎資料 DataFrame，已按 draw_date 升序排列
        """
        df = df.sort_values("draw_date").reset_index(drop=True)
        self._df = df

        min_periods = self.freq_window + 5  # 至少要有足夠的歷史期數
        feature_list = []

        logger.info(f"KNN：建立 {len(df) - min_periods} 期特徵向量（可能需要一些時間）...")

        for i in range(min_periods, len(df)):
            df_upto = df.iloc[:i]
            fv = self._build_feature_vector(df_upto)
            feature_list.append(fv)

        self._X          = np.array(feature_list, dtype=np.float32)
        self._start_idx  = min_periods  # X[0] 對應 df.iloc[min_periods]

        # 標準化特徵（使不同量級的特徵具有可比性）
        self._mean = self._X.mean(axis=0)
        self._std  = self._X.std(axis=0) + 1e-8
        X_scaled   = (self._X - self._mean) / self._std

        # 建立 KNN 模型（只用於查詢，不訓練標籤）
        self._knn = NearestNeighbors(n_neighbors=self.k + 1, metric="euclidean", n_jobs=-1)
        self._knn.fit(X_scaled)

        self._X_scaled   = X_scaled
        self._is_fitted  = True
        logger.info(f"KNN 建立完成，特徵矩陣 shape={self._X.shape}")
        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        以最後一期的狀態為查詢，找 K 個最相似的歷史期，
        統計其「下一期」的號碼頻率，選出 top-n_white。

        Returns:
            {"white_balls": [int,...]}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        # 建立最新一期的查詢向量
        query = self._build_feature_vector(self._df)
        query_scaled = (query - self._mean) / self._std

        # 找最相似的 K 期
        distances, indices = self._knn.kneighbors([query_scaled])
        neighbor_indices = indices[0]

        # 統計 K 個最近鄰的「下一期」出現哪些號碼
        vote_counts = np.zeros(len(WHITE_NUMBERS), dtype=int)

        for idx in neighbor_indices:
            actual_df_idx = idx + self._start_idx + 1  # 下一期的索引
            if actual_df_idx >= len(self._df):
                continue
            next_row = self._df.iloc[actual_df_idx]
            appeared = [int(next_row[c]) - 1 for c in WHITE_COLS]
            for a in appeared:
                vote_counts[a] += 1

        top_indices = np.argsort(vote_counts)[::-1][:n_white]
        top_white   = sorted([WHITE_NUMBERS[i] for i in top_indices])

        logger.info(f"KNN（k={self.k}）推薦：白球={top_white}")
        return {"white_balls": top_white, "mega_balls": []}
