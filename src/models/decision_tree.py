"""
decision_tree.py — 決策樹多標籤分類預測
==========================================
概念：
  對每個號碼（1–47）分別訓練一棵決策樹分類器，
  預測「下一期此號碼是否出現」（二元分類）。

特徵：
  - 每個號碼的 current_gap（距上次出現幾期）
  - 每個號碼的最近30期出現頻率
  - 最近一期的奇偶比、高低比、總和、連號數

學習價值：
  理解決策樹的資訊增益/Gini分裂，
  多標籤分類策略（One-vs-Rest），
  以及過擬合問題（剪枝 max_depth）。
"""

import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)
from src.features.base_stats import (
    compute_white_ball_frequency, build_number_current_gap, build_number_gap_stats,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))
WHITE_COLS    = ["n1", "n2", "n3", "n4", "n5"]
N_WHITE       = len(WHITE_NUMBERS)


class DecisionTreePredictor:
    """
    決策樹多標籤彩票預測器。

    為每個號碼獨立訓練決策樹分類器：
      輸入特徵  → 188維（gap 47 + gap_ratio 47 + freq_30 47 + freq_100 47）
      輸出標籤  → 47維二元向量（是否出現）

    訓練策略：時序切分（前 train_ratio 訓練，後面測試），不 shuffle。
    """

    def __init__(
        self,
        max_depth:    int   = 5,
        min_samples:  int   = 20,
        freq_window:  int   = 30,
        lookback:     int   = 100,
        train_ratio:  float = 0.8,
    ):
        """
        Args:
            max_depth:    決策樹最大深度（控制過擬合）
            min_samples:  葉節點最少樣本數
            freq_window:  近期頻率計算視窗
            lookback:     建構樣本時，每個樣本使用前 lookback 期資料
            train_ratio:  時序訓練集比例
        """
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.freq_window = freq_window
        self.lookback    = lookback
        self.train_ratio = train_ratio
        self._model      = None
        self._last_fv    = None   # 最後一期的特徵向量
        self._is_fitted  = False

    def _build_feature_vector(self, df_upto: pd.DataFrame) -> np.ndarray:
        """為截至某期的資料建立特徵向量（188維）。"""
        # gap + avg_gap（單次遍歷同時計算）
        current_gap, avg_gap = build_number_gap_stats(df_upto)
        gap_features       = np.array([current_gap[n] for n in WHITE_NUMBERS], dtype=np.float32)
        avg_gap_features   = np.array([avg_gap[n]     for n in WHITE_NUMBERS], dtype=np.float32)
        gap_ratio_features = gap_features / np.maximum(avg_gap_features, 1.0)

        # 近期頻率（freq_window，預設 30）
        freq30         = compute_white_ball_frequency(df_upto, window=self.freq_window)
        freq30_features = np.array([freq30.get(n, 0) for n in WHITE_NUMBERS], dtype=np.float32)

        # 長期頻率（freq_100）
        freq100         = compute_white_ball_frequency(df_upto, window=min(100, len(df_upto)))
        freq100_features = np.array([freq100.get(n, 0) for n in WHITE_NUMBERS], dtype=np.float32)

        return np.concatenate([gap_features, gap_ratio_features, freq30_features, freq100_features])

    def _build_label_vector(self, row: pd.Series) -> np.ndarray:
        """從一期開獎記錄建立 47 維二元標籤向量。"""
        appeared = set(int(row[c]) for c in WHITE_COLS)
        return np.array([1 if n in appeared else 0 for n in WHITE_NUMBERS], dtype=np.int8)

    def fit(self, df: pd.DataFrame) -> "DecisionTreePredictor":
        """
        從歷史資料建立特徵矩陣與標籤矩陣，訓練決策樹。

        Args:
            df: 開獎資料 DataFrame，已按 draw_date 升序排列
        """
        df = df.sort_values("draw_date").reset_index(drop=True)

        min_periods = max(self.freq_window + 5, self.lookback)
        X_list, y_list = [], []

        logger.info(f"決策樹：建立訓練樣本（共 {len(df) - min_periods} 期）...")

        for i in range(min_periods, len(df)):
            df_upto = df.iloc[:i]
            fv      = self._build_feature_vector(df_upto)
            label   = self._build_label_vector(df.iloc[i])
            X_list.append(fv)
            y_list.append(label)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int8)

        # 儲存最後一期特徵向量（用於 predict）
        self._last_fv = self._build_feature_vector(df)

        # 時序切分（不 shuffle）
        split_idx = int(len(X) * self.train_ratio)
        X_train, y_train = X[:split_idx], y[:split_idx]

        # 訓練多標籤決策樹
        base_clf = DecisionTreeClassifier(
            max_depth        = self.max_depth,
            min_samples_leaf = self.min_samples,
            class_weight     = "balanced",   # 處理不平衡（出現 vs 未出現）
            random_state     = 42,
        )
        self._model = MultiOutputClassifier(base_clf, n_jobs=-1)
        self._model.fit(X_train, y_train)

        self._is_fitted = True
        logger.info(f"決策樹訓練完成（訓練集 {split_idx} 期，特徵 {X.shape[1]} 維）")
        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        以最後一期特徵向量預測下一期，
        選出預測機率最高的 n_white 顆白球。

        Returns:
            {"white_balls": [int,...], "mega_balls": []}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        fv = self._last_fv.reshape(1, -1)
        # predict_proba 回傳 list of arrays，每個 array shape=(1, 2)
        proba_list = self._model.predict_proba(fv)
        # 取每個號碼「出現」的機率（class=1）
        proba_appear = np.array([p[0][1] for p in proba_list])

        top_indices = np.argsort(proba_appear)[::-1][:n_white]
        top_white   = sorted([WHITE_NUMBERS[i] for i in top_indices])
        proba_map   = {WHITE_NUMBERS[i]: float(proba_appear[i]) for i in range(len(WHITE_NUMBERS))}

        logger.info(f"決策樹推薦：白球={top_white}")
        return {"white_balls": top_white, "mega_balls": [], "proba": proba_map}

    def get_feature_importance(self) -> pd.DataFrame:
        """
        回傳各特徵的平均重要性（跨所有 47 棵樹的均值）。

        Returns:
            pd.DataFrame，欄位：feature_name, importance
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        # 每棵子樹的 feature_importances_
        importances = np.array([
            est.feature_importances_
            for est in self._model.estimators_
        ]).mean(axis=0)

        feature_names = (
            [f"gap_{n}"       for n in WHITE_NUMBERS] +
            [f"gap_ratio_{n}" for n in WHITE_NUMBERS] +
            [f"freq30_{n}"    for n in WHITE_NUMBERS] +
            [f"freq100_{n}"   for n in WHITE_NUMBERS]
        )
        return (
            pd.DataFrame({"feature_name": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
