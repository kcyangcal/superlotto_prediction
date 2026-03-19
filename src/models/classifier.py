"""
classifier.py — 多標籤分類器訓練流程（整合版）
=================================================
提供一個 LotteryClassifier 類別，把所有訓練步驟封裝起來：
  1. 載入資料
  2. 建構特徵矩陣
  3. 時序切分
  4. 訓練 Random Forest + XGBoost
  5. 評估並輸出結果

這個類別是 scripts/run_prediction.py 的主要呼叫對象。
"""

import logging
import numpy as np
import pandas as pd

from config.settings import LOOKBACK_PERIODS, LOG_LEVEL, LOG_FORMAT
from src.features.feature_builder import build_ml_feature_matrix
from src.models.trainer import (
    time_series_split,
    train_random_forest,
    train_xgboost,
    evaluate_model,
    save_model,
    load_model,
    precision_at_k,
    WHITE_NUMBERS,
)
from src.models.baseline import FrequencyBaseline

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class LotteryClassifier:
    """
    SuperLotto Plus 多標籤分類器。

    使用範例：
        clf = LotteryClassifier()
        clf.fit(df)                        # 訓練所有模型
        clf.evaluate()                     # 印出評估指標
        result = clf.predict_next()        # 預測下一期
    """

    def __init__(self, lookback: int = LOOKBACK_PERIODS):
        self.lookback       = lookback
        self.rf_model       = None
        self.xgb_model      = None
        self.baseline       = None
        self._X_train       = None
        self._X_test        = None
        self._y_train       = None
        self._y_test        = None
        self._X_all         = None  # 全部特徵（用於最終預測）
        self._y_all         = None
        self._is_fitted     = False

    def fit(self, df: pd.DataFrame, train_xgb: bool = True) -> "LotteryClassifier":
        """
        完整訓練流程。

        Args:
            df:        開獎資料 DataFrame（from repository.get_all_draws()）
            train_xgb: 是否同時訓練 XGBoost（較慢，預設 True）

        Returns:
            self
        """
        logger.info("=" * 60)
        logger.info("開始訓練 LotteryClassifier")
        logger.info(f"  資料筆數：{len(df)}")
        logger.info(f"  Lookback：{self.lookback} 期")

        # ── Step 1：建構特徵矩陣 ──────────────────────────────────────────────
        X, y, indices = build_ml_feature_matrix(df, lookback=self.lookback)
        self._X_all = X
        self._y_all = y

        # ── Step 2：時序切分 ──────────────────────────────────────────────────
        self._X_train, self._X_test, self._y_train, self._y_test = (
            time_series_split(X, y)
        )

        # ── Step 3：訓練基準模型 ──────────────────────────────────────────────
        self.baseline = FrequencyBaseline().fit(df)

        # ── Step 4：訓練 Random Forest ────────────────────────────────────────
        self.rf_model = train_random_forest(self._X_train, self._y_train)

        # ── Step 5：訓練 XGBoost（可選）──────────────────────────────────────
        if train_xgb:
            try:
                self.xgb_model = train_xgboost(self._X_train, self._y_train)
            except Exception as e:
                logger.warning(f"XGBoost 訓練失敗：{e}（略過）")

        self._is_fitted = True
        logger.info("LotteryClassifier 訓練完成")
        logger.info("=" * 60)
        return self

    def evaluate(self) -> dict:
        """
        評估所有模型，印出對比結果。

        Returns:
            dict: 各模型的評估指標
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        logger.info("\n📊 模型評估結果")
        logger.info("-" * 50)

        results = {}

        # 基準模型（頻率選號）評估
        baseline_pred = self.baseline.predict(n_white=5)
        baseline_p5 = np.mean([
            sum(1 for b in baseline_pred["white_balls"] if b in set(row))
            for row in (
                [{WHITE_NUMBERS[j]: bool(self._y_test[i, j]) for j in range(47)}
                 for i in range(len(self._y_test))]
            )
        ]) / 5
        logger.info(f"L0 基準模型  Precision@5 = {baseline_p5*100:.2f}%")
        results["baseline"] = {"precision_at_5": baseline_p5}

        # Random Forest 評估
        if self.rf_model:
            rf_results = evaluate_model(self.rf_model, self._X_test, self._y_test)
            results["random_forest"] = rf_results
            logger.info(f"L1 Random Forest Precision@5 = {rf_results['precision_at_5_pct']}")

        # XGBoost 評估
        if self.xgb_model:
            xgb_results = evaluate_model(self.xgb_model, self._X_test, self._y_test)
            results["xgboost"] = xgb_results
            logger.info(f"L2 XGBoost       Precision@5 = {xgb_results['precision_at_5_pct']}")

        logger.info(f"隨機基準         Precision@5 = {5/47*100:.2f}%（理論值）")
        logger.info("-" * 50)
        return results

    def predict_next(self, top_k: int = 5) -> dict:
        """
        使用最後一期的特徵預測「下一期」可能出現的號碼。

        注意：這只是基於歷史統計模式的機率估計，不代表任何實際預測能力。

        Args:
            top_k: 推薦幾顆白球（通常 = 5）

        Returns:
            dict: {
                "rf_prediction":   {"white_balls": [...], "probabilities": {...}},
                "xgb_prediction":  {...},
                "baseline_prediction": {...},
                "ensemble_prediction": {...},  # 三模型機率平均
            }
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        # 取最後一筆特徵（最新一期的資料之後的預測）
        last_features = self._X_all[[-1]]
        result = {}

        def _top_k_from_proba(proba_list, k):
            """從 MultiOutputClassifier.predict_proba 輸出中取 top-k 號碼"""
            probs = np.array([p[0, 1] for p in proba_list])
            top_indices = np.argsort(probs)[::-1][:k]
            top_numbers = sorted([WHITE_NUMBERS[i] for i in top_indices])
            prob_dict   = {WHITE_NUMBERS[i]: float(probs[i]) for i in top_indices}
            return top_numbers, prob_dict

        # Random Forest 預測
        if self.rf_model:
            rf_proba = self.rf_model.predict_proba(last_features)
            rf_nums, rf_probs = _top_k_from_proba(rf_proba, top_k)
            result["rf_prediction"] = {
                "white_balls": rf_nums,
                "probabilities": rf_probs,
            }

        # XGBoost 預測
        if self.xgb_model:
            xgb_proba = self.xgb_model.predict_proba(last_features)
            xgb_nums, xgb_probs = _top_k_from_proba(xgb_proba, top_k)
            result["xgb_prediction"] = {
                "white_balls": xgb_nums,
                "probabilities": xgb_probs,
            }

        # 基準模型預測
        result["baseline_prediction"] = self.baseline.predict(n_white=top_k)

        # 整合預測（平均機率）
        if self.rf_model and self.xgb_model:
            rf_p   = np.array([p[0, 1] for p in self.rf_model.predict_proba(last_features)])
            xgb_p  = np.array([p[0, 1] for p in self.xgb_model.predict_proba(last_features)])
            avg_p  = (rf_p + xgb_p) / 2
            top_idx = np.argsort(avg_p)[::-1][:top_k]
            ensemble_nums = sorted([WHITE_NUMBERS[i] for i in top_idx])
            result["ensemble_prediction"] = {"white_balls": ensemble_nums}

        return result

    def save_models(self):
        """存檔所有訓練好的模型"""
        if self.rf_model:
            save_model(self.rf_model, "rf_model.pkl")
        if self.xgb_model:
            save_model(self.xgb_model, "xgb_model.pkl")
        logger.info("所有模型已存檔")
