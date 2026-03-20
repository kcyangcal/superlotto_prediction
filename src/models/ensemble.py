"""
ensemble.py — 集成學習（全模型加權投票）
==========================================
概念：
  整合所有子模型的預測結果，透過加權投票選出最終推薦號碼。

  策略：
    1. 每個子模型預測出「建議白球」列表（各 5 顆）
    2. 每顆球依據其所屬模型的權重累加投票分數
    3. 選出累計分數最高的 5 顆球

  支援的子模型：
    - FrequencyBaseline（頻率基準）
    - MonteCarloPredictor（蒙地卡羅）
    - MarkovPredictor（馬可夫鏈）
    - BayesianPredictor（貝氏推論）
    - KNNPredictor（KNN 最近鄰）
    - DecisionTreePredictor（決策樹）
    - GeneticPredictor（遺傳演算法）

學習價值：
  理解集成學習的基本思路——多個弱學習器組合可比單一模型更穩健；
  以及如何設計加權投票機制。
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))


# 預設各模型權重（總和不需為 1，會自動正規化）
DEFAULT_WEIGHTS: Dict[str, float] = {
    "frequency":    1.0,
    "monte_carlo":  1.0,
    "markov":       1.0,
    "bayesian":     1.2,   # 有先驗知識，略微提高權重
    "knn":          1.2,   # 相似期推薦，略微提高權重
    "decision_tree": 0.8,  # 決策樹容易過擬合，權重略低
    "genetic":      0.8,   # GA 依賴適應度設計，較主觀
}


class EnsemblePredictor:
    """
    多模型加權投票集成預測器。

    使用方式：
        ensemble = EnsemblePredictor(weights={...})
        ensemble.fit(df)         # 訓練所有子模型
        result = ensemble.predict()
        report = ensemble.get_model_comparison()
    """

    def __init__(
        self,
        weights:         Optional[Dict[str, float]] = None,
        knn_k:           int   = 10,
        mc_simulations:  int   = 100_000,
        dt_max_depth:    int   = 5,
        ga_generations:  int   = 300,
        enable_knn:      bool  = True,   # KNN 建構慢，可選關閉
        enable_dt:       bool  = True,   # 決策樹建構較慢，可選關閉
    ):
        """
        Args:
            weights:        各模型權重字典（key 對應模型名稱）
            knn_k:          KNN 的 k 值
            mc_simulations: 蒙地卡羅模擬次數
            dt_max_depth:   決策樹最大深度
            ga_generations: 遺傳演算法代數
            enable_knn:     是否啟用 KNN（慢，約 20–60 秒）
            enable_dt:      是否啟用決策樹（中等速度）
        """
        self.weights        = weights or DEFAULT_WEIGHTS
        self.knn_k          = knn_k
        self.mc_simulations = mc_simulations
        self.dt_max_depth   = dt_max_depth
        self.ga_generations = ga_generations
        self.enable_knn     = enable_knn
        self.enable_dt      = enable_dt

        self._models:      Dict[str, object] = {}
        self._predictions: Dict[str, List[int]] = {}
        self._is_fitted    = False

    def fit(self, df: pd.DataFrame) -> "EnsemblePredictor":
        """
        依序訓練所有啟用的子模型。

        Args:
            df: 開獎資料 DataFrame
        """
        # 延遲匯入避免循環依賴
        from src.models.baseline       import FrequencyBaseline
        from src.models.monte_carlo    import MonteCarloPredictor
        from src.models.markov         import MarkovPredictor
        from src.models.bayesian       import BayesianPredictor
        from src.models.knn_model      import KNNPredictor
        from src.models.decision_tree  import DecisionTreePredictor
        from src.models.genetic        import GeneticPredictor

        model_configs = [
            ("frequency",    FrequencyBaseline()),
            ("monte_carlo",  MonteCarloPredictor(n_simulations=self.mc_simulations)),
            ("markov",       MarkovPredictor()),
            ("bayesian",     BayesianPredictor()),
            ("genetic",      GeneticPredictor(n_generations=self.ga_generations)),
        ]

        if self.enable_knn:
            model_configs.append(("knn", KNNPredictor(k=self.knn_k)))
        if self.enable_dt:
            model_configs.append(("decision_tree", DecisionTreePredictor(max_depth=self.dt_max_depth)))

        for name, model in model_configs:
            try:
                logger.info(f"集成：訓練子模型 [{name}]...")
                model.fit(df)
                self._models[name] = model
                logger.info(f"集成：[{name}] 完成")
            except Exception as e:
                logger.warning(f"集成：[{name}] 訓練失敗，跳過。錯誤：{e}")

        self._is_fitted = True
        logger.info(f"集成學習訓練完成，共 {len(self._models)} 個子模型")
        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        執行加權投票，選出得分最高的號碼。

        Returns:
            {"white_balls": [int,...], "mega_balls": [], "model_votes": {...}}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        # 各模型分別預測
        vote_score = np.zeros(len(WHITE_NUMBERS), dtype=np.float64)

        for name, model in self._models.items():
            try:
                result   = model.predict(n_white=n_white)
                pred     = result.get("white_balls", [])
                weight   = self.weights.get(name, 1.0)

                self._predictions[name] = pred
                for ball in pred:
                    idx = ball - WHITE_BALL_MIN
                    if 0 <= idx < len(WHITE_NUMBERS):
                        vote_score[idx] += weight
            except Exception as e:
                logger.warning(f"集成：[{name}] 預測失敗，跳過。錯誤：{e}")

        top_indices = np.argsort(vote_score)[::-1][:n_white]
        top_white   = sorted([WHITE_NUMBERS[i] for i in top_indices])

        logger.info(f"集成學習推薦：白球={top_white}")
        return {
            "white_balls": top_white,
            "mega_balls":  [],
            "model_votes": dict(self._predictions),
        }

    def get_model_comparison(self) -> pd.DataFrame:
        """
        回傳所有子模型的預測結果比較表。

        Returns:
            pd.DataFrame，欄位：model, balls, weight
        """
        if not self._predictions:
            raise RuntimeError("請先呼叫 predict()")

        rows = []
        for name, balls in self._predictions.items():
            rows.append({
                "model":  name,
                "balls":  str(sorted(balls)),
                "weight": self.weights.get(name, 1.0),
            })
        return pd.DataFrame(rows).sort_values("weight", ascending=False).reset_index(drop=True)

    def get_vote_scores(self) -> pd.DataFrame:
        """
        回傳所有號碼的累計投票分數。

        Returns:
            pd.DataFrame，欄位：number, vote_score，降序排列
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit() 和 predict()")

        vote_score = np.zeros(len(WHITE_NUMBERS), dtype=np.float64)
        for name, balls in self._predictions.items():
            weight = self.weights.get(name, 1.0)
            for ball in balls:
                idx = ball - WHITE_BALL_MIN
                if 0 <= idx < len(WHITE_NUMBERS):
                    vote_score[idx] += weight

        df = pd.DataFrame({
            "number":     WHITE_NUMBERS,
            "vote_score": vote_score,
        })
        return df.sort_values("vote_score", ascending=False).reset_index(drop=True)
