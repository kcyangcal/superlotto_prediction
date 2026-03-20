"""
bayesian.py — 貝氏推論預測
============================
概念：
  使用貝氏更新（Bayesian Updating）來估計每個號碼「下一期出現」的後驗機率。

  P(號碼出現 | 最近觀測) ∝ P(最近觀測 | 號碼出現) × P(號碼出現)

  實作策略（Naive Bayes 風格）：
  - 先驗（Prior）：歷史出現頻率
  - 似然（Likelihood）：根據 gap 值估計——
      gap 大（很久沒出現）→ 似然較高（「理論上快要出現了」）
      gap 小（剛剛出現）  → 似然較低

  重要：這只是統計練習，彩票的獨立性使似然估計在理論上無意義，
        但這是標準 Bayesian 框架的應用練習。
"""

import logging
import numpy as np
import pandas as pd

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    MEGA_BALL_MIN, MEGA_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)
from src.features.base_stats import compute_gap_stats, compute_white_ball_frequency

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))


class BayesianPredictor:
    """
    貝氏更新彩票預測器。

    後驗分數 = 正規化先驗 × 正規化似然

    先驗：全歷史頻率（反映長期趨勢）
    似然：基於 Gap 的「overdue」分數（gap 越大，似然越高）
    """

    def __init__(self, prior_weight: float = 0.5, likelihood_weight: float = 0.5):
        """
        Args:
            prior_weight:      先驗的權重（0–1）
            likelihood_weight: 似然的權重（1 - prior_weight）
        """
        self.prior_weight      = prior_weight
        self.likelihood_weight = likelihood_weight
        self._posterior        = None
        self._is_fitted        = False

    def fit(self, df: pd.DataFrame) -> "BayesianPredictor":
        """
        計算先驗與似然，更新後驗分數。

        Args:
            df: 開獎資料 DataFrame
        """
        df_sorted = df.sort_values("draw_date").reset_index(drop=True)

        # ── 先驗：歷史頻率（正規化為機率）────────────────────────────────────
        freq = compute_white_ball_frequency(df_sorted)
        prior = freq.astype(float)
        prior = prior / prior.sum()  # 正規化

        # ── 似然：Gap 分數 ─────────────────────────────────────────────────
        gap_stats   = compute_gap_stats(df_sorted)
        current_gap = gap_stats["current_gap"]

        # Gap 轉換為似然：使用 softmax(gap) 使大 gap 的號碼有更高似然
        # 減去平均值避免數值溢位
        gap_vals   = current_gap.reindex(WHITE_NUMBERS).fillna(0).values.astype(float)
        gap_shifted = gap_vals - gap_vals.mean()
        likelihood  = np.exp(gap_shifted)
        likelihood  = likelihood / likelihood.sum()  # 正規化
        likelihood  = pd.Series(likelihood, index=WHITE_NUMBERS)

        # ── 後驗：加權融合 ─────────────────────────────────────────────────
        posterior = (
            self.prior_weight      * prior
            + self.likelihood_weight * likelihood
        )
        self._posterior = posterior / posterior.sum()  # 再次正規化確保和為 1

        self._is_fitted = True
        logger.info("貝氏推論後驗計算完成")
        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        選出後驗機率最高的號碼。

        Returns:
            {"white_balls": [int,...]}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        top_white = (
            self._posterior
            .nlargest(n_white)
            .index.sort_values()
            .tolist()
        )
        logger.info(f"貝氏推論推薦：白球={top_white}")
        return {"white_balls": top_white, "mega_balls": []}

    def get_posterior(self) -> pd.DataFrame:
        """回傳完整後驗分析表"""
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")
        return (
            self._posterior
            .rename("posterior_prob")
            .reset_index()
            .rename(columns={"index": "number"})
            .sort_values("posterior_prob", ascending=False)
        )
