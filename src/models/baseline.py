"""
baseline.py — L0 基準模型（頻率統計）
=======================================
最簡單的「預測」策略：挑選歷史出現頻率最高的 5 顆球。

為什麼需要基準模型？
  所有 ML 模型都必須和基準模型比較。
  若 Random Forest 的 Precision@5 與基準相同甚至更低，
  代表模型沒有學到任何有用的資訊，只是在複製頻率分佈。

使用方式：
  baseline = FrequencyBaseline()
  baseline.fit(df)
  prediction = baseline.predict(n_white=5, n_mega=1)
"""

import logging
import pandas as pd

from config.settings import LOG_LEVEL, LOG_FORMAT, SHORT_WINDOW, MID_WINDOW
from src.features.base_stats import compute_white_ball_frequency, compute_mega_ball_frequency

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class FrequencyBaseline:
    """
    基於歷史頻率的基準預測模型。

    預測邏輯：
      • 白球：選出全歷史 + 近期（短/中視窗）出現頻率最高的號碼
      • Mega：選出全歷史頻率最高的號碼
      • 最終採用「加權融合」：全歷史佔 50%，近期佔 50%
    """

    def __init__(self):
        self._white_freq_all   = None
        self._white_freq_short = None
        self._white_freq_mid   = None
        self._mega_freq_all    = None
        self._is_fitted        = False

    def fit(self, df: pd.DataFrame) -> "FrequencyBaseline":
        """
        從歷史資料計算頻率分佈。

        Args:
            df: 開獎資料 DataFrame（需包含 n1-n5, mega_number）

        Returns:
            self（支援鏈式呼叫）
        """
        self._white_freq_all   = compute_white_ball_frequency(df)
        self._white_freq_short = compute_white_ball_frequency(df, window=SHORT_WINDOW)
        self._white_freq_mid   = compute_white_ball_frequency(df, window=MID_WINDOW)
        self._mega_freq_all    = compute_mega_ball_frequency(df)
        self._is_fitted        = True

        logger.info("FrequencyBaseline.fit() 完成")
        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        根據頻率選出推薦號碼。

        融合策略：
          綜合分數 = 全歷史頻率 × 0.5 + 近期（中視窗）頻率 × 0.3 + 近期（短視窗）頻率 × 0.2
          理念：不完全忽略長期趨勢，但給近期資料更多權重

        Args:
            n_white: 要推薦幾顆白球（通常 = 5）
            n_mega:  要推薦幾顆 Mega 球（通常 = 1）

        Returns:
            dict: {"white_balls": [int,...], "mega_balls": [int,...]}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit() 再呼叫 predict()")

        # 加權融合三種頻率分佈
        combined_score = (
            self._white_freq_all.astype(float)   * 0.50
            + self._white_freq_mid.astype(float) * 0.30
            + self._white_freq_short.astype(float) * 0.20
        )

        # 取分數最高的 n_white 個白球號碼
        top_white = combined_score.nlargest(n_white).index.sort_values().tolist()
        # Mega 球只看全歷史頻率
        top_mega  = self._mega_freq_all.nlargest(n_mega).index.tolist()

        result = {"white_balls": top_white, "mega_balls": top_mega}
        logger.info(f"FrequencyBaseline 推薦：白球={top_white}，Mega={top_mega}")
        return result

    def get_probability_ranking(self) -> pd.DataFrame:
        """
        回傳所有白球號碼的綜合分數排名（用於分析）。

        Returns:
            pd.DataFrame，按分數降序排列，包含各頻率細節
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        combined = (
            self._white_freq_all.astype(float)   * 0.50
            + self._white_freq_mid.astype(float) * 0.30
            + self._white_freq_short.astype(float) * 0.20
        )

        df = pd.DataFrame({
            "number":      combined.index,
            "freq_all":    self._white_freq_all.values,
            f"freq_{MID_WINDOW}":   self._white_freq_mid.values,
            f"freq_{SHORT_WINDOW}": self._white_freq_short.values,
            "combined_score": combined.values,
        })
        return df.sort_values("combined_score", ascending=False).reset_index(drop=True)
