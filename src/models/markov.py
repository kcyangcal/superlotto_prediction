"""
markov.py — 馬可夫鏈預測
==========================
概念：
  建立一階馬可夫轉移模型：
  「上一期出現了號碼 A，下一期出現號碼 B 的機率是多少？」

  轉移矩陣 T[i][j] = 歷史上「第 i 號出現後，下一期第 j 號也出現」的次數

  預測方式：
    1. 取最近一期已出現的 5 顆白球
    2. 對每顆球查轉移矩陣，得到下一期各號碼的條件機率
    3. 將 5 顆球的轉移機率加總，選出機率最高的號碼

學習價值：
  了解馬可夫性質（無記憶性）、轉移矩陣建構、條件機率應用
"""

import logging
import numpy as np
import pandas as pd

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    MEGA_BALL_MIN, MEGA_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_COLS = ["n1", "n2", "n3", "n4", "n5"]
N_WHITE    = WHITE_BALL_MAX - WHITE_BALL_MIN + 1   # 47
N_MEGA     = MEGA_BALL_MAX  - MEGA_BALL_MIN  + 1   # 27


class MarkovPredictor:
    """
    馬可夫鏈號碼轉移預測器。

    轉移矩陣大小：47 × 47（白球）
    T[i][j] = 「號碼 i+1 出現後，下一期號碼 j+1 出現」的次數
    """

    def __init__(self):
        # 白球轉移矩陣（47×47）
        self._white_transition = np.zeros((N_WHITE, N_WHITE), dtype=np.float64)
        # Mega 球轉移矩陣（27×27）
        self._mega_transition  = np.zeros((N_MEGA, N_MEGA),   dtype=np.float64)
        self._last_draw        = None   # 最後一期的號碼（用於預測）
        self._is_fitted        = False

    def fit(self, df: pd.DataFrame) -> "MarkovPredictor":
        """
        從歷史資料建立轉移矩陣。

        Args:
            df: 開獎資料 DataFrame，已按 draw_date 升序排列
        """
        df = df.sort_values("draw_date").reset_index(drop=True)

        for i in range(len(df) - 1):
            curr_row = df.iloc[i]
            next_row = df.iloc[i + 1]

            curr_white = [int(curr_row[c]) - 1 for c in WHITE_COLS]  # 轉為 0-index
            next_white = [int(next_row[c]) - 1 for c in WHITE_COLS]

            # 更新轉移矩陣：當期每顆球 → 下期每顆球
            for c in curr_white:
                for n in next_white:
                    self._white_transition[c][n] += 1

            # Mega 球轉移
            curr_mega = int(curr_row["mega_number"]) - 1
            next_mega = int(next_row["mega_number"]) - 1
            self._mega_transition[curr_mega][next_mega] += 1

        # 正規化每一行（行和為 1，代表條件機率）
        # 若某號碼從未出現，用均等分佈填充（Laplace smoothing）
        for i in range(N_WHITE):
            row_sum = self._white_transition[i].sum()
            if row_sum == 0:
                self._white_transition[i] = np.ones(N_WHITE) / N_WHITE
            else:
                self._white_transition[i] /= row_sum

        for i in range(N_MEGA):
            row_sum = self._mega_transition[i].sum()
            if row_sum == 0:
                self._mega_transition[i] = np.ones(N_MEGA) / N_MEGA
            else:
                self._mega_transition[i] /= row_sum

        # 記錄最後一期的號碼（作為預測起點）
        last = df.iloc[-1]
        self._last_draw = {
            "white": [int(last[c]) for c in WHITE_COLS],
            "mega":  int(last["mega_number"]),
        }

        self._is_fitted = True
        logger.info("馬可夫鏈轉移矩陣建立完成")
        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        根據最後一期號碼，利用轉移矩陣預測下一期。

        預測邏輯：
          將最後一期每顆白球的轉移機率向量加總，
          得到「下一期各號碼的綜合出現機率」，選 top-5。

        Returns:
            {"white_balls": [int,...], "mega_balls": [int,...]}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        # 加總最後一期所有白球的轉移機率
        combined_prob = np.zeros(N_WHITE)
        for num in self._last_draw["white"]:
            combined_prob += self._white_transition[num - 1]  # 轉為 0-index

        # 選機率最高的 n_white 顆
        top_indices = np.argsort(combined_prob)[::-1][:n_white]
        top_white   = sorted([idx + 1 for idx in top_indices])  # 轉回 1-index

        # Mega 球轉移
        mega_idx  = self._last_draw["mega"] - 1
        mega_prob = self._mega_transition[mega_idx]
        top_mega  = [int(np.argmax(mega_prob)) + 1]

        # 正規化機率 map（供 multi-ticket 生成使用）
        total_w = combined_prob.sum()
        norm_w  = combined_prob / total_w if total_w > 0 else combined_prob
        proba_map      = {idx + 1: float(norm_w[idx])    for idx in range(N_WHITE)}
        mega_proba_map = {idx + 1: float(mega_prob[idx]) for idx in range(N_MEGA)}

        logger.info(f"馬可夫鏈推薦：白球={top_white}，Mega={top_mega}")
        return {
            "white_balls": top_white,
            "mega_balls":  top_mega,
            "proba":       proba_map,
            "mega_proba":  mega_proba_map,
        }

    def get_transition_prob(self, number: int) -> pd.Series:
        """
        回傳某號碼的轉移機率向量（用於分析）。

        Args:
            number: 白球號碼（1–47）

        Returns:
            pd.Series，index=目標號碼，values=轉移機率
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")
        probs = self._white_transition[number - 1]
        return pd.Series(probs, index=range(1, N_WHITE + 1), name=f"from_{number}")
