"""
metrics.py — 彩票預測專用評估指標
====================================

【為什麼不用普通 Accuracy？】

  彩票問題的類別嚴重不平衡：
    每期 47 顆球中，只有 5 顆出現（正樣本）= 10.6%
    剩下 42 顆未出現（負樣本）= 89.4%

  如果模型學會「把所有球預測為不出現」：
    Accuracy = 42/47 = 89.4%  ← 看起來很高，但實際上毫無用處

  我們需要能捕捉「模型預測的 5 顆球有幾顆命中」的指標。

【評估指標說明】

  1. Hit@k（命中數）
     預測機率最高的 k 顆球中，實際有幾顆開出來了。
     例如：預測 [3,7,15,22,41]，實際開 [7,12,15,33,44]
     命中 7 和 15 → Hit@5 = 2

  2. Precision@k（命中率）
     Hit@k / k
     上例：Precision@5 = 2/5 = 40%

     隨機基準值（理論期望）：
       E[Hit@5] = 5 × (5/47) ≈ 0.532 顆
       Precision@5_random = 5/47 ≈ 10.64%

  3. Expected Rank of Hits（命中號碼的平均排名）
     在模型給出的 47 顆球排名中，實際開出的 5 顆球平均排在第幾名？
     - 排名越低越好（排名 1 = 最高機率）
     - 隨機基準：(1+47)/2 = 24（排名正中間）
     - 若模型有效，應顯著低於 24

  4. Coverage（覆蓋率）
     在所有測試期中，「至少命中 1 顆球」的比例。
     反映模型是否至少比完全瞎猜好一點。

  5. Multi-Draw Aggregate Hit Distribution
     在多期的測試集中，統計 Hit@5 = 0, 1, 2, 3, 4, 5 各自的比例，
     呈現模型的整體命中分布。

  6. AUC-ROC（二元分類標準指標）
     對每個號碼計算 ROC 曲線下面積，衡量模型區分「出現/未出現」的能力。
     - 0.5 = 隨機猜測
     - 1.0 = 完美預測
     - 實際彩票問題中，能穩定超過 0.52 已相當不易
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))
WHITE_COLS    = ["n1", "n2", "n3", "n4", "n5"]
POOL_SIZE     = WHITE_BALL_MAX - WHITE_BALL_MIN + 1  # 47
N_DRAW        = 5   # 每期開出幾顆白球


# ─────────────────────────────────────────────────────────────────────────────
# 單期指標計算
# ─────────────────────────────────────────────────────────────────────────────

def hit_at_k(predicted_balls: List[int], actual_balls: List[int], k: int = 5) -> int:
    """
    計算單期的命中數（Hit@k）。

    Args:
        predicted_balls: 模型推薦的白球列表（可以超過 k 個，只取前 k 個）
        actual_balls:    本期實際開出的白球列表
        k:               取前幾個推薦號碼來計算

    Returns:
        int: 命中顆數（0–k）

    範例：
        hit_at_k([3,7,15,22,41], [7,12,15,33,44], k=5) → 2
    """
    top_k  = set(predicted_balls[:k])
    actual = set(actual_balls)
    return len(top_k & actual)


def precision_at_k(predicted_balls: List[int], actual_balls: List[int], k: int = 5) -> float:
    """
    計算單期的 Precision@k = Hit@k / k。

    Returns:
        float: 命中率（0.0–1.0）
    """
    return hit_at_k(predicted_balls, actual_balls, k) / k


def expected_rank_of_hits(
    prob_scores: Dict[int, float],
    actual_balls: List[int],
) -> float:
    """
    計算實際開出號碼在模型排名中的平均位置。

    Args:
        prob_scores:  {號碼: 預測分數} 字典（分數越高 = 越推薦）
        actual_balls: 本期實際開出的白球列表

    Returns:
        float: 命中號碼的平均排名（1 = 排名第一，47 = 排名最後）
               若分數字典為空則回傳 NaN

    理解：
        隨機基準 = (1+47)/2 = 24.0
        若模型有效，命中號碼的平均排名應顯著 < 24
    """
    if not prob_scores:
        return float("nan")

    # 按分數由高到低排序，得到每個號碼的排名（1 = 最高分）
    sorted_numbers = sorted(prob_scores.keys(), key=lambda n: -prob_scores[n])
    rank_map = {num: rank + 1 for rank, num in enumerate(sorted_numbers)}

    ranks = [rank_map[b] for b in actual_balls if b in rank_map]
    return float(np.mean(ranks)) if ranks else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 多期彙總指標
# ─────────────────────────────────────────────────────────────────────────────

class FoldMetrics:
    """
    單一 Fold 的評估結果彙整器。

    使用方式：
        metrics = FoldMetrics(fold_info)
        for draw in test_draws:
            metrics.add(predicted_balls, actual_balls, prob_scores)
        summary = metrics.summary()
    """

    def __init__(self, fold_info=None, model_name: str = ""):
        self.fold_info  = fold_info
        self.model_name = model_name
        self._hits:  List[int]   = []
        self._ranks: List[float] = []

    def add(
        self,
        predicted_balls: List[int],
        actual_balls:    List[int],
        prob_scores:     Optional[Dict[int, float]] = None,
        k: int = 5,
    ) -> None:
        """
        登記一期的評估結果。

        Args:
            predicted_balls: 模型推薦號碼
            actual_balls:    實際開出號碼
            prob_scores:     {號碼: 分數} 選填（若提供則計算 rank 指標）
            k:               取前幾個推薦號碼
        """
        self._hits.append(hit_at_k(predicted_balls, actual_balls, k))

        if prob_scores:
            rank = expected_rank_of_hits(prob_scores, actual_balls)
            self._ranks.append(rank)

    def summary(self) -> dict:
        """
        彙整所有已登記期次的評估指標。

        Returns:
            dict 包含以下欄位：
              n_draws:        評估的期次數
              mean_hits:      平均命中數
              precision_at5:  平均 Precision@5
              coverage:       至少命中 1 顆的比例
              hit_dist:       命中數分布（0–5 各佔幾%）
              mean_rank:      命中號碼平均排名（若有分數）
              vs_random_hits: 與隨機基準比較（+正 = 優於隨機）
              hit_list:       每期命中數列表（用於繪圖）
        """
        if not self._hits:
            return {}

        hits_arr    = np.array(self._hits)
        random_base = N_DRAW * N_DRAW / POOL_SIZE   # = 5*5/47 ≈ 0.532

        # 命中數分布
        hit_counts = {i: int(np.sum(hits_arr == i)) for i in range(N_DRAW + 1)}
        hit_pct    = {i: f"{v/len(hits_arr)*100:.1f}%" for i, v in hit_counts.items()}

        result = {
            "model":           self.model_name,
            "n_draws":         len(hits_arr),
            "mean_hits":       float(np.mean(hits_arr)),
            "precision_at5":   float(np.mean(hits_arr)) / N_DRAW,
            "coverage":        float(np.mean(hits_arr > 0)),
            "hit_dist_count":  hit_counts,
            "hit_dist_pct":    hit_pct,
            "vs_random_hits":  float(np.mean(hits_arr)) - random_base,
            "random_baseline": random_base,
            "hit_list":        self._hits,
        }

        if self._ranks:
            result["mean_rank"]       = float(np.nanmean(self._ranks))
            result["random_rank"]     = (POOL_SIZE + 1) / 2   # = 24.0
            result["vs_random_rank"]  = result["mean_rank"] - result["random_rank"]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# Fold 結果彙整 & 報表列印
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_fold_results(fold_results: List[dict]) -> dict:
    """
    將多個 Fold 的 FoldMetrics.summary() 彙整為整體統計。

    Args:
        fold_results: 每個元素為一個 Fold 的 summary() 結果

    Returns:
        dict 包含跨 Fold 的均值、標準差、與逐 Fold 的序列
    """
    if not fold_results:
        return {}

    mean_hits_per_fold = [r["mean_hits"]       for r in fold_results]
    precision_per_fold = [r["precision_at5"]   for r in fold_results]
    coverage_per_fold  = [r["coverage"]        for r in fold_results]

    agg = {
        "n_folds":               len(fold_results),
        "mean_hits_avg":         float(np.mean(mean_hits_per_fold)),
        "mean_hits_std":         float(np.std(mean_hits_per_fold)),
        "precision_at5_avg":     float(np.mean(precision_per_fold)),
        "precision_at5_std":     float(np.std(precision_per_fold)),
        "coverage_avg":          float(np.mean(coverage_per_fold)),
        "random_baseline_hits":  N_DRAW * N_DRAW / POOL_SIZE,
        "random_baseline_prec":  N_DRAW / POOL_SIZE,
        "mean_hits_per_fold":    mean_hits_per_fold,
        "precision_per_fold":    precision_per_fold,
    }

    # 是否顯著優於隨機？（單尾 t-test）
    all_hits = []
    for r in fold_results:
        all_hits.extend(r.get("hit_list", []))

    if len(all_hits) >= 10:
        t_stat, p_value = stats.ttest_1samp(
            all_hits,
            popmean = N_DRAW * N_DRAW / POOL_SIZE   # H0: 均值 = 隨機基準
        )
        agg["ttest_vs_random"] = {
            "t_statistic":  float(t_stat),
            "p_value":      float(p_value),
            "significant":  p_value < 0.05,
            "note": ("顯著優於隨機（p<0.05）" if p_value < 0.05
                     else "未顯著優於隨機"),
        }

    return agg


def print_fold_report(
    model_name:   str,
    fold_results: List[dict],
    fold_infos:   list,
    agg:          dict,
) -> None:
    """
    印出格式化的 Walk-Forward 評估報表。

    輸出格式範例：
    ══════════════════════════════════════════════
     模型：FrequencyBaseline — Walk-Forward 評估
    ══════════════════════════════════════════════
    Fold │ 期間                     │ 測試期數 │ 平均命中 │ Prec@5
    ─────┼──────────────────────────┼──────────┼──────────┼───────
       1 │ 2020-01 → 2022-06        │    52   │   0.52   │ 10.4%
       2 │ 2020-01 → 2022-12        │    53   │   0.58   │ 11.6%
    ...
    ──────────────────────────────────────────────
    整體平均命中：0.55 ± 0.08（隨機基準：0.53）
    整體 Prec@5：11.1%（隨機基準：10.6%）
    """
    SEP = "═" * 62

    print(f"\n{SEP}")
    print(f" 模型：{model_name} — Walk-Forward 評估")
    print(SEP)
    print(f"{'Fold':>4} │ {'訓練集終點':<12} {'測試集範圍':<24} │ {'n_test':>6} │ {'Avg Hits':>8} │ {'Prec@5':>7}")
    print("─" * 62)

    for fold_res, fold_info in zip(fold_results, fold_infos):
        period_str = f"{fold_info.test_start} → {fold_info.test_end}"
        print(
            f"{fold_info.fold_idx:>4} │ {fold_info.train_end:<12} "
            f"{period_str:<24} │ {fold_res['n_draws']:>6} │ "
            f"{fold_res['mean_hits']:>8.3f} │ "
            f"{fold_res['precision_at5']*100:>6.2f}%"
        )

    print("─" * 62)
    rand_h    = agg.get("random_baseline_hits", N_DRAW * N_DRAW / POOL_SIZE)
    rand_p    = agg.get("random_baseline_prec", N_DRAW / POOL_SIZE)
    avg_hits  = agg.get("mean_hits_avg", 0)
    avg_prec  = agg.get("precision_at5_avg", 0)
    std_hits  = agg.get("mean_hits_std", 0)

    print(f"  整體平均命中：{avg_hits:.3f} ± {std_hits:.3f}  （隨機基準：{rand_h:.3f}）")
    print(f"  整體 Prec@5 ：{avg_prec*100:.2f}%            （隨機基準：{rand_p*100:.2f}%）")

    if "ttest_vs_random" in agg:
        t = agg["ttest_vs_random"]
        print(f"  統計檢定    ：t={t['t_statistic']:.3f}, p={t['p_value']:.4f} — {t['note']}")

    print(SEP)


def print_model_comparison_table(
    all_model_agg: Dict[str, dict],
) -> None:
    """
    將所有模型的 Walk-Forward 彙整結果並排顯示，方便比較。

    Args:
        all_model_agg: {模型名稱: aggregate_fold_results() 的結果}
    """
    SEP = "═" * 72
    rand_h = N_DRAW * N_DRAW / POOL_SIZE
    rand_p = N_DRAW / POOL_SIZE

    print(f"\n{SEP}")
    print(" 各模型 Walk-Forward 評估結果總覽")
    print(SEP)
    print(
        f"{'模型':<18} │ {'Avg Hits':>9} │ {'Prec@5':>7} │ "
        f"{'vs Random':>10} │ {'Coverage':>8} │ {'p-value':>8}"
    )
    print("─" * 72)

    # 按平均命中數排序
    sorted_models = sorted(
        all_model_agg.items(),
        key=lambda x: x[1].get("mean_hits_avg", 0),
        reverse=True
    )

    for model_name, agg in sorted_models:
        avg_h = agg.get("mean_hits_avg", 0)
        avg_p = agg.get("precision_at5_avg", 0)
        cov   = agg.get("coverage_avg", 0)
        delta = avg_h - rand_h
        delta_str = f"{delta:+.3f}"

        p_str = "N/A"
        if "ttest_vs_random" in agg:
            p_val = agg["ttest_vs_random"]["p_value"]
            p_str = f"{p_val:.4f}" + ("*" if p_val < 0.05 else " ")

        print(
            f"{model_name:<18} │ {avg_h:>9.3f} │ {avg_p*100:>6.2f}% │ "
            f"{delta_str:>10} │ {cov*100:>7.1f}% │ {p_str:>8}"
        )

    print("─" * 72)
    print(f"{'隨機基準':<18} │ {rand_h:>9.3f} │ {rand_p*100:>6.2f}% │ {'0.000':>10} │ {'—':>8} │")
    print(SEP)
    print("  * p < 0.05 表示與隨機基準的差異在統計上顯著")
    print("  注意：彩票為獨立隨機事件，即使統計顯著亦不代表實際預測能力")
