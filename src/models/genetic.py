"""
genetic.py — 遺傳演算法號碼組合優化
======================================
概念：
  用遺傳演算法（GA）在 47 個號碼中搜尋「最佳」5 球組合。

  演化流程：
    1. 初始化種群：隨機生成 N 個「染色體」（每條 = 5 個不重複號碼）
    2. 適應度評估：計算每個組合的歷史「命中分數」
         fitness = 組合中每顆球的歷史頻率之和
                  + 組合在歷史中共現次數加權
    3. 選擇：輪盤賭選擇（fitness 越高，被選中機率越高）
    4. 交叉：兩條染色體交換部分號碼，生成子代（確保不重複）
    5. 突變：以機率 mutation_rate 將某顆球替換為其他號碼
    6. 重複 2–5，直到達到 n_generations 代

學習價值：
  理解進化計算、適應度函數設計、選擇/交叉/突變算子。

  重要提示：彩票是獨立事件，GA 在理論上無法提高中獎率，
  此實作純為演算法學習練習。
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Tuple

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))
WHITE_COLS    = ["n1", "n2", "n3", "n4", "n5"]

Chromosome = List[int]   # 長度為 5 的號碼列表（已排序）


class GeneticPredictor:
    """
    遺傳演算法號碼組合進化搜尋器。

    適應度函數：
      f(組合) = α × Σ freq(ball)
              + β × co_occurrence_bonus(組合, 歷史)

    其中 co_occurrence_bonus = 組合中所有球對（C(5,2)=10對）在歷史中共同出現的次數之和。
    """

    def __init__(
        self,
        population_size: int   = 200,
        n_generations:   int   = 300,
        mutation_rate:   float = 0.15,
        elite_ratio:     float = 0.10,
        freq_weight:     float = 0.6,
        cooccur_weight:  float = 0.4,
        random_seed:     int   = 42,
    ):
        """
        Args:
            population_size: 種群大小（染色體數量）
            n_generations:   演化代數
            mutation_rate:   每顆球突變的機率
            elite_ratio:     精英保留比例（直接複製到下一代）
            freq_weight:     頻率適應度的權重
            cooccur_weight:  共現適應度的權重
            random_seed:     隨機種子（保證可重現）
        """
        self.population_size = population_size
        self.n_generations   = n_generations
        self.mutation_rate   = mutation_rate
        self.elite_ratio     = elite_ratio
        self.freq_weight     = freq_weight
        self.cooccur_weight  = cooccur_weight
        self.rng             = np.random.default_rng(random_seed)

        self._freq_arr   = None   # shape=(47,)  每個號碼正規化頻率
        self._cooccur    = None   # shape=(47,47) 共現矩陣
        self._best_combo = None
        self._is_fitted  = False

    # ──────────────────────────────────────────────────────────────────────────
    # 資料分析
    # ──────────────────────────────────────────────────────────────────────────

    def _build_frequency(self, df: pd.DataFrame) -> np.ndarray:
        """計算每個號碼的正規化歷史頻率（shape=(47,)）。"""
        counts = np.zeros(len(WHITE_NUMBERS), dtype=np.float64)
        for c in WHITE_COLS:
            for num in df[c]:
                counts[int(num) - 1] += 1
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _build_cooccurrence(self, df: pd.DataFrame) -> np.ndarray:
        """
        建立 47×47 共現矩陣。
        cooccur[i][j] = 號碼 i+1 和號碼 j+1 在同一期出現的次數。
        """
        n = len(WHITE_NUMBERS)
        mat = np.zeros((n, n), dtype=np.float64)
        for _, row in df.iterrows():
            balls = [int(row[c]) - 1 for c in WHITE_COLS]  # 0-index
            for a in range(len(balls)):
                for b in range(a + 1, len(balls)):
                    mat[balls[a]][balls[b]] += 1
                    mat[balls[b]][balls[a]] += 1
        # 正規化到 [0, 1]
        max_val = mat.max()
        if max_val > 0:
            mat /= max_val
        return mat

    # ──────────────────────────────────────────────────────────────────────────
    # 遺傳演算法元件
    # ──────────────────────────────────────────────────────────────────────────

    def _random_chromosome(self) -> Chromosome:
        """隨機生成一條染色體（5個不重複號碼）。"""
        return sorted(self.rng.choice(WHITE_NUMBERS, size=5, replace=False).tolist())

    def _fitness(self, chrom: Chromosome) -> float:
        """
        計算適應度分數。

        freq_score    = Σ freq[ball-1]
        cooccur_score = Σ cooccur[ball_i-1][ball_j-1]（所有球對）
        """
        indices = [n - 1 for n in chrom]  # 0-index

        freq_score = sum(self._freq_arr[i] for i in indices)

        cooccur_score = 0.0
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                cooccur_score += self._cooccur[indices[a]][indices[b]]

        return (
            self.freq_weight    * freq_score
            + self.cooccur_weight * cooccur_score
        )

    def _tournament_select(
        self,
        population:  List[Chromosome],
        fitnesses:   np.ndarray,
        tournament_k: int = 3,
    ) -> Chromosome:
        """錦標賽選擇：隨機取 k 個，選其中 fitness 最高者。"""
        idx = self.rng.choice(len(population), size=tournament_k, replace=False)
        best = idx[np.argmax(fitnesses[idx])]
        return population[best]

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        交叉算子：合併兩個父代號碼池，從中不重複抽取 5 顆，生成兩個子代。
        """
        gene_pool = list(set(parent1 + parent2))

        if len(gene_pool) < 10:
            # 基因多樣性不足時，補充隨機號碼
            extra = [n for n in WHITE_NUMBERS if n not in gene_pool]
            gene_pool += self.rng.choice(extra, size=10 - len(gene_pool), replace=False).tolist()

        child1 = sorted(self.rng.choice(gene_pool, size=5, replace=False).tolist())
        child2 = sorted(self.rng.choice(gene_pool, size=5, replace=False).tolist())
        return child1, child2

    def _mutate(self, chrom: Chromosome) -> Chromosome:
        """
        突變算子：對每顆球以機率 mutation_rate 替換為另一顆未選中的球。
        """
        chrom = list(chrom)
        for i in range(len(chrom)):
            if self.rng.random() < self.mutation_rate:
                available = [n for n in WHITE_NUMBERS if n not in chrom]
                if available:
                    chrom[i] = int(self.rng.choice(available))
        return sorted(chrom)

    # ──────────────────────────────────────────────────────────────────────────
    # 主流程
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "GeneticPredictor":
        """
        從歷史資料計算頻率與共現矩陣，執行遺傳演算法演化。

        Args:
            df: 開獎資料 DataFrame
        """
        df = df.sort_values("draw_date").reset_index(drop=True)
        logger.info("遺傳演算法：計算頻率與共現矩陣...")

        self._freq_arr = self._build_frequency(df)
        self._cooccur  = self._build_cooccurrence(df)

        # 初始化種群
        population: List[Chromosome] = [
            self._random_chromosome() for _ in range(self.population_size)
        ]

        n_elite = max(1, int(self.population_size * self.elite_ratio))
        logger.info(f"遺傳演算法：開始演化（種群={self.population_size}，代數={self.n_generations}）...")

        best_fitness   = -np.inf
        best_chrom     = population[0]

        for gen in range(self.n_generations):
            fitnesses = np.array([self._fitness(c) for c in population])

            # 追蹤最佳個體
            gen_best_idx = int(np.argmax(fitnesses))
            if fitnesses[gen_best_idx] > best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chrom   = population[gen_best_idx]

            # 精英保留（排名前 n_elite 的染色體直接複製）
            elite_indices = np.argsort(fitnesses)[::-1][:n_elite]
            new_population: List[Chromosome] = [population[i] for i in elite_indices]

            # 填充剩餘種群
            while len(new_population) < self.population_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                new_population.append(self._mutate(c1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(c2))

            population = new_population

            if (gen + 1) % 50 == 0:
                logger.info(f"  第 {gen+1} 代，最佳 fitness={best_fitness:.4f}，組合={best_chrom}")

        self._best_combo = best_chrom
        self._is_fitted  = True
        logger.info(f"遺傳演算法演化完成，最佳組合={self._best_combo}，fitness={best_fitness:.4f}")
        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        回傳遺傳演算法找到的最佳號碼組合。

        Returns:
            {"white_balls": [int,...], "mega_balls": []}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        result = sorted(self._best_combo)[:n_white]

        # 用個別號碼的適應度分數（頻率）作為機率 proxy
        total_f = self._freq_arr.sum()
        proba_map = {
            WHITE_NUMBERS[i]: float(self._freq_arr[i] / total_f) if total_f > 0 else 0.0
            for i in range(len(WHITE_NUMBERS))
        }

        logger.info(f"遺傳演算法推薦：白球={result}")
        return {"white_balls": result, "mega_balls": [], "proba": proba_map}
