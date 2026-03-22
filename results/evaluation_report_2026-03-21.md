# SuperLotto Plus — 模型評估完整報告
**日期**：2026-03-21 ｜ **資料範圍**：2020-01-01 後，649 期
**評估方式**：True Walk-Forward Rolling（逐期預測 → 揭曉 → 更新，共 150 期測試）
**初始訓練期數**：200 期

---

## 一、目前使用的特徵（現行模型輸入）

### 1.1 各模型特徵對照表

| 模型 | 特徵類型 | 維度 | 說明 |
|------|----------|------|------|
| FrequencyBaseline | 全史頻率 + 近期頻率（SHORT/MID/LONG window） | 1D排名 | 頻率加權分數排名，取 top-5 |
| MarkovPredictor | 一階轉移機率矩陣（號碼→號碼）| 47×47 | 上期球 → 下期球的條件機率 |
| BayesianPredictor | Gap 作為似然，Beta(α, β) 先驗 | 47 | 距上次出現期數作為超期信號 |
| KNNPredictor | current_gap(47) + freq_30(47) | 94 | 特徵向量相似度 k 近鄰 |
| DecisionTreePredictor | current_gap(47) + freq_30(47) | 94 | 多輸出 DT，每顆球獨立二元分類 |
| RF / XGBoost | gap(47) + freq(各窗口) + 模式特徵 | 94+ | MultiOutputClassifier，47顆球各一個分類器 |
| GeneticPredictor | 全史頻率 + 共現矩陣 | 47×47 | 適應度 = α×頻率 + β×共現 |
| MonteCarloPredictor | 全史頻率作為抽樣權重 | 47 | 10萬次加權模擬計數 |
| EnsemblePredictor | 7個子模型投票 | — | 加權投票（frequency/MC/markov/bayesian/KNN/DT/genetic） |
| MLPPredictor | gap(47) + freq_30(47) | 94→128→64→47 | PyTorch MLP，partial_fit 增量學習 |
| LSTMPredictor | gap(47) + freq_30(47) × seq_len=20 | 94×20→128→47 | PyTorch LSTM，sequence modeling |

### 1.2 現行特徵的完整清單

**base_stats.py 計算的特徵：**
- `current_gap[n]`：每顆球距上次出現的期數（47 維）
- `avg_gap[n]`：每顆球的歷史平均間距
- `freq_short[n]`：近 SHORT_WINDOW 期出現次數
- `freq_mid[n]`：近 MID_WINDOW 期出現次數
- `freq_long[n]`：全史出現次數

**pattern_stats.py 計算的特徵（每期整體統計）：**
- `white_sum`：5顆白球加總
- `white_mean`：白球平均值
- `odd_count / even_count`：奇偶球數量
- `low_count / high_count`：低區(1-23) / 高區(24-47) 球數
- `consecutive_pairs`：相鄰差值為1的連號對數
- `number_range`：最大值 − 最小值
- `unique_last_digits`：個位數多樣性

---

## 二、特徵信號分析結果（2026-03-21）

**方法**：Point-Biserial Correlation（特徵 vs「下期是否出現」的 0/1 標籤）
**資料量**：28,106 行（599 期 × 47 顆球）

| 特徵 | 相關係數 r | p 值 | 顯著性 | 出現時均值 | 未出現均值 | 差值 |
|------|-----------|------|--------|-----------|-----------|------|
| `ball_zone`（號碼區間 0-4） | -0.0198 | 9.23e-04 | *** | 1.794 | 1.882 | -0.087 |
| `freq_100` | +0.0097 | 0.103 | — | 0.107 | 0.106 | +0.001 |
| `ball_is_odd` | -0.0097 | 0.106 | — | 0.497 | 0.512 | -0.016 |
| `cooccur_last`（上期共現頻率） | +0.0095 | 0.110 | — | 0.048 | 0.047 | +0.001 |
| `freq_10` | +0.0092 | 0.123 | — | 0.109 | 0.106 | +0.003 |
| `current_gap` | -0.0062 | 0.300 | — | 8.416 | 8.601 | -0.184 |
| `appeared_last`（上期是否出現） | +0.0052 | 0.383 | — | 0.111 | 0.106 | +0.005 |
| `gap_ratio`（超期比例） | -0.0033 | 0.580 | — | 0.945 | 0.957 | -0.012 |
| `last_sum / last_odd / last_high / last_consec` | 0.0000 | 1.000 | — | — | — | 0 |

### 關鍵結論

1. **只有 `ball_zone`（p<0.001）達統計顯著**，但 r=-0.020 效果量極小（實際無預測力）。
2. **18 個特徵中 17 個未達 p<0.05**，且所有 |r| < 0.02。
3. **`last_sum`、`last_odd_cnt` 等整期統計完全無信號**（r=0.000）：因為這些對同一期的 47 顆球值相同，無法區分哪顆球出現。
4. **「超期就會出現」的迷思不成立**：gap_ratio > 2x 的出現率（10.09%）反而低於 1x（11.1%）。
5. **彩票具備真正的獨立性**：歷史特徵無法有效預測下期號碼。

---

## 三、模型評估結果

### 3.1 True Walk-Forward Rolling 評估（150 期，2026-03-21）

**設定**：初始訓練 200 期，每期預測後更新，各模型依 retrain_every 週期重訓
**隨機基準**：Avg Hits = 0.5319（E[命中] = 5 × 5/47）

| 模型 | Avg Hits | vs 隨機 | Prec@5 | EV/期 | 淨EV | ROI% | retrain_every |
|------|----------|---------|--------|-------|------|------|---------------|
| **XGB** | **0.667** | **+0.135** | 13.33% | $0.184 | $-0.816 | -81.6% | 3 |
| **RF** | 0.627 | +0.095 | 12.53% | $0.174 | $-0.826 | -82.6% | 3 |
| DecisionTree | 0.607 | +0.075 | 12.13% | $0.171 | $-0.829 | -82.9% | 1 |
| Frequency | 0.587 | +0.055 | 11.73% | $0.140 | $-0.860 | -86.0% | 1 |
| Genetic | 0.587 | +0.055 | 11.73% | $0.085 | $-0.915 | -91.5% | 10 |
| KNN | 0.567 | +0.035 | 11.33% | $0.085 | $-0.915 | -91.5% | 1 |
| Ensemble | 0.567 | +0.035 | 11.33% | $0.083 | $-0.918 | -91.8% | 30 |
| MonteCarlo | 0.560 | +0.028 | 11.20% | $0.060 | $-0.940 | -94.0% | 5 |
| Markov | 0.533 | +0.001 | 10.67% | $0.133 | $-0.867 | -86.7% | 1 |
| MLP | 0.513 | -0.019 | 10.27% | $0.074 | $-0.926 | -92.6% | 10 |
| LSTM | 0.500 | -0.032 | 10.00% | $0.158 | $-0.842 | -84.2% | 10 |
| Bayesian | 0.480 | -0.052 | 9.60% | $0.073 | $-0.927 | -92.7% | 1 |
| **隨機基準** | **0.532** | 0 | 10.64% | — | — | — | — |

### 3.2 命中數分布（Hit@5 = 0~5）

| 模型 | 0中 | 1中 | 2中 | 3中 | 4中+ |
|------|-----|-----|-----|-----|------|
| XGB | 46.0% | 42.0% | 11.3% | 0.7% | 0.0% |
| RF | 47.3% | 43.3% | 8.7% | 0.7% | 0.0% |
| DecisionTree | 48.7% | 42.7% | 8.0% | 0.7% | 0.0% |
| LSTM | 56.7% | 37.3% | 5.3% | 0.7% | 0.0% |
| 隨機期望 | ~49.4% | ~41.2% | ~9.0% | ~0.4% | ~0.0% |

### 3.3 Concept Drift（移動平均命中數 @50期）

| 模型 | 前50期 | 中50期 | 後50期 | 趨勢 |
|------|--------|--------|--------|------|
| XGB | 0.620 | 0.725 | 0.647 | 高峰在中段 |
| RF | 0.520 | 0.608 | 0.745 | 持續改善 ↑ |
| DecisionTree | 0.560 | 0.686 | 0.549 | 中段衰退 |
| Frequency | 0.420 | 0.745 | 0.608 | 先低後高 |
| MLP | 0.500 | 0.510 | 0.549 | 小幅改善 |
| LSTM | 0.520 | 0.471 | 0.510 | 中段衰退 |

### 3.4 Multi-Ticket 策略比較（買 1~5 組）

每期生成前 K 組最高機率組合（`n_candidates=12`，C(12,5)=792 種），依機率分數排名選 top-K。

| 模型 | 買1組 ROI | 買2組 ROI | 買3組 ROI | 買4組 ROI | 買5組 ROI |
|------|----------|----------|----------|----------|----------|
| XGB | -81.6% | -81.6% | -81.6% | -81.6% | -81.6% |
| RF | -82.6% | -82.6% | -82.6% | -82.6% | -82.6% |
| **Ensemble** | -91.8% | -83.2% | **-80.6%** | -83.4% | -85.2% |
| LSTM | -84.2% | -83.8% | -83.7% | -83.7% | -82.1% |
| DecisionTree | -82.9% | -86.0% | -87.7% | -88.4% | -85.6% |
| Genetic | -91.5% | -83.4% | -83.2% | -83.3% | -84.8% |

**最佳策略**：Ensemble 買 3 組，ROI = -80.6%（所有模型×策略中損失最少）
**XGB/RF 的 ROI 不隨買幾組改變**：因為其機率分布集中，多組票幾乎重複

---

## 四、評估方法說明

### 4.1 True Walk-Forward Rolling

```
for i in range(initial_train, total_draws):
    model.fit(df[:i])          # 使用截至上期的完整歷史
    pred = model.predict()     # 預測第 i 期
    actual = df[i]             # 揭曉實際結果
    record_hit(pred, actual)   # 記錄命中數與獎金
    # 下一輪 fit 會包含第 i 期資料
```

**重訓週期**（MODEL_RETRAIN_DEFAULTS）：
- 快速模型（Bayesian/Markov/Frequency/KNN/DecisionTree）：每1期重訓
- 中速（RF/XGB）：每3期
- 慢速（Ensemble）：每30期
- 非常慢（MonteCarlo）：每5期，（Genetic/MLP/LSTM）：每10期

### 4.2 Multi-Ticket 生成邏輯

```python
def generate_top_k_tickets(white_proba, mega_proba, n_tickets=5, n_candidates=12):
    # 1. 取機率最高的 12 顆球作為候選
    candidates = sorted(white_proba, key=lambda b: -white_proba[b])[:12]
    # 2. 枚舉 C(12,5)=792 種組合，依機率分數排名
    scored = [(sum(white_proba[b] for b in combo), combo)
              for combo in combinations(candidates, 5)]
    scored.sort(reverse=True)
    # 3. 回傳 top-K 組
    return [{"white_balls": wb, "mega_ball": best_mega} for _, wb in scored[:n_tickets]]
```

### 4.3 EV 計算

獎金表（實際 Pari-Mutuel 資料）：

| 命中 | 獎金 |
|------|------|
| 5+Mega | $7,000,000 |
| 5 | $25,241 |
| 4+Mega | $1,402 |
| 4 | $108 |
| 3+Mega | $61 |
| 3 | $11 |
| 2+Mega | $12 |
| 1+Mega | $2 |
| 0+Mega | $1 |

`EV/期 = mean(sum of prizes across all draws)`
`Net EV = EV - 票價($1 × N張)`
`ROI% = Net EV / 成本 × 100`

---

## 五、檔案索引

| 檔案 | 說明 |
|------|------|
| `results/rolling_eval_summary.csv` | 12模型 × 25指標，每次執行自動追加 |
| `results/rolling_150draws_classic_models_2026-03-21.txt` | 10個經典模型完整輸出 |
| `results/feature_analysis.csv` | 18個候選特徵的 PB相關係數排名 |
| `scripts/run_rolling_nn.py` | 主要評估腳本（含 multi-ticket 策略） |
| `scripts/feature_analysis.py` | 特徵信號分析腳本 |

---

*報告生成：2026-03-21*
