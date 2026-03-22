# SuperLotto Plus ML Pipeline — 新 Chat 接續指引

> 本文件提供足夠的脈絡，讓新的 chat session 能直接接續專案工作。

---

## 一、專案概覽

**目的**：純學習用的 ML 管線，練習資料工程、爬蟲、特徵工程、ML 建模與評估。
**資料**：加州 SuperLotto Plus 歷史開獎（2000–2025，約 2,691 期；評估用 2020 後 649 期）
**語言**：Python 3.12，虛擬環境 `.venv`
**路徑**：`d:\010_Github\superlotto_prediction`
**DB**：`data/superlotto.db`（SQLite，git 忽略）

> ⚠️ 彩票為獨立均勻隨機事件，任何模型均無法預測結果。本專案學習價值在技術本身。

---

## 二、技術棧

| 類別        | 套件                                             |
|-------------|--------------------------------------------------|
| 資料處理    | pandas, numpy, sqlite3                           |
| 機器學習    | scikit-learn, xgboost                            |
| 深度學習    | torch 2.10.0+cpu（.venv 內，CPU-only）           |
| 統計        | scipy                                            |
| 爬蟲        | requests, beautifulsoup4                         |
| 執行腳本    | python scripts/xxx.py（bash with PYTHONIOENCODING=utf-8） |

---

## 三、目前完整目錄結構

```
superlotto_prediction/
├── config/
│   └── settings.py              # 全域設定：DB路徑、號碼範圍、LOG設定
├── data/
│   ├── superlotto.db            # SQLite DB（git 忽略）
│   └── exports/                 # CSV 匯出輸出
├── docs/
│   ├── evaluation_pipeline.md   # 評估架構說明（含 EV、per-draw 方法）
│   └── backend_theory.md        # 所有模型數學原理（12 個模型）
├── results/                     # ← 評估結果存放（2026-03-21 新增）
│   ├── rolling_eval_summary.csv              # 12模型×25指標，每次執行自動追加
│   ├── rolling_150draws_classic_models_2026-03-21.txt  # 10個模型完整輸出
│   ├── feature_analysis.csv                  # 18個特徵的 PB相關係數排名
│   └── evaluation_report_2026-03-21.md       # 完整評估報告（特徵+模型+策略）
├── src/
│   ├── scraper/
│   │   ├── client.py            # HTTP Session + 指數退避重試
│   │   ├── parser.py            # JSON/HTML 解析 + 資料驗證
│   │   └── runner.py            # 全量/增量抓取流程
│   ├── database/
│   │   ├── schema.sql           # 6 張資料表 DDL
│   │   ├── connection.py        # SQLite context manager
│   │   └── repository.py        # CRUD / UPSERT 封裝
│   ├── features/
│   │   ├── base_stats.py        # 頻率統計、Gap 間隔分析
│   │   ├── pattern_stats.py     # 奇偶/高低/連號/總和特徵
│   │   └── feature_builder.py   # 整合特徵 → ML 訓練矩陣
│   ├── models/
│   │   ├── baseline.py          # FrequencyBaseline（頻率排名）
│   │   ├── monte_carlo.py       # MonteCarloPredictor（10萬次模擬）
│   │   ├── markov.py            # MarkovPredictor（一階轉移機率）
│   │   ├── bayesian.py          # BayesianPredictor（Gap先驗+Beta後驗）
│   │   ├── knn_model.py         # KNNPredictor（特徵相似度）
│   │   ├── decision_tree.py     # DecisionTreePredictor（多輸出DT）
│   │   ├── classifier.py        # RF + XGBoost（MultiOutputClassifier）
│   │   ├── genetic.py           # GeneticPredictor（GA 演化搜尋）
│   │   ├── neural_network.py    # MLPPredictor + LSTMPredictor（PyTorch）
│   │   ├── ensemble.py          # EnsemblePredictor（加權投票）
│   │   ├── trainer.py           # 訓練流程 + 時序切分
│   │   └── predictor.py         # 輸出格式化
│   └── evaluation/
│       ├── walk_forward.py      # WalkForwardSplit（日期切分器）
│       └── metrics.py           # FoldMetrics + PRIZE_TABLE + EV 計算
├── scripts/
│   ├── init_db.py               # 初始化資料庫（建 6 張表）
│   ├── scrape_all.py            # 全量歷史抓取
│   ├── scrape_history.py        # lotterycorner.com 歷史補充
│   ├── scrape_latest.py         # 增量更新（每週執行）
│   ├── build_features.py        # 重建特徵表
│   ├── run_prediction.py        # 輸出所有模型當期預測（含對比表）
│   ├── run_evaluation.py        # Walk-Forward 評估（per-draw 滾動）
│   ├── run_rolling_nn.py        # 逐期 Rolling 評估（含所有模型+multi-ticket）
│   └── feature_analysis.py     # ← 新增：特徵信號 PB 相關係數分析
├── tests/
│   ├── test_parser.py
│   ├── test_repository.py
│   └── test_features.py
├── notebooks/                   # EDA 分析 Notebooks
├── requirements.txt
├── README.md
├── PROMPT.md                    # ← 本文件
└── .venv/                       # Python 3.12 虛擬環境（git 忽略）
```

---

## 四、已實作的 12 個模型

| # | 模型 | 檔案 | 特徵輸入 | 關鍵特色 |
|---|------|------|----------|----------|
| 1 | FrequencyBaseline | baseline.py | 全史+近期頻率 | 頻率加權排名，所有模型的基準線 |
| 2 | MonteCarloPredictor | monte_carlo.py | 全史頻率（抽樣權重） | 10萬次模擬，捕捉號碼共現 |
| 3 | MarkovPredictor | markov.py | 轉移矩陣 47×47 | 一階轉移機率矩陣（號碼→號碼）|
| 4 | BayesianPredictor | bayesian.py | Gap（距上次出現期數） | Gap 作為似然，Beta 先驗 |
| 5 | KNNPredictor | knn_model.py | current_gap(47)+freq_30(47)=94維 | 特徵相似度 k 近鄰 |
| 6 | DecisionTreePredictor | decision_tree.py | current_gap(47)+freq_30(47)=94維 | MultiOutputClassifier + DT |
| 7 | RF + XGBoost | classifier.py | gap+freq+pattern特徵 94維+ | MultiOutputClassifier，47顆球各一 |
| 8 | GeneticPredictor | genetic.py | 全史頻率+共現矩陣 | 適應度=頻率+共現，300代演化 |
| 9 | MLPPredictor | neural_network.py | current_gap(47)+freq_30(47)=94維 | PyTorch MLP，94→128→64→47，partial_fit |
| 10 | LSTMPredictor | neural_network.py | 94維 × seq_len=20 | PyTorch LSTM，sequence modeling，gradient clip |
| 11 | EnsemblePredictor | ensemble.py | 7個子模型投票 | 加權投票（frequency/MC/markov/bayesian/KNN/DT/genetic）|

所有模型介面統一：`model.fit(df)` + `model.predict(n_white=5)` → `{"white_balls": [...], "mega_balls": [...], "proba": {...}}`

> 注意：所有模型的 `predict()` 現在均回傳 `"proba"` 欄位（各球出現機率字典），供 multi-ticket 策略使用。

---

## 五、現行特徵架構

### 已使用的特徵

| 特徵 | 來源 | 維度 | 使用模型 |
|------|------|------|----------|
| `current_gap[n]` | base_stats.py | 47 | KNN, DT, RF, XGB, MLP, LSTM |
| `freq_window[n]` (30期) | base_stats.py | 47 | KNN, DT, RF, XGB, MLP, LSTM |
| `白球總和` | pattern_stats.py | 1 | RF, XGB |
| `奇偶/高低比` | pattern_stats.py | 4 | RF, XGB |
| `連號對數` | pattern_stats.py | 1 | RF, XGB |
| `號碼跨度` | pattern_stats.py | 1 | RF, XGB |
| `全史頻率` | base_stats.py | 47 | Frequency, MC, Bayesian, Genetic |
| `共現矩陣 47×47` | genetic.py 內建 | 2209 | Genetic, Ensemble(via genetic) |
| `轉移矩陣 47×47` | markov.py 內建 | 2209 | Markov |

### 特徵信號分析結論（2026-03-21，PB相關係數）

| 特徵 | \|r\| | p 值 | 結論 |
|------|-------|------|------|
| `ball_zone`（號碼區間） | 0.0198 | \*\*\* | 唯一顯著，但效果量極小 |
| `current_gap` | 0.0062 | 0.300 | 不顯著 |
| `freq_10/30/100` | <0.010 | >0.10 | 不顯著 |
| `gap_ratio`（超期比） | 0.0033 | 0.580 | 不顯著 |
| `appeared_last` | 0.0052 | 0.383 | 不顯著 |
| `last_sum/odd/consec` | 0.0000 | 1.000 | 完全無信號 |

**核心結論**：彩票為真實獨立隨機事件，所有候選特徵 \|r\| < 0.02，特徵工程的改善空間極為有限。

---

## 六、評估管線設計

### 評估腳本對比

| 腳本 | 切分方式 | 模型更新頻率 | 適用模型 |
|------|----------|-------------|---------|
| `run_evaluation.py` | Walk-Forward Fold（日期） | 每 Fold 重訓一次 | 快速模型 |
| `run_rolling_nn.py` | 全資料逐期滾動 | 依 MODEL_RETRAIN_DEFAULTS | 所有模型（含NN） |

### run_rolling_nn.py 關鍵設計

**MODEL_RETRAIN_DEFAULTS**（各模型重訓週期）：
```python
{"bayesian": 1, "markov": 1, "frequency": 1, "knn": 1, "decision_tree": 1,
 "rf": 3, "xgb": 3, "ensemble": 30,
 "montecarlo": 5, "genetic": 10, "mlp": 10, "lstm": 10}
```

**Multi-Ticket 策略**（`generate_top_k_tickets()`）：
- 每期生成最多 5 組候選組合
- 取機率最高的 12 顆球（n_candidates），枚舉 C(12,5)=792 種組合
- 依機率分數（sum of proba）排名，回傳 top-K
- 評估買1到5組的 EV / 淨EV / ROI

**自動存 Summary CSV**：每次跑完自動追加到 `results/rolling_eval_summary.csv`

### EV 計算

獎金表（實際 Pari-Mutuel）：5+Mega=$7M / 5=$25,241 / 4+Mega=$1,402 / 4=$108 / 3+Mega=$61 / 3=$11 / 2+Mega=$12 / 1+Mega=$2 / 0+Mega=$1

---

## 七、最新評估結果（2026-03-21，150期 Rolling）

### 命中率排名（隨機基準 Avg Hits = 0.532）

| 排名 | 模型 | Avg Hits | vs 隨機 | EV/期 | ROI% |
|------|------|----------|---------|-------|------|
| 1 | **XGB** | **0.667** | **+0.135** | $0.184 | -81.6% |
| 2 | RF | 0.627 | +0.095 | $0.174 | -82.6% |
| 3 | DecisionTree | 0.607 | +0.075 | $0.171 | -82.9% |
| 4 | Frequency | 0.587 | +0.055 | $0.140 | -86.0% |
| 5 | Genetic | 0.587 | +0.055 | $0.085 | -91.5% |
| 6 | KNN | 0.567 | +0.035 | $0.085 | -91.5% |
| 7 | Ensemble | 0.567 | +0.035 | $0.083 | -91.8% |
| 8 | MonteCarlo | 0.560 | +0.028 | $0.060 | -94.0% |
| 9 | Markov | 0.533 | +0.001 | $0.133 | -86.7% |
| 10 | MLP | 0.513 | -0.019 | $0.074 | -92.6% |
| 11 | LSTM | 0.500 | -0.032 | $0.158 | -84.2% |
| 12 | Bayesian | 0.480 | -0.052 | $0.073 | -92.7% |

### Multi-Ticket 最佳策略

| 最佳策略 | ROI |
|---------|-----|
| **Ensemble 買3組** | -80.6%（所有組合中最低損失）|
| XGB 買1組 | -81.6%（命中率最高） |
| LSTM 買5組 | -82.1% |

> **注意**：所有 ROI 均為負值，這是彩票設計的必然結果（賭場優勢）。

### 神經網路 vs 傳統模型

MLP（0.513）和 LSTM（0.500）均低於隨機基準（0.532），在 649 期資料量下出現過擬合。XGB/RF 的優勢可能部分來自小樣本統計噪音，建議以更長時間窗口驗證。

---

## 八、重要設計決策記錄

| 決策 | 原因 |
|------|------|
| Per-Draw 滾動預測 | 比「Fold 訓練一次」更正確，反映真實持續學習場景 |
| Expanding Window | 保留所有歷史資料（彩票號碼分布相對穩定）|
| `proba` 統一介面 | 所有模型 predict() 均回傳機率字典，供 multi-ticket 使用 |
| `pos_weight = 8.4` | 5顆中/42顆未中的類別不平衡補償（分類模型）|
| Pari-Mutuel 獎金表 | 使用實際某期開獎資料，更接近真實 EV |
| `PYTHONIOENCODING=utf-8` | Windows cp950 終端無法顯示中文 Unicode |
| PyTorch CPU-only | 本機無 CUDA GPU，安裝 torch+cpu 節省磁碟 |
| Gradient Clipping LSTM | 防止梯度爆炸，clip_grad_norm(max_norm=1.0) |
| ensemble retrain_every=30 | Ensemble 內含 MC+GA，每期重訓成本太高（原來3，改為30）|
| results/ 自動存 CSV | `save_summary_csv()` 追加模式，方便跨次執行比較 |

---

## 九、快速上手命令

```bash
cd d:/010_Github/superlotto_prediction

# 確認環境
PYTHONIOENCODING=utf-8 .venv/Scripts/python -c "import torch, pandas, scipy; print('OK')"

# 執行當期預測（所有模型）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_prediction.py

# Rolling 評估（所有模型，150期）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_rolling_nn.py \
  --test-draws 150 --initial-train 200 \
  --models frequency,bayesian,markov,montecarlo,knn,decision_tree,genetic,rf,xgb,ensemble \
  --n-tickets 5 --n-candidates 12

# Rolling 評估（MLP + LSTM）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_rolling_nn.py \
  --test-draws 150 --models mlp,lstm \
  --mlp-epochs 100 --lstm-epochs 80

# Rolling 評估（全部12個模型）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_rolling_nn.py \
  --test-draws 150 --models all

# 特徵信號分析
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/feature_analysis.py

# Walk-Forward Fold 評估（快速）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_evaluation.py \
  --train-months 18 --test-months 3 \
  --models frequency,markov,bayesian --show-folds

# 增量更新資料（每週執行）
.venv/Scripts/python scripts/scrape_latest.py
.venv/Scripts/python scripts/build_features.py
```

---

## 十、可能的下一步

### A. 擴大資料量（驗證現有結果是否只是噪音）
- 把評估資料從 2020 後（649 期）改為 2010 後（~1,600 期）
- 重跑 150 期 rolling，確認 XGB/RF 優勢是否持續

### B. 延長評估期數（更嚴格的統計檢定）
- 目前 test-draws=150，樣本量不足以區分模型間差異
- 改為 test-draws=400，讓 Avg Hits 的標準誤更小

### C. 超參數調優（XGB/RF 基礎上改善）
- XGB：`n_estimators`、`max_depth`、`learning_rate`
- RF：`n_estimators`、`min_samples_leaf`
- 工具：Optuna（貝氏超參數搜索）

### D. 特徵工程迭代（效果有限但值得一試）
- 加入 `gap_ratio`、`freq_5`、`freq_100` 到 KNN/DT
- 加入 `ball_zone` bias 修正（唯一顯著特徵）
- 目標：XGB/RF 從 0.667/0.627 能否再提升？

### E. 生成當期實際預測
- 用最佳模型（XGB、Ensemble）對下一期開獎做預測
- 輸出：推薦號碼 + 機率分布 + 多組候選

### F. 視覺化（Notebook 更新）
- ROI 隨期數的累積曲線
- 命中數分布直方圖
- 特徵重要性（XGB feature importance）

---

*最後更新：2026-03-21*
