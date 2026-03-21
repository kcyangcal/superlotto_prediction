# SuperLotto Plus ML Pipeline — 新 Chat 接續指引

> 本文件提供足夠的脈絡，讓新的 chat session 能直接接續專案工作。

---

## 一、專案概覽

**目的**：純學習用的 ML 管線，練習資料工程、爬蟲、特徵工程、ML 建模與評估。
**資料**：加州 SuperLotto Plus 歷史開獎（2000–2025，約 2,691 期）
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
│   └── run_rolling_nn.py        # 逐期 Rolling 評估（含 MLP/LSTM）
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

| # | 模型              | 檔案                   | 關鍵特色                              |
|---|-------------------|------------------------|---------------------------------------|
| 1 | FrequencyBaseline | baseline.py            | 頻率加權排名，所有模型的基準線        |
| 2 | MonteCarloPredictor | monte_carlo.py       | 10萬次模擬，捕捉號碼共現              |
| 3 | MarkovPredictor   | markov.py              | 一階轉移機率矩陣（號碼→號碼）         |
| 4 | BayesianPredictor | bayesian.py            | Gap 作為似然，Beta 先驗               |
| 5 | KNNPredictor      | knn_model.py           | 94維特徵相似度 k 近鄰                 |
| 6 | DecisionTreePredictor | decision_tree.py   | MultiOutputClassifier + DT            |
| 7 | RF + XGBoost      | classifier.py          | 多輸出分類，處理類別不平衡            |
| 8 | GeneticPredictor  | genetic.py             | 適應度=頻率+共現，300代演化           |
| 9 | MLPPredictor      | neural_network.py      | PyTorch MLP，94→128→64→47，partial_fit|
|10 | LSTMPredictor     | neural_network.py      | PyTorch LSTM，seq_len=20，gradient clip|
|11 | EnsemblePredictor | ensemble.py            | 加權投票（7個子模型）                 |

所有模型介面統一：`model.fit(df)` + `model.predict(n_white=5)` → `{"white_balls": [...], "mega_balls": [...]}`

---

## 五、評估管線設計

### 評估腳本對比

| 腳本                  | 切分方式              | 模型更新頻率     | 適用模型                |
|-----------------------|-----------------------|------------------|-------------------------|
| `run_evaluation.py`   | Walk-Forward Fold     | **每期重訓**     | 快速模型（<100ms/期）   |
| `run_rolling_nn.py`   | 全資料逐期滾動        | partial_fit/重訓 | 含 MLP/LSTM             |

### 關鍵設計：Per-Draw 滾動（不是 Fold 一次訓練）

每個測試期：先用截至上期的全部歷史訓練 → 預測 → 記錄命中 → 把本期加入訓練集 → 下一期。

### EV 計算（獎金期望值）

獎金表（實際某期 Pari-Mutuel 資料）：

| 命中         | 獎金        |
|-------------|-------------|
| 5 + Mega    | $7,000,000  |
| 5           | $25,241     |
| 4 + Mega    | $1,402      |
| 4           | $108        |
| 3 + Mega    | $61         |
| 3           | $11         |
| 2 + Mega    | $12         |
| 1 + Mega    | $2          |
| 0 + Mega    | $1          |

EV 欄位：`ev_per_draw`、`net_ev_per_draw`（= EV - $1）、`roi_pct`

---

## 六、目前狀況（截至 2026-03-20）

### ✅ 已完成

- [x] 資料爬蟲：CA Lottery API + lotterycorner.com 補充（2,691 期）
- [x] SQLite 資料庫：6 張表，Raw/Derived 分離
- [x] 特徵工程：Gap、頻率（短/中/長期）、模式特徵
- [x] 12 個模型全部實作，介面統一
- [x] Walk-Forward 評估管線（`run_evaluation.py`）
  - Per-Draw 滾動預測（每期重訓，不是 Fold 一次訓練）
  - EV 計算（含 Pari-Mutuel 獎金表）
  - FoldMetrics、t-test 顯著性、跨模型比較表
- [x] Rolling NN 評估管線（`run_rolling_nn.py`）
  - 支援 MLP/LSTM partial_fit
  - EV 追蹤
  - Concept Drift 移動平均分析
- [x] 兩份技術文件（`docs/evaluation_pipeline.md`、`docs/backend_theory.md`）
- [x] Windows cp950 編碼問題修正（用 `PYTHONIOENCODING=utf-8` 執行）
- [x] PyTorch 2.10.0+cpu 安裝於 .venv (Python 3.12)

### 最新評估結果（2026-03-20，train=18m, test=3m，18 Folds）

| 模型      | Avg Hits | Prec@5  | EV/期   | 淨EV    | p-val |
|-----------|----------|---------|---------|---------|-------|
| markov    | 0.583    | 11.66%  | $0.151  | $-0.849 | 0.113 |
| frequency | 0.570    | 11.41%  | $0.352  | $-0.648 | 0.225 |
| bayesian  | 0.491    | 9.82%   | $0.104  | $-0.896 | 0.162 |
| 隨機基準  | 0.532    | 10.64%  | —       | —       |       |

**結論**：所有模型 p > 0.05，未能顯著超越隨機基準，符合彩票設計的理論預期。

---

## 七、待完成項目

### 優先級高
- [ ] **run_rolling_nn.py 實際執行** — MLP/LSTM 的 rolling 結果尚未跑過，需確認 EV 計算正確
- [ ] **完整 run_evaluation.py 執行**（含 montecarlo） — 目前只測了 frequency/markov/bayesian，montecarlo 較慢尚未納入
- [ ] **git commit** — 目前所有新增/修改的檔案尚未提交

### 優先級中
- [ ] **Notebook 更新**：`01_data_exploration.ipynb` 和 `02_feature_analysis.ipynb` 尚未補充 EV 視覺化和新模型比較
- [ ] **README.md 更新**：目前 README 的模型列表和架構圖未反映 12 個模型與評估管線
- [ ] **RF/XGBoost walk-forward 測試** — 這兩個模型在 walk-forward 中過慢，可考慮每 Fold 訓練一次（非 per-draw）

### 優先級低
- [ ] **單元測試補充**：`tests/` 目前只有 `test_parser.py`、`test_repository.py`、`test_features.py`，新模型和評估模組缺乏測試
- [ ] **超參數調優**：MLP 的 epoch、hidden size；LSTM 的 seq_len、hidden；可用 Optuna 做 HPO
- [ ] **Sliding Window 對比**：目前只用 Expanding Window，可加 Sliding Window 比較 Concept Drift 效果
- [ ] **Jackpot EV 模擬**：頭獎為 Pari-Mutuel 滾動累計，實際金額每期不同；可爬取歷史 jackpot 金額做更精確的 EV 回測

---

## 八、快速上手命令

```bash
# 確認環境
cd d:/010_Github/superlotto_prediction
.venv/Scripts/python -c "import torch, pandas, scipy; print('OK')"

# 執行當期預測（所有模型）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_prediction.py

# Walk-Forward 評估（快速）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_evaluation.py \
  --train-months 18 --test-months 3 \
  --models frequency,markov,bayesian --show-folds

# Walk-Forward 評估（完整含 MonteCarlo）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_evaluation.py \
  --models frequency,montecarlo,markov,bayesian

# Rolling NN 評估（快速，跳過 LSTM）
PYTHONIOENCODING=utf-8 .venv/Scripts/python scripts/run_rolling_nn.py \
  --test-draws 100 --no-lstm --models mlp,bayesian,markov,frequency

# 增量更新資料（每週執行）
.venv/Scripts/python scripts/scrape_latest.py
.venv/Scripts/python scripts/build_features.py
```

---

## 九、重要設計決策記錄

| 決策                       | 原因                                                            |
|----------------------------|-----------------------------------------------------------------|
| Per-Draw 滾動預測           | 比「Fold 訓練一次」更正確，反映真實持續學習場景                  |
| Expanding Window            | 保留所有歷史資料（彩票號碼分布相對穩定）                        |
| pos_weight = 8.4            | 5顆中/42顆未中的類別不平衡補償（所有分類模型）                   |
| Pari-Mutuel 獎金表          | 使用實際某期開獎資料而非固定估算值，更接近真實 EV               |
| PYTHONIOENCODING=utf-8      | Windows cp950 終端無法顯示中文 Unicode，需明確設定輸出編碼      |
| PyTorch CPU-only in .venv   | 本機無 CUDA GPU，安裝 torch+cpu 版以節省磁碟空間                |
| Gradient Clipping LSTM      | LSTM 序列訓練容易梯度爆炸，clip_grad_norm(max_norm=1.0) 防止    |
| lr×0.1 for LSTM fine-tune   | partial_fit 時降低學習率以緩解 Catastrophic Forgetting          |

---

*最後更新：2026-03-20*
