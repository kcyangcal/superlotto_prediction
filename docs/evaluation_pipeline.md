# Evaluation Pipeline — 模型驗證架構說明

> **專案**：SuperLotto Plus ML 預測管線
> **目的**：建立嚴謹的時間序列模型評估機制，確保評估結果能反映真實預測能力

---

## 1. 為什麼評估架構比模型本身更重要？

在任何 ML 專案中，「如何評估」決定了「你知道什麼是真的」。

對彩票這類資料，一個常見的錯誤是：

```python
# ❌ 錯誤做法——隨機切分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

這樣做的後果：**訓練集包含「來自未來的資料」**。
例如 2023 年的資料進入訓練集，但 2021 年的資料在測試集。
模型在訓練時已「看過」它要預測的未來，評估結果因此虛高。
這個問題在業界稱為 **Data Leakage（資料洩漏）**，是時間序列 ML 最常見的致命錯誤。

---

## 2. Walk-Forward Validation 原理

### 2.1 靜態時序切分 vs. 滾動驗證

**靜態切分（Static Split）**：

```
▸ 訓練集 ──────────────────────────▸ ┃ ◂─ 測試集 ──▸
  2020-01                  2024-06  ┃   2024-07  2025-12
                                    ┃
                                    ▲ 這條線永遠固定
```

缺點：只評估了「一個時間點」的泛化能力，可能是偶然的好或偶然的壞。

**Walk-Forward Validation（滾動驗證）**：

```
Fold 1: [── Train ────────────────][─ Test ─]
Fold 2: [── Train ────────────────────────][─ Test ─]
Fold 3: [── Train ─────────────────────────────][─ Test ─]
Fold 4: [── Train ──────────────────────────────────][─ Test ─]
         2020-01                                        2025-12
```

優點：
- 測試集永遠是「訓練集看不見的未來資料」
- 可觀察模型準確率隨時間的波動（穩定性分析）
- 等效於「模擬真實使用場景：定期重訓、定期評估」

### 2.2 擴展視窗 vs. 滑動視窗

本專案採用**擴展視窗（Expanding Window）**：

| 特性           | 擴展視窗（本專案）               | 滑動視窗                        |
|----------------|----------------------------------|---------------------------------|
| 訓練集起點     | 固定（2020-01）                  | 隨 Fold 向後移動                |
| 訓練集長度     | 逐 Fold 增長                     | 固定長度                        |
| 資料利用效率   | 高（保留所有歷史）               | 低（丟棄舊資料）                |
| 適用場景       | 長期趨勢重要的問題               | 最近趨勢比長期趨勢更重要        |
| 彩票適用性     | **較適合**（號碼分布相對穩定）   | 若懷疑有 Concept Drift 可嘗試   |

### 2.3 本專案的切分參數

```
初始訓練視窗：24 個月（約 208 期）
測試視窗大小：6  個月（約  52 期）
滾動步長    ：6  個月（非重疊測試集）
起始日期    ：2020-01-01
```

以 2020–2026 資料計算，大約可產生 **8–10 個 Fold**。

---

## 3. 評估指標說明

### 3.1 主要指標：Avg Hits@5（平均命中數）

**定義**：模型推薦的 5 顆白球中，實際開出幾顆的期望值。

```
每期命中數 = len(set(推薦5顆) ∩ set(實際5顆))
Avg Hits   = mean(所有測試期的每期命中數)
```

**隨機基準（理論值）**：

依超幾何分布（Hypergeometric Distribution）：

```
E[命中數] = k × K/N  = 5 × 5/47 ≈ 0.532 顆
```

其中：N=47（總球數）、K=5（實際開出）、k=5（我們推薦的數量）

**解讀**：
- `Avg Hits ≈ 0.532`：與隨機相當，模型沒有任何預測能力
- `Avg Hits > 0.532`：優於隨機（但需統計顯著性檢定）
- `Avg Hits < 0.532`：差於隨機（模型有反效果）

### 3.2 Precision@5

```
Precision@5 = Avg Hits / 5
隨機基準   = 5/47 ≈ 10.64%
```

### 3.3 Coverage（覆蓋率）

```
Coverage = P(命中數 ≥ 1)
         = 測試期中「至少猜對 1 顆」的比例
```

### 3.4 統計顯著性：單尾 t-test

為了判斷「優於隨機」是真實效果還是統計噪音：

```
H₀: 模型的平均命中數 = 隨機基準（0.532）
H₁: 模型的平均命中數 > 0.532
```

若 **p > 0.05**，我們不能拒絕虛無假設——差異很可能只是隨機波動。

> **重要提醒**：即使某模型的 p < 0.05，這只代表「在歷史資料上統計顯著高於隨機」，
> **不代表對未來開獎有任何預測能力**。彩票是獨立隨機事件，理論上任何方法都無法提高中獎率。

---

## 4. Per-Draw 滾動預測（正確評估方法）

### 4.1 為什麼「每 Fold 訓練一次」是錯的

舊版 `run_evaluation.py` 的問題：

```python
# ❌ 舊版錯誤做法
model.fit(train_df)
prediction = model.predict()
# 用同一組 5 球對該 Fold 所有 52 期做比較
for row in test_df.iterrows():
    hits = hit_at_k(prediction, row)   # 模型從未更新！
```

問題：測試視窗內的第 2 期、第 3 期… 已知的開獎結果沒有被納入訓練。
這相當於假設「模型在測試期間永遠不學習新資料」，不符合真實使用場景。

### 4.2 正確的 Per-Draw 滾動評估

```python
# ✅ 正確做法：每期預測前，用截至上一期的完整歷史訓練
cumulative = train_df.copy()

for draw_i, (_, row) in enumerate(test_df.iterrows()):
    actual_balls = [int(row[c]) for c in WHITE_COLS]

    # ① 以截至上一期的全部歷史訓練（不含本期）
    model = build_model(model_name)
    model.fit(cumulative)

    # ② 預測（此時模型看不到本期開獎）
    prediction = model.predict(n_white=5)
    predicted_balls = prediction.get("white_balls", [])

    # ③ 記錄命中數與獎金 EV（此時才「揭曉」本期開獎）
    fold_metric.add(predicted_balls, actual_balls, predicted_mega, actual_mega)

    # ④ 把本期實際開獎加入累積訓練資料，供下一期使用
    cumulative = pd.concat([cumulative, test_df.iloc[[draw_i]]], ignore_index=True)
```

**速度**：Bayesian/Markov/Frequency 每次重訓 1–10ms，52 期 × 8 Fold 合計 < 10 秒。

---

## 5. 獎金期望值（EV）計算

### 5.1 SuperLotto Plus 獎金結構

以下為實際某期開獎的 Pari-Mutuel 獎金（每期依銷售量和中獎人數浮動）：

| 命中組合     | 獎金          |
|-------------|---------------|
| 5 + Mega    | $7,000,000    |
| 5           | $25,241       |
| 4 + Mega    | $1,402        |
| 4           | $108          |
| 3 + Mega    | $61           |
| 3           | $11           |
| 2 + Mega    | $12           |
| 1 + Mega    | $2            |
| 0 + Mega    | $1            |

票價：$1.00

### 5.2 EV 計算邏輯

```python
TICKET_COST = 1.0

# 當模型有預測 Mega 球時：
mega_hit = (predicted_mega == actual_mega)
prize = PRIZE_TABLE[(white_hits, mega_hit)]

# 當模型未預測 Mega 球時（多數模型）：
# 假設隨機猜 1/27 機率命中 Mega，計算期望獎金
prize = (1/27) * PRIZE_TABLE[(hits, True)] + (26/27) * PRIZE_TABLE[(hits, False)]

# 每期淨 EV
net_ev = prize - TICKET_COST
```

### 5.3 實際評估結果（2020-2025，per-draw 滾動）

```
模型           │ Avg Hits │ Prec@5  │ vs Rnd  │  EV/期  │   淨EV  │  ROI%
───────────────┼──────────┼─────────┼─────────┼─────────┼─────────┼────────
markov         │   0.583  │ 11.66%  │ +0.051  │ $0.1508 │ $-0.849 │ -84.9%
frequency      │   0.570  │ 11.41%  │ +0.039  │ $0.3521 │ $-0.648 │ -64.8%
bayesian       │   0.491  │  9.82%  │ -0.041  │ $0.1037 │ $-0.896 │ -89.6%
隨機基準       │   0.532  │ 10.64%  │  0.000  │    —    │    —    │    —
```

**結論**：
- 所有模型的 p-value > 0.05（未顯著優於隨機基準）
- 淨 EV 全為負值（每注平均虧損 $0.65 ~ $0.90），這是彩票內建的莊家優勢
- `frequency` 的 EV 相對較高是因為特定 Fold 碰到 3 中（$11）的情況

---

## 6. 評估模組架構

```
src/evaluation/
├── __init__.py
├── walk_forward.py     # WalkForwardSplit 類別：產生 Fold 的訓練/測試資料對
└── metrics.py          # FoldMetrics, PRIZE_TABLE, calc_prize, EV 計算

scripts/
├── run_evaluation.py   # Walk-Forward 評估（per-draw 滾動，適合快速模型）
└── run_rolling_nn.py   # 逐期滾動評估（含 MLP/LSTM，適合神經網路）
```

### 核心類別

**`WalkForwardSplit`**（`walk_forward.py`）

| 方法                     | 說明                                          |
|--------------------------|-----------------------------------------------|
| `split(df)`              | Generator，逐一產生 `(train_df, test_df, fold_info)` |
| `get_fold_summary(df)`   | 只顯示切分摘要，不訓練，用於預覽                |

**`FoldMetrics`**（`metrics.py`）

| 方法                                                       | 說明                        |
|------------------------------------------------------------|-----------------------------|
| `add(predicted, actual, predicted_mega, actual_mega, k=5)` | 登記一期，自動計算獎金 EV   |
| `summary()`                                                | 回傳 Fold 彙整：命中/EV/ROI |

---

## 7. 執行方式

```bash
# 快速測試（3 個快速模型，3 個月測試視窗，含 EV 輸出）
PYTHONIOENCODING=utf-8 python scripts/run_evaluation.py \
  --train-months 18 \
  --test-months 3 \
  --models frequency,markov,bayesian \
  --show-folds

# 完整評估（含 Fold 詳細命中分布）
PYTHONIOENCODING=utf-8 python scripts/run_evaluation.py \
  --train-months 24 \
  --test-months 6 \
  --models frequency,montecarlo,markov,bayesian

# 神經網路 + 全模型逐期 Rolling 評估
PYTHONIOENCODING=utf-8 python scripts/run_rolling_nn.py \
  --test-draws 200 \
  --models mlp,lstm,bayesian,markov,frequency
```

> **Windows 注意**：加上 `PYTHONIOENCODING=utf-8` 避免中文字符在 cp950 終端顯示亂碼。

### 輸出範例（2026-03-20 實際執行結果）

```
══════════════════════════════════════════════════════════════
 Walk-Forward 切分預覽（train=18m, test=3m, 共 18 Folds）
══════════════════════════════════════════════════════════════
 fold  train_start  train_end   test_start  test_end    n_train  n_test
    1   2020-01-01  2021-06-30  2021-07-03  2021-09-29     157      26
    ...
   18   2020-01-01  2025-09-27  2025-10-01  2025-12-31     600      27

══════════════════════════════════════════════════════════════════════════════════════
 各模型 Walk-Forward 評估結果總覽
══════════════════════════════════════════════════════════════════════════════════════
模型           │ Avg Hits │ Prec@5  │ vs Rnd  │  EV/期   │   淨EV   │  ROI%  │ p-val
───────────────────────────────────────────────────────────────────────────────────
markov         │   0.583  │ 11.66%  │ +0.051  │ $0.1508  │ $-0.8492 │ -84.9% │ 0.1132
frequency      │   0.570  │ 11.41%  │ +0.039  │ $0.3521  │ $-0.6479 │ -64.8% │ 0.2252
bayesian       │   0.491  │  9.82%  │ -0.041  │ $0.1037  │ $-0.8963 │ -89.6% │ 0.1618
隨機基準       │   0.532  │ 10.64%  │  0.000  │    —     │    —     │   —    │
```

---

## 6. Concept Drift 分析

如果你觀察到：後期 Fold 的 `Avg Hits` 顯著低於前期 Fold，這可能代表：

1. **Concept Drift（概念漂移）**：號碼分布的統計規律在時間上發生了變化
2. **過擬合歷史統計**：模型過度依賴早期資料的規律，無法適應新模式

應對方式：
- 嘗試**滑動視窗**（只用最近 N 期訓練）而非擴展視窗
- 縮短 `train_months` 或 `test_months` 以獲得更細緻的分析
- 分析不同時期（2020–2022 vs 2023–2026）的號碼頻率是否有顯著差異

---

## 7. 重要學習收穫

| 學習概念                    | 在本專案的體現                                        |
|-----------------------------|-------------------------------------------------------|
| Data Leakage                | WalkForwardSplit 確保訓練集嚴格早於測試集              |
| 類別不平衡                  | Precision@k 而非 Accuracy 作為主指標                  |
| 隨機基準的重要性            | 所有指標都與理論隨機值比較                            |
| 統計顯著性                  | t-test 避免把「幸運噪音」誤認為「模型能力」            |
| Concept Drift               | 逐 Fold 顯示指標，觀察時序穩定性                      |
| Overfitting vs Underfitting | 若訓練集表現遠優於測試集，說明過擬合                  |

---

*本文件由 SuperLotto Plus ML Pipeline 專案自動生成，最後更新：2026-03-20*
