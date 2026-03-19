# SuperLotto Plus Prediction Pipeline

> ⚠️ **重要聲明**：本專案純粹作為**資料工程與機器學習的練習專案**。加州 SuperLotto Plus 開獎為完全隨機的獨立事件，任何統計模型均無法有效預測彩票結果。本程式輸出不構成任何購票建議。

This is a superlotto prediction modeling repo for learning purpose. There is no clear way to predict random draw in theory but it is fun to do it for scraping and modelling practice.

---

## 專案目標

透過分析加州 SuperLotto Plus 歷史開獎資料，練習以下技術：

| 技術領域 | 練習內容 |
|---------|---------|
| 資料工程 | SQLite Schema 設計、UPSERT 策略、Raw/Derived 分層 |
| 爬蟲工程 | requests Session、指數退避重試、JSON API 解析 |
| 特徵工程 | 頻率統計、Gap 分析、滾動視窗特徵 |
| 機器學習 | 多標籤分類、時序切分、類別不平衡處理 |

---

## 環境設置

### 前置需求
- Python 3.10 以上
- Git

### 1. Clone 專案
```bash
git clone https://github.com/your-username/superlotto_prediction.git
cd superlotto_prediction
```

### 2. 建立虛擬環境

**Windows PowerShell：**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> 若遇到執行政策錯誤：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**macOS / Linux：**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. 安裝套件
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 驗證安裝
```bash
python -c "import requests, pandas, sklearn; print('OK')"
```

---

## 使用方式

按照以下順序執行（首次設置）：

### Step 1：初始化資料庫
```bash
python scripts/init_db.py
```
建立 `data/superlotto.db` 和 6 張資料表。

### Step 2：全量抓取歷史資料
```bash
python scripts/scrape_all.py
```
從 CA Lottery 官方 API 抓取約 2,691 筆歷史開獎資料（約 40 秒）。

### Step 3：計算特徵
```bash
python scripts/build_features.py
```
將原始開獎號碼轉換為機器學習特徵。

### Step 4：訓練模型並輸出預測
```bash
python scripts/run_prediction.py
```

### 日常增量更新（每週三/六開獎後）
```bash
python scripts/scrape_latest.py
python scripts/build_features.py
```

---

## 專案結構

```
superlotto_prediction/
├── config/
│   └── settings.py              # 全域設定（API URL、DB路徑、模型參數）
├── data/
│   ├── superlotto.db            # SQLite 資料庫（git 忽略）
│   └── exports/                 # CSV 匯出（git 忽略）
├── src/
│   ├── scraper/
│   │   ├── client.py            # HTTP Session + 指數退避重試
│   │   ├── parser.py            # JSON 解析 + 資料驗證
│   │   └── runner.py            # 全量/增量抓取流程
│   ├── database/
│   │   ├── schema.sql           # 6 張資料表的 DDL
│   │   ├── connection.py        # SQLite context manager
│   │   └── repository.py       # CRUD / UPSERT 操作封裝
│   ├── features/
│   │   ├── base_stats.py        # 頻率統計、Gap 間隔分析
│   │   ├── pattern_stats.py     # 奇偶/高低/連號/總和特徵
│   │   └── feature_builder.py  # ML 訓練矩陣建構
│   └── models/
│       ├── baseline.py          # L0 頻率基準模型
│       ├── classifier.py        # L1 RF + L2 XGBoost 整合
│       ├── trainer.py           # 訓練/評估/存檔邏輯
│       └── predictor.py         # 輸出格式化
├── notebooks/                   # EDA 與分析 Jupyter Notebooks
├── tests/                       # 單元測試
├── scripts/                     # 一鍵執行腳本
└── requirements.txt
```

---

## 資料庫設計

| 資料表 | 類型 | 說明 |
|--------|------|------|
| `draws` | Raw | 每期開獎原始資料，含擴充欄位（機台/溫濕度） |
| `draw_prizes` | Raw | 每期獎項明細（1對多） |
| `white_ball_stats` | Derived | 白球號碼頻率彙總（可重算） |
| `mega_ball_stats` | Derived | Mega 球頻率彙總（可重算） |
| `draw_features` | Derived | 每期衍生特徵（可重算） |
| `number_gap_history` | Derived | 號碼間隔序列（可重算） |

---

## 機器學習模型

| 層次 | 模型 | 說明 |
|------|------|------|
| L0 | 頻率基準 | 選歷史出現最頻繁的 5 顆球（無監督基準） |
| L1 | Random Forest | `MultiOutputClassifier`，47個獨立二元分類，`class_weight='balanced'` |
| L2 | XGBoost | 同上，`scale_pos_weight=8.4` 處理類別不平衡 |

**評估指標**：Precision@5（預測 5 顆球中幾顆命中）+ AUC-ROC

**隨機基準**：Precision@5 = 5/47 ≈ 10.6%

---

## 執行測試

```bash
pytest tests/ -v
```

---

## License

MIT License — Copyright (c) 2026 KC
