"""
trainer.py — 模型訓練流程
==========================
負責：
  1. 從特徵矩陣中以「時間序列切分」方式劃分訓練/測試集
  2. 訓練 Random Forest 多標籤分類器
  3. 訓練 XGBoost 多標籤分類器
  4. 計算評估指標（AUC-ROC、Precision@K）
  5. 儲存訓練好的模型

關鍵設計原則：
  ⚠️  時間序列切分（Time Series Split）
      訓練集 = 前 train_ratio% 的資料（時間較早）
      測試集 = 後 (1-train_ratio)% 的資料（時間較晚）
      絕對不能用隨機 shuffle！否則「未來」的資料會進入訓練集（look-ahead bias）。
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score

from config.settings import (
    TRAIN_RATIO, RF_N_ESTIMATORS, RF_RANDOM_STATE,
    WHITE_BALL_MIN, WHITE_BALL_MAX, LOG_LEVEL, LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))
MODELS_DIR    = Path(__file__).parent.parent.parent / "data" / "models"


def time_series_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
) -> tuple:
    """
    時間序列資料切分（不 shuffle）。

    前 train_ratio 的資料用於訓練，剩餘用於測試。
    資料必須已按時間順序排列（feature_builder 已確保）。

    Args:
        X:           特徵矩陣 (n_samples, n_features)
        y:           標籤矩陣 (n_samples, 47)
        train_ratio: 訓練集比例（預設 0.80）

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    split_idx = int(n * train_ratio)

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(
        f"時序切分：訓練集 {len(X_train)} 筆，測試集 {len(X_test)} 筆"
        f"（{train_ratio*100:.0f}% / {(1-train_ratio)*100:.0f}%）"
    )
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> MultiOutputClassifier:
    """
    訓練 Random Forest 多標籤分類器。

    為何選 Random Forest：
      • 天然支援非線性關係（不需要特徵縮放）
      • 內建特徵重要性（可解釋性）
      • class_weight='balanced' 自動處理類別不平衡
        （每期只有 5/47 ≈ 10.6% 的球標籤為 1，剩下 89.4% 為 0）
      • n_jobs=-1 使用全部 CPU 核心並行訓練

    為何用 MultiOutputClassifier：
      47 個白球的出現與否是「47個獨立的二元分類問題」。
      MultiOutputClassifier 會為每個輸出維度訓練一個獨立的分類器，
      每個分類器都能學到「哪些特徵影響這個號碼的出現」。

    Args:
        X_train: 訓練特徵矩陣
        y_train: 訓練標籤矩陣（47維）

    Returns:
        訓練好的 MultiOutputClassifier
    """
    logger.info(f"訓練 Random Forest（{RF_N_ESTIMATORS} 棵決策樹，使用所有 CPU 核心）...")

    # Best params from Optuna HPO (30 trials, val Avg Hits=0.7833)
    base_clf = RandomForestClassifier(
        n_estimators     = 50,
        max_depth        = 6,
        min_samples_leaf = 1,
        max_features     = 0.5,
        class_weight     = "balanced",
        random_state     = RF_RANDOM_STATE,
        n_jobs           = 1,   # MultiOutput handles parallelism
    )

    model = MultiOutputClassifier(base_clf, n_jobs=-1)  # 47 classifiers in parallel
    model.fit(X_train, y_train)

    logger.info("Random Forest 訓練完成")
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> MultiOutputClassifier:
    """
    訓練 XGBoost 多標籤分類器（作為 Random Forest 的對比）。

    為何也訓練 XGBoost：
      • Gradient Boosting 對不平衡資料通常效果更好
      • scale_pos_weight 參數可精細控制正負樣本權重
      • 與 RF 比較，了解哪種模型更適合這個問題

    Args:
        X_train: 訓練特徵矩陣
        y_train: 訓練標籤矩陣（47維）

    Returns:
        訓練好的 MultiOutputClassifier（包含47個 XGBoost 分類器）
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        logger.error("未安裝 xgboost！請執行：pip install xgboost")
        raise

    # 正負樣本比例：平均每期有5顆球出現（正樣本），42顆未出現（負樣本）
    # scale_pos_weight = 負樣本數 / 正樣本數 ≈ 42/5 ≈ 8.4
    pos_weight = (WHITE_BALL_MAX - 5) / 5  # ≈ 8.4

    logger.info(f"訓練 XGBoost（scale_pos_weight={pos_weight:.1f}）...")

    # Best params from Optuna HPO (30 trials, val Avg Hits=0.7333)
    base_clf = XGBClassifier(
        n_estimators     = 100,
        max_depth        = 5,
        learning_rate    = 0.050,
        subsample        = 0.630,
        colsample_bytree = 0.773,
        min_child_weight = 5,
        scale_pos_weight = pos_weight,
        eval_metric      = "logloss",
        random_state     = RF_RANDOM_STATE,
        n_jobs           = 1,   # MultiOutput handles parallelism
    )

    model = MultiOutputClassifier(base_clf, n_jobs=-1)  # 47 classifiers in parallel
    model.fit(X_train, y_train)

    logger.info("XGBoost 訓練完成")
    return model


# ────────────────────────────────────────────────────────────────────────────
# 評估指標
# ────────────────────────────────────────────────────────────────────────────

def precision_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: int = 5) -> float:
    """
    計算 Precision@K：預測機率最高的 K 個號碼中，實際有幾個命中。

    這是彩票預測最直觀的評估指標：
      我推薦了 5 個號碼，實際上有幾個開出來了？

    為何不用 Accuracy：
      若全部預測為 0（未出現），Accuracy = 42/47 ≈ 89.4%，看起來很高但毫無意義。

    Args:
        y_true: 實際標籤矩陣 (n_samples, 47)
        y_prob: 預測機率矩陣 (n_samples, 47)
        k:      選取前 K 個（通常 = 5，即選 5 顆球）

    Returns:
        float: 平均 Precision@K（所有測試期的平均）
    """
    precisions = []
    for true_row, prob_row in zip(y_true, y_prob):
        # 取預測機率最高的 k 個號碼的索引
        top_k_indices = np.argsort(prob_row)[::-1][:k]
        # 計算這 k 個號碼中有幾個實際出現
        hits = sum(true_row[i] for i in top_k_indices)
        precisions.append(hits / k)

    return float(np.mean(precisions))


def evaluate_model(
    model: MultiOutputClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    評估模型效能，計算多個指標。

    隨機基準（理論值）：
      Precision@5 = 5/47 ≈ 10.6%（隨機選 5 顆球的期望命中率）
      AUC-ROC     = 0.5（隨機猜測）

    Args:
        model:  訓練好的 MultiOutputClassifier
        X_test: 測試特徵矩陣
        y_test: 測試標籤矩陣

    Returns:
        dict: 各評估指標
    """
    # 取得每個輸出的預測機率（predict_proba 對 MultiOutputClassifier 回傳 list of arrays）
    proba_list = model.predict_proba(X_test)

    # 整理為 (n_samples, 47) 的矩陣
    # 每個分類器的 proba shape = (n_samples, 2)，取索引 1（= 正類機率）
    y_prob = np.column_stack([proba[:, 1] for proba in proba_list])

    p_at_5 = precision_at_k(y_test, y_prob, k=5)

    # Macro AUC-ROC（對每個號碼分別計算，然後取平均）
    try:
        auc = roc_auc_score(y_test, y_prob, average="macro")
    except ValueError as e:
        # 若測試集中某個號碼完全沒出現，AUC 無法計算
        logger.warning(f"AUC-ROC 計算失敗：{e}，改用 samples 平均")
        auc = roc_auc_score(y_test, y_prob, average="samples")

    results = {
        "precision_at_5":    p_at_5,
        "precision_at_5_pct": f"{p_at_5 * 100:.2f}%",
        "auc_roc_macro":     auc,
        "random_baseline_p5": 5 / 47,
        "improvement_over_random": p_at_5 - (5 / 47),
    }

    logger.info(f"📊 評估結果：")
    logger.info(f"   Precision@5 = {p_at_5*100:.2f}%（隨機基準 = {5/47*100:.2f}%）")
    logger.info(f"   AUC-ROC     = {auc:.4f}（隨機基準 = 0.5000）")

    return results


# ────────────────────────────────────────────────────────────────────────────
# 模型存檔與讀取
# ────────────────────────────────────────────────────────────────────────────

def save_model(model, filename: str) -> Path:
    """
    將訓練好的模型序列化存檔（pickle 格式）。
    存放路徑：data/models/{filename}

    Returns:
        Path: 存檔路徑
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"模型已存檔：{path}")
    return path


def load_model(filename: str):
    """
    讀取已存檔的模型。

    Args:
        filename: 模型檔名（相對於 data/models/）

    Returns:
        反序列化後的模型物件
    """
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到模型檔案：{path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"模型已載入：{path}")
    return model
