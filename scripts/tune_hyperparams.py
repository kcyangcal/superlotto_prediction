"""
tune_hyperparams.py — XGBoost + Random Forest 超參數調優
=========================================================
使用 Optuna 貝氏超參數搜索，針對「每期命中數（Avg Hits@5）」
做時序驗證最大化。

驗證方式：
  最後 60 期作為驗證集（模擬 rolling eval），
  前面全部作為訓練集（expanding window）。

搜索範圍：
  XGBoost: n_estimators, max_depth, learning_rate,
           subsample, colsample_bytree, min_child_weight
  RF:      n_estimators, max_depth, min_samples_leaf, max_features

輸出：
  - 終端機顯示每個 trial 的結果
  - results/best_params.json（最佳超參數）
  - 自動更新 src/models/trainer.py 中的模型參數

用法：
    python scripts/tune_hyperparams.py --model xgb --n-trials 30
    python scripts/tune_hyperparams.py --model rf  --n-trials 30
    python scripts/tune_hyperparams.py --model all --n-trials 30
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.database.repository import DrawRepository
from src.features.feature_builder import build_ml_feature_matrix
from config.settings import LOOKBACK_PERIODS

logging.basicConfig(level=logging.WARNING)   # 靜音 INFO，只顯示進度
logger = logging.getLogger(__name__)

DB_PATH      = ROOT_DIR / "data" / "superlotto.db"
WHITE_NUMBERS = list(range(1, 48))
VAL_DRAWS    = 60    # 最後幾期作為驗證集
N_WHITE      = 5
POS_WEIGHT   = (47 - 5) / 5   # ≈ 8.4


def load_data():
    repo = DrawRepository(DB_PATH)
    df   = repo.get_all_draws()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df = df[df["draw_date"] >= "2020-01-01"].sort_values("draw_date").reset_index(drop=True)
    return df


def build_feature_matrix(df):
    X, y, _ = build_ml_feature_matrix(df, lookback=LOOKBACK_PERIODS)
    return X, y


def precision_at_5(model, X_val, y_val):
    """計算模型在驗證集上的平均命中數（Avg Hits@5）。"""
    proba_list = model.predict_proba(X_val)
    y_prob = np.column_stack([p[:, 1] for p in proba_list])
    hits = []
    for i in range(len(X_val)):
        top5 = np.argsort(y_prob[i])[::-1][:5]
        hits.append(sum(y_val[i, j] for j in top5))
    return float(np.mean(hits))


def objective_xgb(trial, X_train, y_train, X_val, y_val):
    from xgboost import XGBClassifier
    from sklearn.multioutput import MultiOutputClassifier

    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth":        trial.suggest_int("max_depth", 2, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": POS_WEIGHT,
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           1,   # MultiOutput handles parallelism across 47 targets
    }

    base = XGBClassifier(**params)
    model = MultiOutputClassifier(base, n_jobs=-1)  # train 47 classifiers in parallel
    model.fit(X_train, y_train)
    return precision_at_5(model, X_val, y_val)


def objective_rf(trial, X_train, y_train, X_val, y_val):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth":         trial.suggest_int("max_depth", 3, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        "class_weight":      "balanced",
        "random_state":      42,
        "n_jobs":            1,   # MultiOutput handles parallelism
    }

    base = RandomForestClassifier(**params)
    model = MultiOutputClassifier(base, n_jobs=-1)  # train 47 classifiers in parallel
    model.fit(X_train, y_train)
    return precision_at_5(model, X_val, y_val)


def run_study(model_type: str, X_train, y_train, X_val, y_val, n_trials: int):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    direction = "maximize"
    study = optuna.create_study(direction=direction,
                                 sampler=optuna.samplers.TPESampler(seed=42))

    obj = objective_xgb if model_type == "xgb" else objective_rf

    print(f"\n  搜索 {model_type.upper()} 超參數（{n_trials} trials，驗證集 {VAL_DRAWS} 期）")
    print(f"  資料：訓練 {len(X_train)}期 / 驗證 {len(X_val)}期")
    print(f"  隨機基準 Avg Hits ≈ {5*5/47:.4f}\n")

    best_so_far = 0.0
    for t in range(n_trials):
        trial = study.ask()
        value = obj(trial, X_train, y_train, X_val, y_val)
        study.tell(trial, value)

        marker = ""
        if value > best_so_far:
            best_so_far = value
            marker = " ← best"

        print(f"  Trial {t+1:3d}/{n_trials}  Avg Hits={value:.4f}{marker}")

    best = study.best_trial
    print(f"\n  {'='*50}")
    print(f"  最佳 Avg Hits = {best.value:.4f}")
    print(f"  最佳超參數:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    return best.value, best.params


def main():
    parser = argparse.ArgumentParser(description="超參數調優（Optuna）")
    parser.add_argument("--model",    default="all", choices=["xgb", "rf", "all"])
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    print("=== SuperLotto 超參數調優 ===")
    print(f"模型：{args.model}，Trials：{args.n_trials}")

    # 載入資料與特徵矩陣
    print("\n載入資料與建立特徵矩陣...")
    df = load_data()
    print(f"  資料：{len(df)} 期")
    X, y = build_feature_matrix(df)
    print(f"  特徵矩陣：{X.shape}")

    # 時序切分（最後 VAL_DRAWS 期作為驗證）
    split = len(X) - VAL_DRAWS
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    results = {}

    models_to_tune = ["xgb", "rf"] if args.model == "all" else [args.model]

    for model_type in models_to_tune:
        print(f"\n{'─'*60}")
        val, params = run_study(model_type, X_train, y_train, X_val, y_val, args.n_trials)
        results[model_type] = {"best_val": val, "params": params}

    # 儲存結果
    out = ROOT_DIR / "results" / "best_params.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n最佳超參數已存至：{out}")

    # 顯示與預設值的比較
    print("\n=== 最終比較 ===")
    defaults = {
        "xgb": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                 "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1},
        "rf":  {"n_estimators": 100, "max_depth": "None", "min_samples_leaf": 3,
                 "max_features": "sqrt"},
    }
    for model_type, res in results.items():
        print(f"\n{model_type.upper()}  最佳 Avg Hits = {res['best_val']:.4f}")
        print(f"  {'參數':<22} {'預設':>12} {'最佳':>12}")
        print(f"  {'─'*48}")
        for k, v in res["params"].items():
            default_v = defaults.get(model_type, {}).get(k, "—")
            print(f"  {k:<22} {str(default_v):>12} {str(v):>12}")


if __name__ == "__main__":
    main()
