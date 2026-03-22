"""
predict_next_draw.py — 使用最佳模型預測下一期號碼
===================================================
用全部歷史資料訓練最佳模型（XGB + Ensemble），
生成下一期最可能的 5 組白球 + Mega 組合。

輸出：
  - 每個模型的 top-5 推薦白球和 Mega
  - Multi-ticket 策略：top-1 ~ top-5 組合
  - 各球機率分布（前10名）
  - 儲存到 results/next_draw_prediction.json

用法：
    python scripts/predict_next_draw.py
    python scripts/predict_next_draw.py --models xgb,rf,ensemble --n-tickets 5
"""

import argparse
import json
import sys
from pathlib import Path
from itertools import combinations
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np

from src.database.repository import DrawRepository

DB_PATH = ROOT_DIR / "data" / "superlotto.db"
WHITE_NUMBERS = list(range(1, 48))
MEGA_NUMBERS  = list(range(1, 28))


def build_model(name: str):
    """建立並回傳指定模型（未訓練）。"""
    if name == "xgb":
        from src.models.classifier import LotteryClassifier
        class _XGBWrapper:
            def fit(self, df):
                self._clf = LotteryClassifier()
                self._clf.fit(df, train_xgb=True)
                return self
            def predict(self, n_white=5, **kw):
                r = self._clf.predict_next(top_k=n_white)
                p = r.get("xgb_prediction", {})
                proba = {WHITE_NUMBERS[i]: float(v)
                         for i, v in enumerate(
                             np.array([pp[0, 1] for pp in
                                       self._clf.xgb_model.predict_proba(
                                           self._clf._X_all[[-1]])]))}
                return {"white_balls": p.get("white_balls", []),
                        "mega_balls": [], "proba": proba}
        return _XGBWrapper()

    if name == "rf":
        from src.models.classifier import LotteryClassifier
        class _RFWrapper:
            def fit(self, df):
                self._clf = LotteryClassifier()
                self._clf.fit(df, train_xgb=False)
                return self
            def predict(self, n_white=5, **kw):
                r = self._clf.predict_next(top_k=n_white)
                p = r.get("rf_prediction", {})
                proba = {WHITE_NUMBERS[i]: float(v)
                         for i, v in enumerate(
                             np.array([pp[0, 1] for pp in
                                       self._clf.rf_model.predict_proba(
                                           self._clf._X_all[[-1]])]))}
                return {"white_balls": p.get("white_balls", []),
                        "mega_balls": [], "proba": proba}
        return _RFWrapper()

    if name == "frequency":
        from src.models.baseline import FrequencyBaseline
        return FrequencyBaseline()

    if name == "ensemble":
        from src.models.ensemble import EnsemblePredictor
        return EnsemblePredictor()

    if name == "bayesian":
        from src.models.bayesian import BayesianPredictor
        return BayesianPredictor()

    if name == "markov":
        from src.models.markov import MarkovPredictor
        return MarkovPredictor()

    if name == "montecarlo":
        from src.models.monte_carlo import MonteCarloPredictor
        return MonteCarloPredictor()

    if name == "decision_tree":
        from src.models.decision_tree import DecisionTreePredictor
        return DecisionTreePredictor()

    if name == "knn":
        from src.models.knn_model import KNNPredictor
        return KNNPredictor()

    raise ValueError(f"未知模型：{name}")


def generate_top_k_tickets(white_proba, mega_proba, n_tickets=5, n_candidates=12):
    """從機率分布生成前 K 組最可能的組合。"""
    sorted_balls = sorted(white_proba.keys(), key=lambda b: -white_proba[b])
    candidates   = sorted_balls[:n_candidates]
    scored = [
        (sum(white_proba[b] for b in combo), sorted(combo))
        for combo in combinations(candidates, 5)
    ]
    scored.sort(reverse=True)
    best_mega = max(mega_proba.keys(), key=lambda m: mega_proba[m]) if mega_proba else None
    return [{"white_balls": wb, "mega_ball": best_mega, "score": round(sc, 6)}
            for sc, wb in scored[:n_tickets]]


def main():
    parser = argparse.ArgumentParser(description="預測下一期開獎號碼")
    parser.add_argument("--models",     default="xgb,rf,ensemble,frequency,markov")
    parser.add_argument("--n-tickets",  type=int, default=5)
    parser.add_argument("--n-candidates", type=int, default=12)
    parser.add_argument("--start-date", default="2020-01-01")
    args = parser.parse_args()

    # 載入資料
    repo = DrawRepository(DB_PATH)
    df   = repo.get_all_draws()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df = df[df["draw_date"] >= args.start_date].sort_values("draw_date").reset_index(drop=True)

    last_draw = df.iloc[-1]
    last_date = str(last_draw["draw_date"])[:10]

    print(f"{'='*65}")
    print(f" SuperLotto Plus — 下一期預測")
    print(f" 最後已知開獎：{last_date}（第 {last_draw['draw_number']} 期）")
    print(f" 訓練資料：{len(df)} 期（{args.start_date} 起）")
    print(f"{'='*65}")

    model_names = [m.strip() for m in args.models.split(",")]
    all_predictions = {}
    SEP = "─" * 65

    for name in model_names:
        print(f"\n[{name.upper()}] 訓練中...", end=" ", flush=True)
        try:
            model = build_model(name)
            model.fit(df)
            result = model.predict(n_white=5)
            white  = result.get("white_balls", [])
            mega   = result.get("mega_balls", [])
            proba  = result.get("proba", {})
            mega_p = result.get("mega_proba", {})

            print("完成")
            print(f"  推薦白球：{white}  │  Mega：{mega if mega else '—'}")

            # 機率前10名
            if proba:
                top10 = sorted(proba.items(), key=lambda x: -x[1])[:10]
                top10_str = "  ".join(f"{b}({p:.3f})" for b, p in top10)
                print(f"  機率Top10：{top10_str}")

            # Multi-ticket
            if proba:
                tickets = generate_top_k_tickets(proba, mega_p or {},
                                                  args.n_tickets, args.n_candidates)
                print(f"\n  Multi-Ticket 策略（前{args.n_tickets}組）：")
                for i, t in enumerate(tickets, 1):
                    mb = f" + Mega {t['mega_ball']}" if t['mega_ball'] else ""
                    print(f"    第{i}組：{t['white_balls']}{mb}  (score={t['score']:.5f})")

            all_predictions[name] = {
                "white_balls": white,
                "mega_balls":  mega,
                "proba_top10": {str(b): round(p, 5) for b, p in
                                sorted(proba.items(), key=lambda x: -x[1])[:10]} if proba else {},
                "tickets":     tickets if proba else [],
            }

        except Exception as e:
            print(f"失敗：{e}")
            all_predictions[name] = {"error": str(e)}

    # 整合投票（所有模型白球的多數決）
    print(f"\n{SEP}")
    print(" 整合投票（所有模型白球出現次數）")
    print(SEP)
    vote_count = {}
    for name, res in all_predictions.items():
        if "white_balls" in res:
            for b in res["white_balls"]:
                vote_count[b] = vote_count.get(b, 0) + 1

    top_voted = sorted(vote_count.items(), key=lambda x: -x[1])[:10]
    print("  號碼  出現次數")
    for b, c in top_voted:
        bar = "█" * c
        print(f"  {b:3d}   {c} {bar}")

    consensus = sorted([b for b, _ in top_voted[:5]])
    print(f"\n  共識推薦（前5顆）：{consensus}")

    # 儲存
    output = {
        "generated_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
        "last_draw_date": last_date,
        "last_draw_number": int(last_draw["draw_number"]),
        "training_draws": len(df),
        "models":         all_predictions,
        "consensus":      consensus,
        "vote_count":     {str(b): c for b, c in sorted(vote_count.items(), key=lambda x: -x[1])},
    }

    out_path = ROOT_DIR / "results" / "next_draw_prediction.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  預測結果已存至：{out_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
