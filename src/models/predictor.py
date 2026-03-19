"""
predictor.py — 預測結果格式化輸出
====================================
將 LotteryClassifier.predict_next() 的原始輸出
格式化成易讀的報表，並顯示免責聲明。
"""

import logging
from datetime import date

from config.settings import LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

DISCLAIMER = """
╔══════════════════════════════════════════════════════════════╗
║                        ⚠️  重要聲明                          ║
╠══════════════════════════════════════════════════════════════╣
║  本輸出僅為機器學習教學練習之用。                             ║
║  加州 SuperLotto Plus 開獎為完全隨機的獨立事件。              ║
║  任何統計模型均無法有效預測彩票結果。                         ║
║  本程式輸出不構成購票建議。賭博有風險，請自行評估。            ║
╚══════════════════════════════════════════════════════════════╝
"""


def print_prediction_report(prediction: dict, eval_results: dict = None):
    """
    將預測結果格式化為可讀報表並印出。

    Args:
        prediction:   LotteryClassifier.predict_next() 的回傳值
        eval_results: LotteryClassifier.evaluate() 的回傳值（可選，用於顯示模型精度）
    """
    print(DISCLAIMER)
    print(f"📅 預測日期：{date.today()}")
    print("=" * 60)
    print("🎱 SuperLotto Plus 下一期號碼推薦")
    print("=" * 60)

    # ── 各模型推薦號碼 ────────────────────────────────────────────────────────
    models_display = [
        ("L0 頻率基準",   "baseline_prediction"),
        ("L1 Random Forest", "rf_prediction"),
        ("L2 XGBoost",    "xgb_prediction"),
        ("🎯 整合預測",   "ensemble_prediction"),
    ]

    for label, key in models_display:
        if key not in prediction:
            continue
        pred = prediction[key]
        balls = pred.get("white_balls", [])
        balls_str = "  ".join(f"{n:2d}" for n in sorted(balls))
        print(f"\n{label}:")
        print(f"  白球：[ {balls_str} ]")

        # 若有機率資訊，也一併顯示
        if "probabilities" in pred:
            top_probs = sorted(
                pred["probabilities"].items(), key=lambda x: -x[1]
            )[:5]
            prob_str = "  |  ".join(f"{n}: {p*100:.1f}%" for n, p in top_probs)
            print(f"  機率：{prob_str}")

    # ── 模型評估摘要 ──────────────────────────────────────────────────────────
    if eval_results:
        print("\n" + "-" * 60)
        print("📊 模型歷史回測表現（測試集）")
        print(f"  隨機基準     Precision@5 = {5/47*100:.2f}%")
        if "random_forest" in eval_results:
            p5 = eval_results["random_forest"]["precision_at_5_pct"]
            print(f"  Random Forest Precision@5 = {p5}")
        if "xgboost" in eval_results:
            p5 = eval_results["xgboost"]["precision_at_5_pct"]
            print(f"  XGBoost      Precision@5 = {p5}")

    print("=" * 60)
    print("⚠️  再次提醒：以上號碼僅供學習參考，不代表任何預測依據")
