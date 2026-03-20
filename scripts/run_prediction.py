"""
run_prediction.py — 執行完整預測流程
=======================================
依序執行：
  1. 讀取資料庫中的所有開獎資料
  2. 訓練所有模型（RF + XGBoost + 基準 + 蒙地卡羅 + 馬可夫鏈 +
                   貝氏推論 + KNN + 決策樹 + 遺傳演算法 + 集成學習）
  3. 評估 RF/XGBoost 模型在測試集上的表現
  4. 輸出各模型的下一期號碼推薦對照表

使用方式：
  python scripts/run_prediction.py

選項：
  --no-xgb        跳過 XGBoost 訓練（較快）
  --no-knn        跳過 KNN（特徵向量建構耗時 20–60 秒）
  --no-dt         跳過決策樹
  --no-ensemble   跳過集成學習
  --save          訓練後存檔 RF/XGBoost 模型
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.database.repository import DrawRepository
from src.models.classifier   import LotteryClassifier
from src.models.predictor    import print_prediction_report

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

SEPARATOR = "=" * 60


def parse_args():
    parser = argparse.ArgumentParser(description="SuperLotto Plus 號碼預測（全模型）")
    parser.add_argument("--no-xgb",      action="store_true", help="跳過 XGBoost")
    parser.add_argument("--no-knn",      action="store_true", help="跳過 KNN（較耗時）")
    parser.add_argument("--no-dt",       action="store_true", help="跳過決策樹")
    parser.add_argument("--no-ensemble", action="store_true", help="跳過集成學習")
    parser.add_argument("--save",        action="store_true", help="儲存 RF/XGBoost 模型")
    return parser.parse_args()


def run_single_model(name: str, model_cls, df, **kwargs) -> dict:
    """訓練單一模型並回傳預測結果，發生例外時回傳空字典。"""
    try:
        model = model_cls(**kwargs)
        model.fit(df)
        return model.predict()
    except Exception as e:
        logger.warning(f"[{name}] 執行失敗：{e}")
        return {}


def print_comparison_table(all_results: dict) -> None:
    """印出所有模型的白球預測對照表。"""
    print(f"\n{SEPARATOR}")
    print(" SuperLotto Plus — 各模型白球預測對照")
    print(SEPARATOR)
    print(f"{'模型':<20} {'推薦白球 (5 顆)'}")
    print("-" * 60)
    for model_name, result in all_results.items():
        balls = result.get("white_balls", [])
        balls_str = "  ".join(f"{b:2d}" for b in balls) if balls else "（失敗）"
        print(f"{model_name:<20} {balls_str}")
    print(SEPARATOR)

    # 統計每個號碼被幾個模型推薦
    vote_count: dict[int, int] = {}
    for result in all_results.values():
        for b in result.get("white_balls", []):
            vote_count[b] = vote_count.get(b, 0) + 1

    top_by_vote = sorted(vote_count.items(), key=lambda x: -x[1])[:10]
    print("\n[ 跨模型共識排行（被推薦次數最多的白球 Top 10）]")
    print("-" * 40)
    for num, cnt in top_by_vote:
        bar = "#" * cnt
        print(f"  號碼 {num:2d}：{bar} ({cnt} 個模型)")

    # 找出被最多模型同時推薦的組合
    if top_by_vote:
        consensus_balls = sorted([num for num, _ in top_by_vote[:5]])
        print(f"\n  集成共識推薦（Top 5 高票號碼）：{consensus_balls}")

    print()
    print("  *** 提醒：彩票為獨立隨機事件，預測結果僅供學習參考 ***")
    print(SEPARATOR)


def main():
    args = parse_args()

    if not DB_PATH.exists():
        logger.error("找不到資料庫！請先執行：init_db.py -> scrape_all.py")
        sys.exit(1)

    # ── 1. 讀取資料 ──────────────────────────────────────────────────────────
    repo       = DrawRepository(DB_PATH)
    draw_count = repo.get_draw_count()
    logger.info(f"資料庫共有 {draw_count} 筆開獎資料")

    if draw_count < 100:
        logger.warning(f"資料量太少（{draw_count} 筆），模型可靠性很低")

    df = repo.get_all_draws()

    all_results: dict = {}

    # ── 2a. RF + XGBoost（透過 LotteryClassifier）────────────────────────────
    logger.info(f"\n{SEPARATOR}")
    logger.info("訓練 RandomForest / XGBoost 模型...")
    clf = LotteryClassifier()
    clf.fit(df, train_xgb=not args.no_xgb)

    if args.save:
        clf.save_models()

    eval_results = clf.evaluate()
    clf_pred     = clf.predict_next()
    all_results["隨機森林"]  = clf_pred.get("random_forest",  {})
    if not args.no_xgb:
        all_results["XGBoost"] = clf_pred.get("xgboost", {})

    # ── 2b. 各獨立模型 ───────────────────────────────────────────────────────
    from src.models.baseline      import FrequencyBaseline
    from src.models.monte_carlo   import MonteCarloPredictor
    from src.models.markov        import MarkovPredictor
    from src.models.bayesian      import BayesianPredictor
    from src.models.genetic       import GeneticPredictor

    logger.info(f"\n{SEPARATOR}")
    logger.info("訓練頻率基準模型...")
    all_results["頻率基準"] = run_single_model("FrequencyBaseline", FrequencyBaseline, df)

    logger.info(f"\n{SEPARATOR}")
    logger.info("訓練蒙地卡羅模型...")
    all_results["蒙地卡羅"] = run_single_model("MonteCarlo", MonteCarloPredictor, df, n_simulations=100_000)

    logger.info(f"\n{SEPARATOR}")
    logger.info("訓練馬可夫鏈模型...")
    all_results["馬可夫鏈"] = run_single_model("Markov", MarkovPredictor, df)

    logger.info(f"\n{SEPARATOR}")
    logger.info("訓練貝氏推論模型...")
    all_results["貝氏推論"] = run_single_model("Bayesian", BayesianPredictor, df)

    logger.info(f"\n{SEPARATOR}")
    logger.info("訓練遺傳演算法模型...")
    all_results["遺傳演算法"] = run_single_model(
        "Genetic", GeneticPredictor, df, n_generations=300
    )

    if not args.no_knn:
        from src.models.knn_model import KNNPredictor
        logger.info(f"\n{SEPARATOR}")
        logger.info("訓練 KNN 模型（可能需要數十秒）...")
        all_results["KNN"] = run_single_model("KNN", KNNPredictor, df, k=10)

    if not args.no_dt:
        from src.models.decision_tree import DecisionTreePredictor
        logger.info(f"\n{SEPARATOR}")
        logger.info("訓練決策樹模型...")
        all_results["決策樹"] = run_single_model(
            "DecisionTree", DecisionTreePredictor, df, max_depth=5
        )

    # ── 2c. 集成學習 ──────────────────────────────────────────────────────────
    if not args.no_ensemble:
        from src.models.ensemble import EnsemblePredictor
        logger.info(f"\n{SEPARATOR}")
        logger.info("訓練集成學習模型...")
        try:
            ensemble = EnsemblePredictor(
                enable_knn = not args.no_knn,
                enable_dt  = not args.no_dt,
            )
            ensemble.fit(df)
            ens_result = ensemble.predict()
            all_results["集成學習"] = ens_result

            # 印出子模型比較表
            comparison = ensemble.get_model_comparison()
            logger.info("集成子模型預測明細：\n" + comparison.to_string(index=False))
        except Exception as e:
            logger.warning(f"集成學習失敗：{e}")

    # ── 3. 評估報表（RF/XGBoost）──────────────────────────────────────────────
    print_prediction_report(clf_pred, eval_results)

    # ── 4. 全模型對照表 ───────────────────────────────────────────────────────
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
