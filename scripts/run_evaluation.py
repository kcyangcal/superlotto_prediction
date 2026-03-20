"""
run_evaluation.py — Walk-Forward 滾動驗證評估
================================================
對各模型執行 Rolling Hold-Out 評估，觀察每個 Fold 的預測能力，
並輸出跨 Fold 的彙整統計表。

使用方式：
    python scripts/run_evaluation.py

選項：
    --train-months N   初始訓練視窗月數（預設 24）
    --test-months  N   每個測試視窗月數（預設 6）
    --step-months  N   滾動步長月數（預設 6，= test-months 表示非重疊）
    --start-date   D   僅使用此日期後的資料，格式 YYYY-MM-DD（預設 2020-01-01）
    --models       M   逗號分隔的模型名稱，如 frequency,markov,bayesian
                       可用名稱：frequency, montecarlo, markov, bayesian
                       （RF/XGBoost 在 walk-forward 中過慢，預設不啟用）
    --show-folds       印出每個 Fold 的詳細命中分布

範例（快速評估）：
    python scripts/run_evaluation.py --train-months 18 --test-months 3

範例（完整評估）：
    python scripts/run_evaluation.py --models frequency,montecarlo,markov,bayesian
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from config.settings   import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.database.repository import DrawRepository
from src.evaluation.walk_forward import WalkForwardSplit
from src.evaluation.metrics import (
    FoldMetrics,
    aggregate_fold_results,
    print_fold_report,
    print_model_comparison_table,
    WHITE_COLS,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_COLS = ["n1", "n2", "n3", "n4", "n5"]

# ─────────────────────────────────────────────────────────────────────────────
# 模型工廠
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str):
    """
    根據名稱建立模型實例（延遲 import，避免循環依賴）。
    回傳未訓練的模型物件。
    """
    name = name.lower().strip()
    if name == "frequency":
        from src.models.baseline import FrequencyBaseline
        return FrequencyBaseline()
    elif name == "montecarlo":
        from src.models.monte_carlo import MonteCarloPredictor
        return MonteCarloPredictor(n_simulations=50_000)   # walk-forward 用較少次數加速
    elif name == "markov":
        from src.models.markov import MarkovPredictor
        return MarkovPredictor()
    elif name == "bayesian":
        from src.models.bayesian import BayesianPredictor
        return BayesianPredictor()
    else:
        raise ValueError(
            f"未知模型名稱：{name!r}\n"
            "可用名稱：frequency, montecarlo, markov, bayesian"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 單模型 Walk-Forward 執行
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model_walk_forward(
    model_name: str,
    df:         pd.DataFrame,
    splitter:   WalkForwardSplit,
    show_folds: bool = False,
) -> tuple:
    """
    對單一模型執行完整的 Walk-Forward 評估。

    評估邏輯（一個 Fold 的流程）：
      1. 用 train_df 訓練模型
      2. 模型 predict() 輸出推薦的 5 顆白球
      3. 對 test_df 中的「每一期」，計算這 5 顆球的命中數
      4. 收集所有 test 期的命中數，計算均值

    注意：
      這裡模型對 test_df 中的所有期次「只預測一次」（以 train_df 結束點為基準）。
      這是一個「折中方案」——完整的 walk-forward 應每期重訓，但這樣速度太慢。
      理解這個假設的限制非常重要：若市場的統計特性（號碼分布）在測試期間
      發生顯著變化（Concept Drift），這個方法會低估實際誤差。

    Args:
        model_name: 模型名稱（用於 build_model 和報表顯示）
        df:         完整資料 DataFrame
        splitter:   WalkForwardSplit 切分器
        show_folds: 是否在終端輸出每個 Fold 的詳細命中分布

    Returns:
        (fold_results, fold_infos, agg)
    """
    fold_results: List[dict] = []
    fold_infos   = []
    total_folds  = 0

    for train_df, test_df, fold_info in splitter.split(df):
        total_folds += 1
        fold_infos.append(fold_info)

        # ── 訓練 ────────────────────────────────────────────────────────────
        try:
            model = build_model(model_name)
            model.fit(train_df)
        except Exception as e:
            logger.warning(f"Fold {fold_info.fold_idx} [{model_name}] 訓練失敗：{e}")
            fold_results.append({})
            continue

        # ── 預測 ────────────────────────────────────────────────────────────
        try:
            prediction = model.predict(n_white=5)
            predicted_balls = prediction.get("white_balls", [])
        except Exception as e:
            logger.warning(f"Fold {fold_info.fold_idx} [{model_name}] 預測失敗：{e}")
            fold_results.append({})
            continue

        # ── 評估：對 test_df 每一期計算命中數 ─────────────────────────────
        fold_metric = FoldMetrics(fold_info=fold_info, model_name=model_name)

        for _, row in test_df.iterrows():
            actual_balls = [int(row[c]) for c in WHITE_COLS]
            fold_metric.add(predicted_balls, actual_balls, k=5)

        summary = fold_metric.summary()
        fold_results.append(summary)

        if show_folds:
            dist_str = "  ".join(
                f"{k}中:{v}" for k, v in summary.get("hit_dist_count", {}).items()
            )
            print(
                f"  Fold {fold_info.fold_idx:2d} │ 測試 {fold_info.n_test:3d} 期 │ "
                f"Avg Hits: {summary['mean_hits']:.3f} │ {dist_str}"
            )

    # ── 彙整所有 Fold ────────────────────────────────────────────────────────
    valid_results = [r for r in fold_results if r]
    agg = aggregate_fold_results(valid_results)

    logger.info(
        f"[{model_name}] 完成 {total_folds} 個 Fold，"
        f"整體 Avg Hits={agg.get('mean_hits_avg', 0):.3f} "
        f"（隨機基準≈0.532）"
    )

    return fold_results, fold_infos, agg


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk-Forward 滾動驗證評估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train-months", type=int, default=24,   help="初始訓練視窗月數（預設 24）")
    parser.add_argument("--test-months",  type=int, default=6,    help="測試視窗月數（預設 6）")
    parser.add_argument("--step-months",  type=int, default=6,    help="滾動步長月數（預設 6）")
    parser.add_argument("--start-date",   type=str, default="2020-01-01",
                        help="只使用此日期後的資料（預設 2020-01-01）")
    parser.add_argument("--models",       type=str,
                        default="frequency,markov,bayesian",
                        help="逗號分隔的模型名稱（預設 frequency,markov,bayesian）")
    parser.add_argument("--show-folds",   action="store_true",
                        help="印出每個 Fold 的詳細命中分布")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DB_PATH.exists():
        logger.error("找不到資料庫！請先執行：python scripts/init_db.py")
        sys.exit(1)

    # ── 1. 讀取資料 ──────────────────────────────────────────────────────────
    repo = DrawRepository(DB_PATH)
    df   = repo.get_all_draws()
    df["draw_date"] = pd.to_datetime(df["draw_date"])

    # 篩選起始日期
    df = df[df["draw_date"] >= pd.to_datetime(args.start_date)].copy()
    df = df.sort_values("draw_date").reset_index(drop=True)

    logger.info(f"使用資料：{args.start_date} 之後，共 {len(df)} 期")

    if len(df) < 100:
        logger.error(f"資料量不足（只有 {len(df)} 期），無法進行評估")
        sys.exit(1)

    # ── 2. 建立切分器並顯示切分摘要 ─────────────────────────────────────────
    splitter = WalkForwardSplit(
        train_months = args.train_months,
        test_months  = args.test_months,
        step_months  = args.step_months,
    )

    print("\n" + "=" * 62)
    print(" Walk-Forward 切分預覽")
    print("=" * 62)
    try:
        fold_summary = splitter.get_fold_summary(df)
        print(fold_summary.to_string(index=False))
        print(f"\n  總計：{len(fold_summary)} 個 Fold")
    except ValueError as e:
        logger.error(f"切分失敗：{e}")
        sys.exit(1)

    # ── 3. 逐模型執行 Walk-Forward 評估 ─────────────────────────────────────
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    all_agg: Dict[str, dict] = {}

    for model_name in model_names:
        print(f"\n{'─'*62}")
        print(f" 評估模型：{model_name}")
        print(f"{'─'*62}")

        fold_results, fold_infos, agg = evaluate_model_walk_forward(
            model_name  = model_name,
            df          = df,
            splitter    = splitter,
            show_folds  = args.show_folds,
        )

        valid_results = [r for r in fold_results if r]
        valid_infos   = [fi for r, fi in zip(fold_results, fold_infos) if r]

        if valid_results:
            print_fold_report(model_name, valid_results, valid_infos, agg)
            all_agg[model_name] = agg
        else:
            logger.warning(f"模型 {model_name} 無有效評估結果")

    # ── 4. 所有模型彙整比較表 ────────────────────────────────────────────────
    if len(all_agg) > 1:
        print_model_comparison_table(all_agg)

    # ── 5. 學習重點提示 ──────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print(" 學習重點")
    print("=" * 62)
    print(f"  隨機基準 Avg Hits = {5*5/47:.4f}（理論值：5 × 5/47）")
    print(f"  隨機基準 Prec@5  = {5/47*100:.2f}%")
    print()
    print("  • 若模型 Avg Hits 穩定高於 0.532，值得進一步分析原因")
    print("  • 若各 Fold 波動很大（std 高），表示模型不穩定")
    print("  • Concept Drift：若後期 Fold 明顯差於前期，")
    print("    表示歷史統計規律可能隨時間改變")
    print("  • p > 0.05 表示與隨機的差異可能只是統計噪音")
    print("=" * 62)


if __name__ == "__main__":
    main()
