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

import numpy as np
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
    TICKET_COST,
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
    對單一模型執行完整的 Walk-Forward 評估（Per-Draw 版本）。

    【修正後的正確評估流程】

      每個 Fold 內部，對測試視窗的每一期做獨立預測：

        cumulative = train_df（2020-01 ~ 2021-12）
        ↓
        第 1 期預測：用 cumulative 訓練 → 預測 → vs 2022-01-05 實際開獎 → 記錄
        把 2022-01-05 加入 cumulative
        ↓
        第 2 期預測：用 cumulative（多了一期）訓練 → 預測 → vs 2022-01-08 → 記錄
        把 2022-01-08 加入 cumulative
        ↓
        ... 重複直到 Fold 結束

      這樣每期預測使用的是「到上一期為止的全部歷史」，
      模型永遠不會「看到未來」，且每期都能學到最新資訊。

      與舊版的差別：
        舊版：訓練一次 → 用同一組球對整個 Fold 的 52 期比較（不更新）
        新版：每期都重訓 → 每期給出不同推薦 → 逐期比較（正確）

      速度說明：
        Bayesian/Markov/Frequency 每次重訓約 1–10ms，52 期合計 < 1 秒。
        MonteCarlo 每次約 500ms，52 期約 26 秒（較慢）。

    Args:
        model_name: 模型名稱
        df:         完整資料 DataFrame（已按日期升序）
        splitter:   WalkForwardSplit 切分器
        show_folds: 是否顯示每個 Fold 的詳細結果

    Returns:
        (fold_results, fold_infos, agg)
    """
    fold_results: List[dict] = []
    fold_infos   = []
    total_folds  = 0

    for train_df, test_df, fold_info in splitter.split(df):
        total_folds += 1
        fold_infos.append(fold_info)
        fold_metric = FoldMetrics(fold_info=fold_info, model_name=model_name)

        # ── Per-Draw 滾動預測 ────────────────────────────────────────────────
        # cumulative 從訓練集開始，每看到一期實際開獎就擴充一期
        cumulative = train_df.copy()

        test_rows = list(test_df.iterrows())
        for draw_i, (_, row) in enumerate(test_rows):
            actual_balls = [int(row[c]) for c in WHITE_COLS]
            actual_mega  = int(row["mega_number"]) if "mega_number" in row.index else None

            # ① 以「目前為止的所有歷史（不含本期）」訓練模型
            try:
                model = build_model(model_name)
                model.fit(cumulative)
            except Exception as e:
                logger.warning(
                    f"Fold {fold_info.fold_idx} draw {draw_i+1} [{model_name}] 訓練失敗：{e}"
                )
                # 訓練失敗：記錄 0 命中並繼續
                fold_metric.add([], actual_balls, None, actual_mega)
                cumulative = pd.concat(
                    [cumulative, test_df.iloc[[draw_i]]], ignore_index=True
                )
                continue

            # ② 預測（此時模型看不到 row 這期的開獎）
            try:
                prediction      = model.predict(n_white=5)
                predicted_balls = prediction.get("white_balls", [])
                mega_list       = prediction.get("mega_balls", [])
                predicted_mega  = int(mega_list[0]) if mega_list else None
            except Exception as e:
                logger.warning(
                    f"Fold {fold_info.fold_idx} draw {draw_i+1} [{model_name}] 預測失敗：{e}"
                )
                predicted_balls, predicted_mega = [], None

            # ③ 記錄命中數與獎金（此時才「揭曉」本期實際開獎）
            fold_metric.add(
                predicted_balls, actual_balls,
                predicted_mega,  actual_mega,
                k=5,
            )

            # ④ 把本期實際開獎加入累積訓練資料，供下一期使用
            cumulative = pd.concat(
                [cumulative, test_df.iloc[[draw_i]]], ignore_index=True
            )

        # ── 彙整本 Fold ──────────────────────────────────────────────────────
        summary = fold_metric.summary()
        fold_results.append(summary)

        if show_folds:
            ev   = summary.get("ev_per_draw", 0)
            net  = summary.get("net_ev_per_draw", 0)
            dist = "  ".join(
                f"{k}中:{v}" for k, v in summary.get("hit_dist_count", {}).items()
            )
            print(
                f"  Fold {fold_info.fold_idx:2d} | 測試 {fold_info.n_test:3d} 期 | "
                f"Avg Hits: {summary['mean_hits']:.3f} | "
                f"EV/期: ${ev:.4f} (淨: ${net:+.4f}) | {dist}"
            )

    # ── 彙整所有 Fold ────────────────────────────────────────────────────────
    valid_results = [r for r in fold_results if r]
    agg = aggregate_fold_results(valid_results)

    # 加入跨 Fold 的 EV 統計
    ev_per_fold = [r.get("ev_per_draw", 0) for r in valid_results]
    if ev_per_fold:
        agg["ev_avg"]     = float(np.mean(ev_per_fold))
        agg["net_ev_avg"] = agg["ev_avg"] - TICKET_COST
        agg["roi_pct"]    = agg["net_ev_avg"] / TICKET_COST * 100

    logger.info(
        f"[{model_name}] 完成 {total_folds} 個 Fold | "
        f"Avg Hits={agg.get('mean_hits_avg', 0):.3f} | "
        f"EV/期=${agg.get('ev_avg', 0):.4f} | "
        f"淨EV=${agg.get('net_ev_avg', -1):.4f}"
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
