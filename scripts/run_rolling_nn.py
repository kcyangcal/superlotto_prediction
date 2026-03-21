"""
run_rolling_nn.py — 神經網路 + 全模型逐期 Rolling 評估
=========================================================

【評估邏輯：True Walk-Forward，每期更新】

  與 run_evaluation.py 的差別：

  run_evaluation.py（Fold 級評估）：
    每個 Fold 訓練一次 → 預測整個測試視窗的所有期次
    優點：快速；缺點：測試視窗內模型不更新，無法反映持續學習

  run_rolling_nn.py（逐期評估，本腳本）：
    ┌─────────────────────────────────────────────────────┐
    │  for each draw in test window:                      │
    │    1. predict  (使用截至上一期的所有資料)            │
    │    2. reveal   (看到本期實際開獎)                    │
    │    3. update   (將本期資料加入模型)                  │
    │    4. 記錄命中數                                     │
    └─────────────────────────────────────────────────────┘
    優點：最接近真實使用場景；缺點：較慢（尤其 LSTM）

  這是「Online Learning（線上學習）」在時序預測中的標準評估框架。

【更新策略的設計決策】

  快速模型（Bayesian, Markov, Frequency）：
    每期完整重訓（O(n) 時間，對 n=2000 期非常快）

  神經網路（MLP, LSTM）：
    使用 partial_fit()——保留現有模型參數，只做幾步梯度更新。
    這避免了「每期從頭重訓神經網路」的高昂計算成本，
    同時允許模型「持續學習」新資料。

  重訓週期選項（--retrain-every N）：
    若 N > 1，神經網路每 N 期才做一次完整重訓，
    其餘期次只做 partial_fit 微調。
    這模擬了實際部署中「定期全量重訓 + 每日增量更新」的常見模式。

使用方式：
    # 快速測試（只評估 100 期）
    python scripts/run_rolling_nn.py --test-draws 100 --no-lstm

    # 完整評估（MLP + LSTM + 其他模型）
    python scripts/run_rolling_nn.py --test-draws 200 --models mlp,lstm,bayesian,markov,frequency

    # 觀察每期詳細結果
    python scripts/run_rolling_nn.py --verbose --test-draws 50 --models mlp,bayesian
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from config.settings   import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.database.repository import DrawRepository
from src.evaluation.metrics  import (
    hit_at_k,
    calc_prize,
    expected_prize_no_mega,
    TICKET_COST,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_COLS = ["n1", "n2", "n3", "n4", "n5"]
RANDOM_EXPECTED_HITS = 5 * 5 / 47   # ≈ 0.532


# ─────────────────────────────────────────────────────────────────────────────
# 模型工廠
# ─────────────────────────────────────────────────────────────────────────────

def build_model(name: str, **kwargs):
    """根據名稱建立模型實例。"""
    name = name.lower().strip()
    if name == "mlp":
        from src.models.neural_network import MLPPredictor
        return MLPPredictor(
            epochs=kwargs.get("epochs", 80),
            fine_tune_epochs=kwargs.get("fine_tune_epochs", 5),
        )
    elif name == "lstm":
        from src.models.neural_network import LSTMPredictor
        return LSTMPredictor(
            seq_len=kwargs.get("seq_len", 20),
            epochs=kwargs.get("epochs", 60),
            fine_tune_epochs=kwargs.get("fine_tune_epochs", 5),
        )
    elif name == "bayesian":
        from src.models.bayesian import BayesianPredictor
        return BayesianPredictor()
    elif name == "markov":
        from src.models.markov import MarkovPredictor
        return MarkovPredictor()
    elif name == "frequency":
        from src.models.baseline import FrequencyBaseline
        return FrequencyBaseline()
    elif name == "knn":
        from src.models.knn_model import KNNPredictor
        return KNNPredictor(k=kwargs.get("k", 10))
    else:
        raise ValueError(f"未知模型：{name!r}")


def supports_partial_fit(model) -> bool:
    """判斷模型是否支援 partial_fit（增量學習）。"""
    return hasattr(model, "partial_fit")


# ─────────────────────────────────────────────────────────────────────────────
# Rolling 評估核心
# ─────────────────────────────────────────────────────────────────────────────

def rolling_evaluate_model(
    model_name:      str,
    df:              pd.DataFrame,
    initial_train_n: int,
    test_draws:      int,
    retrain_every:   int  = 1,
    verbose:         bool = False,
    **model_kwargs,
) -> dict:
    """
    對單一模型執行 True Walk-Forward 逐期評估。

    流程：
      ① 初始訓練：使用前 initial_train_n 期訓練模型
      ② 對接下來每一期（最多 test_draws 期）：
         a) 預測下一期（5顆球）
         b) 看實際開獎
         c) 記錄命中數
         d) 更新模型（partial_fit 或 全量重訓）
      ③ 彙整統計

    Args:
        model_name:      模型名稱
        df:              完整開獎資料（已按日期升序）
        initial_train_n: 初始訓練期數
        test_draws:      評估期數
        retrain_every:   每幾期做一次完整重訓（只適用於 NN）
                         1 = 每期完整重訓（最準確，最慢）
                         N>1 = 其餘期次用 partial_fit（折中方案）
        verbose:         是否逐期印出結果
        **model_kwargs:  傳入 build_model 的額外參數

    Returns:
        dict 包含：hits_list, predictions, dates, mean_hits, std_hits, elapsed_sec
    """
    hits_list:       List[int]   = []
    prizes_list:     List[float] = []   # 每期實際獲得的獎金
    predictions_log: List[dict]  = []
    model            = None
    t_start          = time.time()
    n_total          = min(initial_train_n + test_draws, len(df))

    logger.info(f"\n[{model_name}] 開始 Rolling 評估")
    logger.info(f"  初始訓練：{initial_train_n} 期，評估：{test_draws} 期")

    for i in range(initial_train_n, n_total):
        train_df   = df.iloc[:i]
        actual_row = df.iloc[i]
        actual_balls = [int(actual_row[c]) for c in WHITE_COLS]

        # ── 初次訓練 or 週期性完整重訓 ────────────────────────────────────
        draw_in_test = i - initial_train_n + 1
        needs_full_retrain = (
            model is None
            or (retrain_every > 1 and draw_in_test % retrain_every == 1)
            or (retrain_every == 1)   # 每期完整重訓（快速模型用）
        )

        if needs_full_retrain:
            try:
                model = build_model(model_name, **model_kwargs)
                model.fit(train_df)
            except Exception as e:
                logger.warning(f"  [{model_name}] 第 {i} 期訓練失敗：{e}")
                hits_list.append(0)
                prizes_list.append(0.0)
                continue
        elif supports_partial_fit(model):
            # ── 增量更新（partial_fit）────────────────────────────────────
            try:
                model.partial_fit(train_df)
            except Exception as e:
                logger.debug(f"  [{model_name}] partial_fit 失敗，跳過：{e}")

        # ── 預測 ────────────────────────────────────────────────────────
        try:
            result           = model.predict(n_white=5)
            predicted_balls  = result.get("white_balls", [])
            proba_map        = result.get("proba", {})
            mega_list        = result.get("mega_balls", [])
            predicted_mega   = int(mega_list[0]) if mega_list else None
        except Exception as e:
            logger.warning(f"  [{model_name}] 第 {i} 期預測失敗：{e}")
            hits_list.append(0)
            prizes_list.append(0.0)
            continue

        # ── 計算命中數 & 獎金 EV ─────────────────────────────────────────
        hits         = hit_at_k(predicted_balls, actual_balls, k=5)
        actual_mega  = int(actual_row["mega_number"]) if "mega_number" in actual_row.index else None

        if predicted_mega is not None and actual_mega is not None:
            mega_hit = (predicted_mega == actual_mega)
            prize    = calc_prize(hits, mega_hit)
        else:
            # 模型未預測 Mega：用期望值（隨機猜 1/27 命中 Mega 的加權）
            prize    = expected_prize_no_mega(hits)

        hits_list.append(hits)
        prizes_list.append(prize)

        predictions_log.append({
            "draw_idx":    i,
            "draw_date":   str(actual_row.get("draw_date", "")),
            "actual":      sorted(actual_balls),
            "predicted":   sorted(predicted_balls),
            "hits":        hits,
            "prize":       prize,
        })

        if verbose:
            hit_marker = "*" * hits if hits > 0 else "."
            print(
                f"  [{model_name}] #{i:4d} {actual_row.get('draw_date', '')} | "
                f"Pred:{sorted(predicted_balls)} | "
                f"Actual:{sorted(actual_balls)} | "
                f"Hits:{hits} {hit_marker} | Prize:${prize:.2f}"
            )

        # 每 50 期顯示進度
        if (draw_in_test) % 50 == 0:
            running_avg = np.mean(hits_list)
            logger.info(
                f"  [{model_name}] 已評估 {draw_in_test} 期，"
                f"目前 Avg Hits={running_avg:.3f}（隨機基準≈{RANDOM_EXPECTED_HITS:.3f}）"
            )

    elapsed  = time.time() - t_start
    mean_h   = float(np.mean(hits_list))   if hits_list   else 0.0
    std_h    = float(np.std(hits_list))    if hits_list   else 0.0
    ev_draw  = float(np.mean(prizes_list)) if prizes_list else 0.0
    net_ev   = ev_draw - TICKET_COST
    roi_pct  = net_ev / TICKET_COST * 100

    logger.info(
        f"[{model_name}] 完成 {len(hits_list)} 期評估 | "
        f"Avg Hits={mean_h:.4f} | EV/期=${ev_draw:.4f} | "
        f"淨EV=${net_ev:+.4f} | 耗時={elapsed:.1f}s"
    )

    return {
        "model_name":    model_name,
        "hits_list":     hits_list,
        "prizes_list":   prizes_list,
        "predictions":   predictions_log,
        "n_draws":       len(hits_list),
        "mean_hits":     mean_h,
        "std_hits":      std_h,
        "precision_at5": mean_h / 5.0,
        "coverage":      float(np.mean(np.array(hits_list) > 0)),
        "ev_per_draw":   ev_draw,
        "net_ev":        net_ev,
        "roi_pct":       roi_pct,
        "elapsed_sec":   elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 結果報表
# ─────────────────────────────────────────────────────────────────────────────

def print_rolling_report(all_results: Dict[str, dict]) -> None:
    """印出跨模型的 Rolling 評估比較報表。"""
    SEP    = "═" * 78
    rand_h = RANDOM_EXPECTED_HITS
    rand_p = 5 / 47

    print(f"\n{SEP}")
    print(" True Walk-Forward Rolling 評估結果（逐期更新）")
    print(SEP)
    print(
        f"{'模型':<14} │ {'n':>5} │ {'Avg Hits':>9} │ {'Prec@5':>7} │ "
        f"{'vs Rnd':>7} │ {'EV/期':>8} │ {'淨EV':>8} │ {'ROI%':>7}"
    )
    print("─" * 84)

    sorted_models = sorted(all_results.items(), key=lambda x: -x[1].get("mean_hits", 0))

    for name, r in sorted_models:
        delta   = r["mean_hits"] - rand_h
        delta_s = f"{delta:+.4f}"
        ev      = r.get("ev_per_draw", 0.0)
        net_ev  = r.get("net_ev",      -TICKET_COST)
        roi     = r.get("roi_pct",     -100.0)
        print(
            f"{name:<14} │ {r['n_draws']:>5} │ {r['mean_hits']:>9.4f} │ "
            f"{r['precision_at5']*100:>6.2f}% │ "
            f"{delta_s:>7} │ ${ev:>7.4f} │ ${net_ev:>+7.4f} │ {roi:>6.1f}%"
        )

    print("─" * 84)
    print(
        f"{'隨機基準':<14} │ {'—':>5} │ {rand_h:>9.4f} │ "
        f"{rand_p*100:>6.2f}% │ {'0.0000':>7} │ {'—':>8} │ {'—':>8} │"
    )
    print(SEP)

    # 命中數分布
    print(f"\n命中數分布（Hit@5 = 0~5 各佔多少比例）")
    print("─" * 78)
    header = f"{'模型':<14} │ " + " │ ".join(f" {i}中 " for i in range(6))
    print(header)
    print("─" * 78)
    for name, r in sorted_models:
        hits_arr = np.array(r["hits_list"])
        dist     = [f"{np.mean(hits_arr==i)*100:5.1f}%" for i in range(6)]
        print(f"{name:<14} │ " + " │ ".join(dist))
    print(SEP)

    # 時序分析：命中數隨時間的變化（每 50 期的移動平均）
    print(f"\n移動平均命中數（每 50 期）— 觀察 Concept Drift")
    print("─" * 78)
    for name, r in sorted_models:
        hits_arr = np.array(r["hits_list"])
        window   = 50
        if len(hits_arr) < window:
            continue
        rolling_avg = [
            np.mean(hits_arr[max(0, i-window):i+1])
            for i in range(len(hits_arr))
        ]
        # 只顯示每 50 期的節點
        checkpoints = list(range(window-1, len(hits_arr), window))
        ckpt_str    = "  ".join(
            f"@{cp+1}:{rolling_avg[cp]:.3f}" for cp in checkpoints
        )
        print(f"  {name:<14}: {ckpt_str}")

    print(SEP)
    print("  * 命中 1 顆以上")
    print(f"  隨機基準 Avg Hits ~= {rand_h:.4f}（E[Hit@5] = 5 x 5/47）")
    print("  注意：樣本量有限，與隨機的差異可能只是統計噪音")
    print(SEP)


def save_results_csv(all_results: Dict[str, dict], output_dir: Path) -> None:
    """將每個模型的逐期預測記錄匯出為 CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, r in all_results.items():
        if not r.get("predictions"):
            continue
        df = pd.DataFrame(r["predictions"])
        path = output_dir / f"rolling_{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        logger.info(f"  已匯出：{path}")


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="神經網路 + 全模型逐期 Rolling 評估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--start-date",     default="2020-01-01", help="使用此日期後的資料")
    parser.add_argument("--initial-train",  type=int, default=200, help="初始訓練期數（預設 200）")
    parser.add_argument("--test-draws",     type=int, default=150, help="評估期數（預設 150）")
    parser.add_argument("--retrain-every",  type=int, default=10,
                        help="NN 每幾期完整重訓一次（其餘用 partial_fit，預設 10）")
    parser.add_argument("--models",
                        default="mlp,lstm,bayesian,markov,frequency",
                        help="逗號分隔的模型名稱")
    parser.add_argument("--no-lstm",        action="store_true", help="跳過 LSTM（無 PyTorch 時使用）")
    parser.add_argument("--verbose",        action="store_true", help="逐期印出預測結果")
    parser.add_argument("--export-csv",     action="store_true", help="將預測記錄匯出為 CSV")
    parser.add_argument("--lstm-epochs",    type=int, default=60,  help="LSTM 初始訓練 epochs")
    parser.add_argument("--mlp-epochs",     type=int, default=80,  help="MLP 初始訓練 epochs")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DB_PATH.exists():
        logger.error("找不到資料庫！請先執行：python scripts/init_db.py")
        sys.exit(1)

    # ── 1. 讀取並篩選資料 ────────────────────────────────────────────────────
    repo = DrawRepository(DB_PATH)
    df   = repo.get_all_draws()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df = df[df["draw_date"] >= pd.to_datetime(args.start_date)].copy()
    df = df.sort_values("draw_date").reset_index(drop=True)

    logger.info(f"使用資料：{args.start_date} 之後，共 {len(df)} 期")

    min_required = args.initial_train + 10
    if len(df) < min_required:
        logger.error(f"資料量不足（{len(df)} 期 < {min_required}）")
        sys.exit(1)

    # ── 2. 解析模型列表 ───────────────────────────────────────────────────────
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.no_lstm and "lstm" in model_names:
        model_names.remove("lstm")
        logger.info("已跳過 LSTM 模型")

    actual_test_draws = min(args.test_draws, len(df) - args.initial_train)
    logger.info(f"評估設定：初始訓練 {args.initial_train} 期，評估 {actual_test_draws} 期")
    logger.info(f"模型列表：{model_names}")
    logger.info(f"NN 每 {args.retrain_every} 期完整重訓，其餘用 partial_fit")

    # ── 3. 逐模型評估 ─────────────────────────────────────────────────────────
    all_results: Dict[str, dict] = {}

    for model_name in model_names:
        print(f"\n{'─'*60}")
        print(f" Rolling 評估：{model_name.upper()}")
        print(f"{'─'*60}")

        # 快速模型（非 NN）每期完整重訓，NN 使用 retrain_every
        is_nn    = model_name in ("mlp", "lstm")
        r_every  = args.retrain_every if is_nn else 1

        model_kwargs = {}
        if model_name == "mlp":
            model_kwargs["epochs"] = args.mlp_epochs
        elif model_name == "lstm":
            model_kwargs["epochs"] = args.lstm_epochs

        try:
            result = rolling_evaluate_model(
                model_name      = model_name,
                df              = df,
                initial_train_n = args.initial_train,
                test_draws      = actual_test_draws,
                retrain_every   = r_every,
                verbose         = args.verbose,
                **model_kwargs,
            )
            all_results[model_name.upper()] = result
        except ImportError as e:
            logger.warning(f"[{model_name}] 缺少依賴，跳過：{e}")
        except Exception as e:
            logger.warning(f"[{model_name}] 評估失敗：{e}")

    # ── 4. 輸出比較報表 ───────────────────────────────────────────────────────
    if not all_results:
        logger.error("沒有任何模型完成評估")
        sys.exit(1)

    print_rolling_report(all_results)

    # ── 5. 匯出 CSV（選用）───────────────────────────────────────────────────
    if args.export_csv:
        from config.settings import EXPORTS_DIR
        save_results_csv(all_results, EXPORTS_DIR / "rolling_eval")
        print(f"\n  CSV 已匯出至：{EXPORTS_DIR / 'rolling_eval'}")


if __name__ == "__main__":
    main()
