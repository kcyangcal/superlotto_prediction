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
from itertools import combinations as itertools_combinations
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

MAX_TICKETS = 5   # 最多同時評估幾組組合策略

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_COLS = ["n1", "n2", "n3", "n4", "n5"]
RANDOM_EXPECTED_HITS = 5 * 5 / 47   # ≈ 0.532


# ─────────────────────────────────────────────────────────────────────────────
# 模型工廠
# ─────────────────────────────────────────────────────────────────────────────

# 各模型預設 retrain_every（每幾期重訓一次；1 = 每期）
# NN 模型由 --retrain-every 控制，其餘依速度設定
MODEL_RETRAIN_DEFAULTS = {
    # 快速模型：每期重訓
    "bayesian":     1,
    "markov":       1,
    "frequency":    1,
    "knn":          1,
    "decision_tree":1,
    # 中速模型：每 3 期重訓
    "rf":           3,
    "xgb":          3,
    "ensemble":     30,  # 每 30 期重訓（ensemble 內含 MC+GA，重訓成本高）
    # 慢速模型：每 5 期重訓
    "montecarlo":   5,
    # 非常慢：每 10 期重訓
    "genetic":      10,
    # NN 由 CLI --retrain-every 控制
    "mlp":          10,
    "lstm":         10,
}

# 所有支援的模型列表（--models all 時使用）
ALL_MODEL_NAMES = [
    "frequency", "bayesian", "markov", "knn",
    "decision_tree", "montecarlo", "genetic",
    "rf", "xgb", "ensemble",
    "mlp", "lstm",
]


def build_model(name: str, **kwargs):
    """根據名稱建立模型實例。"""
    name = name.lower().strip()

    # ── Neural Networks ───────────────────────────────────────────────────────
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

    # ── 統計 / 機率模型 ───────────────────────────────────────────────────────
    elif name == "bayesian":
        from src.models.bayesian import BayesianPredictor
        return BayesianPredictor()
    elif name == "markov":
        from src.models.markov import MarkovPredictor
        return MarkovPredictor()
    elif name == "frequency":
        from src.models.baseline import FrequencyBaseline
        return FrequencyBaseline()
    elif name == "montecarlo":
        from src.models.monte_carlo import MonteCarloPredictor
        return MonteCarloPredictor(
            n_simulations=kwargs.get("n_simulations", 50_000)
        )

    # ── ML 模型 ───────────────────────────────────────────────────────────────
    elif name == "knn":
        from src.models.knn_model import KNNPredictor
        return KNNPredictor(k=kwargs.get("k", 10))
    elif name == "decision_tree":
        from src.models.decision_tree import DecisionTreePredictor
        return DecisionTreePredictor()
    elif name == "genetic":
        from src.models.genetic import GeneticPredictor
        return GeneticPredictor(
            n_generations=kwargs.get("n_generations", 100),
            population_size=kwargs.get("population_size", 30),
        )

    # ── RF / XGBoost 包裝器（classifier.py 介面不標準，用薄包裝） ──────────────
    elif name in ("rf", "xgb"):
        from src.models.classifier import LotteryClassifier

        _target = name   # capture for closure

        class _ClassifierWrapper:
            """將 LotteryClassifier.predict_next() 包裝為標準 predict() 介面。"""
            def fit(self, df):
                self._clf = LotteryClassifier()
                train_xgb = (_target == "xgb")
                self._clf.fit(df, train_xgb=train_xgb)
                return self

            def predict(self, n_white: int = 5, **kw) -> dict:
                result   = self._clf.predict_next(top_k=n_white)
                key      = f"{_target}_prediction"
                pred_d   = result.get(key, {})
                balls    = pred_d.get("white_balls", [])
                proba    = pred_d.get("probabilities", {})
                return {"white_balls": balls, "mega_balls": [], "proba": proba}

        return _ClassifierWrapper()

    # ── 集成模型 ──────────────────────────────────────────────────────────────
    elif name == "ensemble":
        from src.models.ensemble import EnsemblePredictor
        return EnsemblePredictor()

    else:
        raise ValueError(
            f"未知模型：{name!r}。可用：{ALL_MODEL_NAMES}"
        )


def supports_partial_fit(model) -> bool:
    """判斷模型是否支援 partial_fit（增量學習）。"""
    return hasattr(model, "partial_fit")


def generate_top_k_tickets(
    white_proba:  dict,
    mega_proba:   dict,
    n_tickets:    int = MAX_TICKETS,
    n_candidates: int = 12,
) -> List[dict]:
    """
    從白球機率分布中生成機率最高的 n_tickets 組彩票組合。

    做法：
      1. 取 top n_candidates 個白球（降低枚舉量：C(12,5)=792）
      2. 對所有 C(n_candidates,5) 組合，以 sum(prob) 排分
      3. 取最高分的 n_tickets 組
      4. 每組配上機率最高的 Mega 球（若有 mega_proba 則確定配，否則 None）

    Args:
        white_proba:  {球號: 機率} 47 顆白球的機率 map
        mega_proba:   {球號: 機率} Mega 球機率 map（可為空 dict）
        n_tickets:    要生成幾組組合
        n_candidates: 從機率最高的幾顆球中枚舉組合（trade-off 速度 vs 覆蓋）

    Returns:
        list of {"white_balls": [int, ...], "mega_ball": int or None}
    """
    if not white_proba:
        return []

    # ── 1. 取 top n_candidates 白球候選 ─────────────────────────────────────
    sorted_balls = sorted(white_proba.keys(), key=lambda b: -white_proba[b])
    candidates   = sorted_balls[:n_candidates]

    # ── 2. 枚舉組合，按 sum(prob) 排序 ──────────────────────────────────────
    scored = [
        (sum(white_proba[b] for b in combo), sorted(combo))
        for combo in itertools_combinations(candidates, 5)
    ]
    scored.sort(reverse=True)

    # ── 3. 最可能的 Mega 球 ──────────────────────────────────────────────────
    best_mega = (
        max(mega_proba.keys(), key=lambda m: mega_proba[m])
        if mega_proba else None
    )

    # ── 4. 組裝彩票 ─────────────────────────────────────────────────────────
    return [
        {"white_balls": white_balls, "mega_ball": best_mega}
        for _, white_balls in scored[:n_tickets]
    ]


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
    n_tickets:       int  = MAX_TICKETS,
    n_candidates:    int  = 12,
    **model_kwargs,
) -> dict:
    """
    對單一模型執行 True Walk-Forward 逐期評估。

    流程：
      ① 初始訓練：使用前 initial_train_n 期訓練模型
      ② 對接下來每一期（最多 test_draws 期）：
         a) 預測下一期（最多 n_tickets 組組合）
         b) 看實際開獎
         c) 記錄各組命中數與獎金
         d) 更新模型（partial_fit 或 全量重訓）
      ③ 彙整統計（含買 1~n_tickets 組的多策略比較）

    Args:
        model_name:      模型名稱
        df:              完整開獎資料（已按日期升序）
        initial_train_n: 初始訓練期數
        test_draws:      評估期數
        retrain_every:   每幾期做一次完整重訓（只適用於 NN）
                         1 = 每期完整重訓（最準確，最慢）
                         N>1 = 其餘期次用 partial_fit（折中方案）
        verbose:         是否逐期印出結果
        n_tickets:       每期生成幾組候選組合（用於多策略評估，預設 5）
        n_candidates:    從幾顆最高機率球中枚舉組合（預設 12，C(12,5)=792）
        **model_kwargs:  傳入 build_model 的額外參數

    Returns:
        dict 包含：hits_list, predictions, dates, mean_hits, std_hits, elapsed_sec,
                   multi_ticket（買 1~n_tickets 組各策略的 EV/淨EV/ROI）
    """
    hits_list:          List[int]         = []
    prizes_list:        List[float]       = []   # 每期實際獲得的獎金（買第1組）
    multi_prizes_list:  List[List[float]] = []   # 每期各組票的獎金 [p1, p2, ..., pK]
    predictions_log:    List[dict]        = []
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
            mega_proba_map   = result.get("mega_proba", {})
            mega_list        = result.get("mega_balls", [])
            predicted_mega   = int(mega_list[0]) if mega_list else None
        except Exception as e:
            logger.warning(f"  [{model_name}] 第 {i} 期預測失敗：{e}")
            hits_list.append(0)
            prizes_list.append(0.0)
            multi_prizes_list.append([0.0] * n_tickets)
            continue

        actual_mega = int(actual_row["mega_number"]) if "mega_number" in actual_row.index else None

        # ── 生成 top-K 組合 ───────────────────────────────────────────────
        # 若有機率 map，枚舉最可能的 n_tickets 組；否則全部用第一組預測填充
        if proba_map:
            tickets = generate_top_k_tickets(
                proba_map, mega_proba_map, n_tickets, n_candidates
            )
            if not tickets:
                tickets = [{"white_balls": predicted_balls, "mega_ball": predicted_mega}]
            # 不足 n_tickets 組時，用最後一組補齊（避免 index 越界）
            while len(tickets) < n_tickets:
                tickets.append(tickets[-1])
        else:
            tickets = [{"white_balls": predicted_balls, "mega_ball": predicted_mega}] * n_tickets

        # ── 計算每張票的命中數 & 獎金 ────────────────────────────────────
        draw_prizes: List[float] = []
        for ticket in tickets:
            t_balls = ticket["white_balls"]
            t_mega  = ticket["mega_ball"]
            t_hits  = hit_at_k(t_balls, actual_balls, k=5)
            if t_mega is not None and actual_mega is not None:
                t_prize = calc_prize(t_hits, t_mega == actual_mega)
            else:
                t_prize = expected_prize_no_mega(t_hits)
            draw_prizes.append(t_prize)

        multi_prizes_list.append(draw_prizes)

        # ── 單組統計（買第 1 組，與舊邏輯相容）────────────────────────────
        hits  = hit_at_k(tickets[0]["white_balls"], actual_balls, k=5)
        prize = draw_prizes[0]
        hits_list.append(hits)
        prizes_list.append(prize)

        predictions_log.append({
            "draw_idx":    i,
            "draw_date":   str(actual_row.get("draw_date", "")),
            "actual":      sorted(actual_balls),
            "predicted":   sorted(tickets[0]["white_balls"]),
            "hits":        hits,
            "prize":       prize,
        })

        if verbose:
            hit_marker = "*" * hits if hits > 0 else "."
            print(
                f"  [{model_name}] #{i:4d} {actual_row.get('draw_date', '')} | "
                f"Pred1:{sorted(tickets[0]['white_balls'])} | "
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

    # ── 多組策略彙整：買 1~n_tickets 組的 EV/淨EV/ROI ──────────────────────
    multi_ticket: Dict[int, dict] = {}
    if multi_prizes_list:
        actual_k = len(multi_prizes_list[0])   # 實際生成的票數
        for buy_n in range(1, actual_k + 1):
            per_draw_ev = [sum(dp[:buy_n]) for dp in multi_prizes_list]
            ev_n    = float(np.mean(per_draw_ev))
            cost_n  = buy_n * TICKET_COST
            net_n   = ev_n - cost_n
            roi_n   = net_n / cost_n * 100
            multi_ticket[buy_n] = {
                "ev_per_draw":   ev_n,
                "cost_per_draw": cost_n,
                "net_ev":        net_n,
                "roi_pct":       roi_n,
            }

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
        "multi_ticket":  multi_ticket,
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

    # ── Multi-Ticket Strategy 比較表 ────────────────────────────────────────
    _print_multi_ticket_table(sorted_models)


def _print_multi_ticket_table(sorted_models: list) -> None:
    """
    印出「每期買 N 組最可能組合」的 EV/淨EV/ROI 比較表。

    表格格式：
      模型         │ 買1組              │ 買2組              │ ... │ 買5組
                   │ EV    淨EV   ROI%  │ EV    淨EV   ROI%  │ ...
    """
    # 確認有 multi_ticket 資料
    has_multi = any(r.get("multi_ticket") for _, r in sorted_models)
    if not has_multi:
        return

    # 取最大票數
    max_k = max(
        (max(r["multi_ticket"].keys()) for _, r in sorted_models if r.get("multi_ticket")),
        default=0
    )
    if max_k == 0:
        return

    col_w   = 24   # 每個「買N組」欄位寬度
    name_w  = 14
    SEP2    = "═" * (name_w + 3 + max_k * (col_w + 3))

    print(f"\n{SEP2}")
    print(f" Multi-Ticket Strategy 比較（每期買 N 組最可能組合）")
    print(f" 成本 = N × $1；獎金 = 各組票獎金加總；淨EV = 總獎金 − 總成本")
    print(SEP2)

    # 標題行
    header_cols = " │ ".join(f"{'買'+str(n)+'組':^{col_w}}" for n in range(1, max_k + 1))
    print(f"{'模型':<{name_w}} │ {header_cols}")

    sub_header  = " │ ".join(
        f"{'EV':>7}  {'淨EV':>7}  {'ROI%':>6}" for _ in range(max_k)
    )
    print(f"{'':<{name_w}} │ {sub_header}")
    print("─" * (name_w + 3 + max_k * (col_w + 3)))

    for name, r in sorted_models:
        mt = r.get("multi_ticket", {})
        if not mt:
            continue
        cols = []
        for n in range(1, max_k + 1):
            s = mt.get(n, {})
            ev  = s.get("ev_per_draw",   0.0)
            nev = s.get("net_ev",        0.0)
            roi = s.get("roi_pct",     -100.0)
            cols.append(f"${ev:>5.3f}  ${nev:>+6.3f}  {roi:>5.1f}%")
        print(f"{name:<{name_w}} │ " + " │ ".join(cols))

    print("─" * (name_w + 3 + max_k * (col_w + 3)))
    print("  注意：票組間可能有重複號碼；買多組不代表覆蓋更多不同號碼")
    print(f"{SEP2}")


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


def save_summary_csv(all_results: Dict[str, dict], output_path: Path) -> None:
    """
    將所有模型的 summary 指標（含 multi-ticket 策略）存為一個 CSV，
    方便跨次執行比較。
    """
    import csv, datetime
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    run_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    for name, r in all_results.items():
        n        = len(r.get("hits_list", []))
        avg_hits = r.get("mean_hits", 0.0)
        prec5    = r.get("precision_at5", 0.0)
        vs_rnd   = avg_hits - (5 * 5 / 47)
        ev       = r.get("ev_per_draw", 0.0)
        net_ev   = r.get("net_ev", 0.0)
        roi      = r.get("roi_pct", -100.0)
        elapsed  = r.get("elapsed_sec", 0.0)

        base_row = {
            "run_time":   run_ts,
            "model":      name,
            "n_draws":    n,
            "avg_hits":   round(avg_hits, 4),
            "prec_at5":   round(prec5, 4),
            "vs_random":  round(vs_rnd, 4),
            "ev_per_draw": round(ev, 4),
            "net_ev":     round(net_ev, 4),
            "roi_pct":    round(roi, 2),
            "elapsed_sec": round(elapsed, 1),
        }

        # multi-ticket 欄位：buy_1 到 buy_5
        mt = r.get("multi_ticket", {})
        for k in range(1, 6):
            s = mt.get(k, {})
            base_row[f"buy{k}_ev"]     = round(s.get("ev_per_draw", 0.0), 4)
            base_row[f"buy{k}_net_ev"] = round(s.get("net_ev", 0.0), 4)
            base_row[f"buy{k}_roi"]    = round(s.get("roi_pct", -100.0), 2)

        rows.append(base_row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    file_exists = output_path.exists()

    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Summary CSV 已儲存/追加：{output_path}（{len(rows)} 行）")


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
                        default="frequency,bayesian,markov,montecarlo,knn,decision_tree,genetic,rf,xgb,ensemble",
                        help=f"逗號分隔的模型名稱，或 'all'。可用：{ALL_MODEL_NAMES}")
    parser.add_argument("--no-lstm",        action="store_true", help="跳過 LSTM（無 PyTorch 時使用）")
    parser.add_argument("--verbose",        action="store_true", help="逐期印出預測結果")
    parser.add_argument("--export-csv",     action="store_true", help="將預測記錄匯出為 CSV")
    parser.add_argument("--lstm-epochs",    type=int, default=60,  help="LSTM 初始訓練 epochs")
    parser.add_argument("--mlp-epochs",     type=int, default=80,  help="MLP 初始訓練 epochs")
    parser.add_argument("--n-tickets",      type=int, default=MAX_TICKETS,
                        help=f"每期生成幾組候選組合（multi-ticket 策略，預設 {MAX_TICKETS}）")
    parser.add_argument("--n-candidates",   type=int, default=12,
                        help="枚舉組合時使用前幾顆高機率球（預設 12，C(12,5)=792）")
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
    raw_models  = args.models.strip().lower()
    if raw_models == "all":
        model_names = [m for m in ALL_MODEL_NAMES]
    else:
        model_names = [m.strip() for m in raw_models.split(",") if m.strip()]

    if args.no_lstm and "lstm" in model_names:
        model_names.remove("lstm")
        logger.info("已跳過 LSTM 模型")

    actual_test_draws = min(args.test_draws, len(df) - args.initial_train)
    logger.info(f"評估設定：初始訓練 {args.initial_train} 期，評估 {actual_test_draws} 期")
    logger.info(f"模型列表：{model_names}")
    logger.info(f"NN 每 {args.retrain_every} 期完整重訓，慢速模型依預設週期重訓")

    # ── 3. 逐模型評估 ─────────────────────────────────────────────────────────
    all_results: Dict[str, dict] = {}

    for model_name in model_names:
        print(f"\n{'─'*60}")
        print(f" Rolling 評估：{model_name.upper()}")
        print(f"{'─'*60}")

        # 決定 retrain_every：NN 用 CLI 參數，其他依 MODEL_RETRAIN_DEFAULTS
        is_nn   = model_name in ("mlp", "lstm")
        r_every = args.retrain_every if is_nn else MODEL_RETRAIN_DEFAULTS.get(model_name, 1)
        logger.info(f"  [{model_name}] retrain_every={r_every}")

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
                n_tickets       = args.n_tickets,
                n_candidates    = args.n_candidates,
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

    # ── 5. 自動存 summary CSV ─────────────────────────────────────────────────
    summary_path = ROOT_DIR / "results" / "rolling_eval_summary.csv"
    save_summary_csv(all_results, summary_path)
    print(f"\n  Summary CSV 已儲存：{summary_path}")

    # ── 6. 匯出逐期預測 CSV（選用）──────────────────────────────────────────
    if args.export_csv:
        from config.settings import EXPORTS_DIR
        save_results_csv(all_results, EXPORTS_DIR / "rolling_eval")
        print(f"\n  逐期 CSV 已匯出至：{EXPORTS_DIR / 'rolling_eval'}")


if __name__ == "__main__":
    main()
