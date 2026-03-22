"""
feature_analysis.py — 特徵信號探索分析
=========================================
對每個候選特徵計算與「下期是否出現」的相關性，
找出哪些特徵有統計預測力。

分析方法：
  Point-Biserial Correlation（連續特徵 vs 二元標籤）
  適合用來快速篩選特徵，不保證在模型中一定有效。

輸出：
  1. 特徵重要性排名表（相關係數 + p 值）
  2. 各特徵的分組統計（出現 vs 未出現時的平均值）
  3. 儲存到 results/feature_analysis.csv

用法：
    python scripts/feature_analysis.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

from src.database.repository import DrawRepository
from config.settings import WHITE_BALL_MIN, WHITE_BALL_MAX

DB_PATH      = ROOT_DIR / "data" / "superlotto.db"
WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))
WHITE_COLS    = ["n1", "n2", "n3", "n4", "n5"]
WARM_UP       = 50   # 最少需要 50 期才開始建樣本


# ─────────────────────────────────────────────────────────────────────────────
# 特徵計算
# ─────────────────────────────────────────────────────────────────────────────

def build_features_at(df: pd.DataFrame, i: int) -> pd.DataFrame:
    """
    以截至第 i-1 期（不含第 i 期）的歷史，為 47 顆球各自建立一列特徵。
    標籤 y = 第 i 期這顆球是否出現。

    回傳 DataFrame，shape=(47, n_features + 2)，含 ball, y 欄位。
    """
    history = df.iloc[:i]          # 截至上期的所有資料
    last    = df.iloc[i - 1]       # 最近一期
    prev2   = df.iloc[i - 2] if i >= 2 else None   # 前兩期
    actual  = df.iloc[i]           # 本期（用來建標籤）

    n_total = len(history)

    # ── 全史頻率與間距 ─────────────────────────────────────────────────────
    # last_seen[n] = 最後出現期的 index (0-based)
    last_seen = {}
    for idx in range(n_total):
        for c in WHITE_COLS:
            last_seen[int(history.iloc[idx][c])] = idx

    # 各號碼 gap 列表（用來算 avg_gap）
    gap_lists = {n: [] for n in WHITE_NUMBERS}
    ls_tmp = {}
    for idx in range(n_total):
        for c in WHITE_COLS:
            n = int(history.iloc[idx][c])
            if n in ls_tmp:
                gap_lists[n].append(idx - ls_tmp[n])
            ls_tmp[n] = idx

    avg_gap = {n: np.mean(gap_lists[n]) if gap_lists[n] else n_total
               for n in WHITE_NUMBERS}
    current_gap = {n: (n_total - 1) - last_seen.get(n, -1)
                   for n in WHITE_NUMBERS}

    # ── 多窗口頻率 ────────────────────────────────────────────────────────
    def freq_in_window(w):
        win = history.tail(w)
        counts = {}
        for c in WHITE_COLS:
            for n in win[c]:
                counts[int(n)] = counts.get(int(n), 0) + 1
        return {n: counts.get(n, 0) / w for n in WHITE_NUMBERS}

    freq5   = freq_in_window(5)
    freq10  = freq_in_window(10)
    freq30  = freq_in_window(30)
    freq100 = freq_in_window(min(100, n_total))

    # ── 共現：上期球 vs 候選球的共現頻率 ────────────────────────────────
    last_balls = {int(last[c]) for c in WHITE_COLS}
    cooccur_with_last = {n: 0.0 for n in WHITE_NUMBERS}
    if len(history) >= 2:
        comat = np.zeros((48, 48), dtype=np.int32)
        for idx in range(n_total):
            balls = [int(history.iloc[idx][c]) for c in WHITE_COLS]
            for a, b in combinations(balls, 2):
                comat[a][b] += 1
                comat[b][a] += 1
        total_draws = n_total
        for n in WHITE_NUMBERS:
            score = sum(comat[n][lb] for lb in last_balls if lb != n)
            cooccur_with_last[n] = score / total_draws

    # ── 上期特徵（整期統計） ─────────────────────────────────────────────
    last_balls_list = [int(last[c]) for c in WHITE_COLS]
    last_sum        = sum(last_balls_list)
    last_odd_cnt    = sum(1 for b in last_balls_list if b % 2 == 1)
    last_high_cnt   = sum(1 for b in last_balls_list if b > 23)
    last_consec     = sum(1 for a, b in zip(sorted(last_balls_list),
                                             sorted(last_balls_list)[1:])
                          if b - a == 1)

    # ── 本期標籤 ─────────────────────────────────────────────────────────
    actual_balls = {int(actual[c]) for c in WHITE_COLS}

    rows = []
    for n in WHITE_NUMBERS:
        cg  = current_gap[n]
        ag  = avg_gap[n]
        rows.append({
            "ball": n,
            "y":    int(n in actual_balls),

            # Gap 特徵
            "current_gap":      cg,
            "avg_gap":          ag,
            "gap_ratio":        cg / ag if ag > 0 else 1.0,   # 超期比例（>1 = 過期）
            "gap_zscore":       (cg - ag) / max(np.std(gap_lists[n]), 1)
                                if gap_lists[n] else 0.0,

            # 頻率特徵
            "freq_5":           freq5[n],
            "freq_10":          freq10[n],
            "freq_30":          freq30[n],
            "freq_100":         freq100[n],
            "freq_ratio_5_30":  freq5[n] / max(freq30[n], 1e-6),   # 近熱 vs 中熱

            # 重複出現
            "appeared_last":    int(n in last_balls),
            "appeared_prev2":   int(prev2 is not None and
                                    n in {int(prev2[c]) for c in WHITE_COLS}),

            # 共現（與上期球的歷史共現頻率）
            "cooccur_last":     cooccur_with_last[n],

            # 上期整體統計（對每顆球相同，但模型可學全局上下文）
            "last_sum":         last_sum,
            "last_odd_cnt":     last_odd_cnt,
            "last_high_cnt":    last_high_cnt,
            "last_consec":      last_consec,

            # 球本身屬性
            "ball_is_odd":      int(n % 2 == 1),
            "ball_zone":        (n - 1) // 10,   # 0=1-10, 1=11-20, 2=21-30, 3=31-40, 4=41-47
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=== SuperLotto 特徵信號分析 ===\n")

    repo = DrawRepository(DB_PATH)
    df   = repo.get_all_draws()
    df["draw_date"] = pd.to_datetime(df["draw_date"])
    df = df[df["draw_date"] >= "2020-01-01"].sort_values("draw_date").reset_index(drop=True)
    print(f"資料：{len(df)} 期（2020-01-01 之後）\n")

    print(f"建立特徵矩陣（每期47顆球，共 {len(df) - WARM_UP} 期）...")
    print("（這需要幾分鐘，共現矩陣計算較慢）\n")

    all_rows = []
    step = max(1, (len(df) - WARM_UP) // 20)
    for i in range(WARM_UP, len(df) - 1):
        if (i - WARM_UP) % step == 0:
            pct = (i - WARM_UP) / (len(df) - 1 - WARM_UP) * 100
            print(f"  {pct:.0f}%...", end="\r", flush=True)
        rows = build_features_at(df, i)
        rows["draw_idx"] = i
        all_rows.append(rows)

    data = pd.concat(all_rows, ignore_index=True)
    print(f"\n特徵矩陣完成：{len(data):,} 行 × {len(data.columns)} 欄\n")

    # ── Point-Biserial Correlation ────────────────────────────────────────
    feature_cols = [c for c in data.columns if c not in ("ball", "y", "draw_idx")]
    y = data["y"].values

    results = []
    for feat in feature_cols:
        x = data[feat].values.astype(float)
        # 移除 NaN
        mask = ~np.isnan(x)
        corr, pval = stats.pointbiserialr(x[mask], y[mask])
        mean_pos = x[mask & (y == 1)].mean()   # 出現時的平均值
        mean_neg = x[mask & (y == 0)].mean()   # 未出現時的平均值
        results.append({
            "feature":    feat,
            "corr":       corr,
            "abs_corr":   abs(corr),
            "p_value":    pval,
            "mean_appear":  round(mean_pos, 4),
            "mean_not_app": round(mean_neg, 4),
            "delta":      round(mean_pos - mean_neg, 4),
            "sig":        "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "")),
        })

    res_df = pd.DataFrame(results).sort_values("abs_corr", ascending=False).reset_index(drop=True)

    # ── 輸出結果 ──────────────────────────────────────────────────────────
    print("=" * 80)
    print(" 特徵預測力排名（Point-Biserial Correlation vs 下期是否出現）")
    print("=" * 80)
    print(f"{'特徵':<22} {'Corr':>8} {'|Corr|':>8} {'p-value':>12} {'sig':>4} │ {'出現均值':>10} {'未出現':>10} {'差值':>8}")
    print("─" * 88)
    for _, row in res_df.iterrows():
        print(f"{row['feature']:<22} {row['corr']:>+8.4f} {row['abs_corr']:>8.4f} "
              f"{row['p_value']:>12.2e} {row['sig']:>4} │ "
              f"{row['mean_appear']:>10.4f} {row['mean_not_app']:>10.4f} {row['delta']:>+8.4f}")
    print("─" * 88)
    print("  sig: *** p<0.001  ** p<0.01  * p<0.05\n")

    # ── 分析結論 ─────────────────────────────────────────────────────────
    sig_feats = res_df[res_df["p_value"] < 0.05]
    print(f"顯著特徵（p<0.05）：{len(sig_feats)}/{len(res_df)} 個")
    print(f"最強正相關：{res_df[res_df['corr']>0].iloc[0]['feature']} (r={res_df[res_df['corr']>0].iloc[0]['corr']:+.4f})")
    print(f"最強負相關：{res_df[res_df['corr']<0].iloc[0]['feature']} (r={res_df[res_df['corr']<0].iloc[0]['corr']:+.4f})\n")

    # ── 儲存 ─────────────────────────────────────────────────────────────
    out = ROOT_DIR / "results" / "feature_analysis.csv"
    res_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"結果已存至：{out}")

    # ── appeared_last 的詳細分布 ─────────────────────────────────────────
    print("\n─── appeared_last（上期出現）對下期出現率的影響 ───")
    grp = data.groupby("appeared_last")["y"].agg(["mean", "count"])
    grp.index = ["未在上期出現", "在上期出現"]
    grp.columns = ["下期出現率", "樣本數"]
    print(grp.to_string())

    print("\n─── gap_ratio 分組（1=已到期, >1=超期）vs 下期出現率 ───")
    data["gap_ratio_bin"] = pd.cut(data["gap_ratio"],
                                    bins=[0, 0.5, 1.0, 1.5, 2.0, 99],
                                    labels=["<0.5x", "0.5-1x", "1-1.5x", "1.5-2x", ">2x"])
    grp2 = data.groupby("gap_ratio_bin", observed=True)["y"].agg(["mean", "count"])
    grp2.columns = ["下期出現率", "樣本數"]
    print(grp2.to_string())

    print("\n─── freq_ratio_5_30（近期熱度 / 中期熱度）vs 下期出現率 ───")
    data["freq_ratio_bin"] = pd.cut(data["freq_ratio_5_30"],
                                     bins=[0, 0.5, 1.0, 2.0, 5.0, 99],
                                     labels=["冷<0.5x", "0.5-1x", "1-2x", "2-5x", "極熱>5x"])
    grp3 = data.groupby("freq_ratio_bin", observed=True)["y"].agg(["mean", "count"])
    grp3.columns = ["下期出現率", "樣本數"]
    print(grp3.to_string())


if __name__ == "__main__":
    main()
