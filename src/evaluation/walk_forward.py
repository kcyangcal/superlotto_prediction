"""
walk_forward.py — Rolling Hold-Out (Walk-Forward) 時序切分器
==============================================================

【為什麼時間序列「不能」隨機切分？】

  一般的 train_test_split(shuffle=True) 做法：
  ┌─────────────────────────────────────────┐
  │  資料點:  A B C D E F G H I J          │
  │  Train:   A   C D   F G   I J  (random) │
  │  Test:      B       E   H              │
  └─────────────────────────────────────────┘
  問題：資料點 C（2023年）出現在訓練集，資料點 B（2022年）卻在測試集。
  在訓練時，模型「看到了未來的資料」→ Data Leakage（資料洩漏）。
  這種情況下計算出的準確率是「虛假的高」，完全無法反映真實預測能力。

  正確的時序切分：
  ┌─────────────────────────────────────────┐
  │  資料點:  A B C D E F G H I J          │
  │  Train:   A B C D    (過去)             │
  │  Test:            E F G H I J  (未來)   │
  └─────────────────────────────────────────┘
  保證訓練集的所有資料都「早於」測試集的所有資料。

【Walk-Forward Validation 的進階版本】

  比靜態一刀切更好的做法：訓練視窗隨著時間「滾動前進」。
  這模擬了真實使用場景——每隔半年重新訓練模型後再預測。

  Fold 1: [──────Train──────][─Test─]
  Fold 2: [────────Train────────][─Test─]
  Fold 3: [──────────Train──────────][─Test─]
  Fold 4: [────────────Train────────────][─Test─]

  優點：
    • 測試集永遠是「未見過的未來資料」
    • 可以觀察模型準確率是否隨時間退化（Concept Drift）
    • 比單一靜態切分更能反映模型的真實泛化能力

【本實作的設計決策】

  使用「擴展視窗 (Expanding Window)」而非「滑動視窗 (Sliding Window)」：
  - 擴展視窗：訓練集起點固定，終點不斷後移（累積更多歷史資料）
  - 滑動視窗：訓練集長度固定，整體向後滑動（只用最近 N 期）

  對彩票這類「可能存在長期趨勢但難以確定的資料」，
  擴展視窗保留了更多歷史資訊，是更保守、更合理的選擇。
"""

import logging
from dataclasses import dataclass
from typing import Generator, List

import pandas as pd

from config.settings import LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 資料結構
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FoldInfo:
    """
    單一 Fold 的元資訊，用於報表輸出。

    屬性：
        fold_idx:      Fold 編號（從 1 開始）
        train_start:   訓練集起始日期
        train_end:     訓練集結束日期
        test_start:    測試集起始日期
        test_end:      測試集結束日期
        n_train:       訓練樣本數（期次數）
        n_test:        測試樣本數（期次數）
    """
    fold_idx:    int
    train_start: str
    train_end:   str
    test_start:  str
    test_end:    str
    n_train:     int
    n_test:      int

    def __str__(self) -> str:
        return (
            f"Fold {self.fold_idx:2d} | "
            f"Train: {self.train_start} → {self.train_end} ({self.n_train:4d} 期) | "
            f"Test: {self.test_start} → {self.test_end} ({self.n_test:3d} 期)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward 切分器
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardSplit:
    """
    基於日期的滾動驗證切分器（擴展視窗版本）。

    設計原則：
      • 以「月份」為單位切分，而非以「期數」切分，
        因為開獎頻率可能因年份不同而有微小差異（停辦期、假日）。
      • 每個 Fold 的測試集長度相同（test_months），
        但訓練集長度逐 Fold 增長（擴展視窗）。
      • 訓練集與測試集之間沒有重疊，且訓練集永遠早於測試集。

    使用範例：
        splitter = WalkForwardSplit(
            train_months=24,   # 初始訓練期：2 年
            test_months=6,     # 每個測試視窗：半年
            step_months=6,     # 每次向前推進：半年
        )
        for train_df, test_df, fold_info in splitter.split(df):
            print(fold_info)
            model.fit(train_df)
            results = evaluate(model, test_df)
    """

    def __init__(
        self,
        train_months: int = 24,
        test_months:  int = 6,
        step_months:  int = 6,
        date_col:     str = "draw_date",
    ):
        """
        Args:
            train_months: 初始訓練視窗大小（月）
            test_months:  每個 Fold 的測試視窗大小（月）
            step_months:  每個 Fold 之間的前進步長（月）
                          通常 = test_months（非重疊）
                          若 < test_months → 測試集之間有重疊（較少見）
            date_col:     DataFrame 中存放日期的欄位名稱
        """
        if train_months <= 0 or test_months <= 0 or step_months <= 0:
            raise ValueError("所有月份參數必須為正整數")

        self.train_months = train_months
        self.test_months  = test_months
        self.step_months  = step_months
        self.date_col     = date_col

    def split(
        self,
        df: pd.DataFrame,
    ) -> Generator[tuple, None, None]:
        """
        對 DataFrame 進行時序切分，以 Generator 方式逐一產生 Fold。

        Args:
            df: 開獎資料，必須包含 date_col 欄位（字串或 datetime 均可）

        Yields:
            (train_df, test_df, fold_info)
            - train_df:  訓練集 DataFrame（時間較早）
            - test_df:   測試集 DataFrame（時間較晚，對模型完全未知）
            - fold_info: FoldInfo 資料物件，含日期範圍與樣本數

        Raises:
            ValueError: 若資料不足以產生任何 Fold
        """
        # 確保日期欄位為 datetime 型別
        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.sort_values(self.date_col).reset_index(drop=True)

        # ── 計算整體資料的日期範圍 ──────────────────────────────────────────
        data_start = df[self.date_col].min()
        data_end   = df[self.date_col].max()

        # 訓練視窗的第一個結束日期
        # 例如：data_start=2020-01，train_months=24 → 第一個 train_end=2021-12
        first_train_end = data_start + pd.DateOffset(months=self.train_months)

        # 檢查是否有足夠資料
        min_required_end = first_train_end + pd.DateOffset(months=self.test_months)
        if min_required_end > data_end:
            raise ValueError(
                f"資料不足！需要至少 {self.train_months + self.test_months} 個月，"
                f"但資料只有 {(data_end - data_start).days // 30} 個月。\n"
                f"請減小 train_months 或 test_months。"
            )

        # ── 逐 Fold 生成切分 ─────────────────────────────────────────────────
        fold_idx = 1
        # train_end 從第一個訓練結束日開始，每次向前推進 step_months
        train_end_cursor = first_train_end

        while True:
            # 測試集：從 train_end 開始，向後 test_months 個月
            test_start = train_end_cursor
            test_end   = train_end_cursor + pd.DateOffset(months=self.test_months)

            # 若測試集超出資料範圍，停止
            if test_end > data_end + pd.DateOffset(days=1):
                break

            # ── 切分資料 ────────────────────────────────────────────────────
            # 訓練集：draw_date < test_start（嚴格早於測試集起始日）
            # 這確保訓練集的「最後一筆資料」不晚於測試集的「第一筆資料」
            #
            # 關鍵：使用 < 而非 <=，防止「同一天的資料」同時出現在訓練和測試集
            train_mask = df[self.date_col] < test_start
            test_mask  = (df[self.date_col] >= test_start) & (df[self.date_col] < test_end)

            train_df = df[train_mask].copy()
            test_df  = df[test_mask].copy()

            # 跳過樣本過少的 Fold（通常不會發生，但作為防禦性程式設計）
            if len(train_df) < 50 or len(test_df) < 5:
                logger.debug(f"Fold {fold_idx}: 樣本過少，跳過")
                train_end_cursor += pd.DateOffset(months=self.step_months)
                fold_idx += 1
                continue

            fold_info = FoldInfo(
                fold_idx    = fold_idx,
                train_start = train_df[self.date_col].min().strftime("%Y-%m-%d"),
                train_end   = train_df[self.date_col].max().strftime("%Y-%m-%d"),
                test_start  = test_df[self.date_col].min().strftime("%Y-%m-%d"),
                test_end    = test_df[self.date_col].max().strftime("%Y-%m-%d"),
                n_train     = len(train_df),
                n_test      = len(test_df),
            )

            logger.info(str(fold_info))
            yield train_df, test_df, fold_info

            # 向前推進
            train_end_cursor += pd.DateOffset(months=self.step_months)
            fold_idx += 1

    def get_fold_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        不執行評估，只回傳所有 Fold 的日期摘要表。
        用於在實際訓練前確認切分方案是否符合預期。

        Returns:
            pd.DataFrame，包含每個 Fold 的訓練/測試期數與日期範圍
        """
        rows = []
        for _, _, info in self.split(df):
            rows.append({
                "fold":         info.fold_idx,
                "train_start":  info.train_start,
                "train_end":    info.train_end,
                "test_start":   info.test_start,
                "test_end":     info.test_end,
                "n_train":      info.n_train,
                "n_test":       info.n_test,
            })
        return pd.DataFrame(rows)
