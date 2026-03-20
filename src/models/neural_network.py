"""
neural_network.py — 神經網路預測模型（MLP + LSTM）
====================================================

【兩種架構的核心差異】

  MLPPredictor（多層感知機）
  ─────────────────────────
  輸入：一個「靜態特徵快照」（94 維）
        [gap_1..47, freq_1..47]
  輸出：47 個號碼各自的出現機率

  特點：忽略時序順序，只看「現在的統計狀態」。
        是最簡單的神經網路形式，適合作為 NN 的入門基準。

  LSTMPredictor（長短期記憶網路）
  ────────────────────────────────
  輸入：過去 N=20 期的「開獎序列」
        每期編碼為 47 維 one-hot 向量 → shape = (20, 47)
  輸出：47 個號碼各自的出現機率

  特點：LSTM 的「記憶機制」能捕捉跨期的時序依賴：
        例如「號碼 7 在 10 期前出現過，3 期前又出現，現在再出現的機率如何」。
        這是其他非序列模型做不到的。

【Rolling Update（增量學習）策略】

  每看到一期新開獎後：
  - MLP：使用 partial_fit 對新特徵+標籤做幾步梯度更新
  - LSTM：在現有模型基礎上，以新一期資料再訓練 fine_tune_epochs 輪

  這模擬了「模型在實際部署中隨時間持續學習」的場景。
  注意：若學習率過高，fine-tuning 可能使模型「遺忘」早期學到的規律
       （Catastrophic Forgetting），這是深度學習中的重要研究課題。

【數學底層（LSTM）】

  LSTM 的核心是三個「閘門（Gate）」：

    遺忘閘：f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
            決定「要忘記多少過去的記憶」

    輸入閘：i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
            C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
            決定「要記住多少新資訊」

    輸出閘：o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
            h_t = o_t ⊙ tanh(C_t)
            決定「輸出多少記憶」

  其中 C_t（Cell State）是跨越長時間的「記憶帶」，
  這讓 LSTM 能記住幾十步之前的資訊，而普通 RNN 做不到。
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from config.settings import (
    WHITE_BALL_MIN, WHITE_BALL_MAX,
    LOG_LEVEL, LOG_FORMAT,
)
from src.features.base_stats import compute_white_ball_frequency, build_number_current_gap

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

WHITE_NUMBERS = list(range(WHITE_BALL_MIN, WHITE_BALL_MAX + 1))
WHITE_COLS    = ["n1", "n2", "n3", "n4", "n5"]
N_WHITE       = len(WHITE_NUMBERS)   # 47


# ─────────────────────────────────────────────────────────────────────────────
# 共用特徵建構函式
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_vector(df_upto: pd.DataFrame, freq_window: int = 30) -> np.ndarray:
    """
    建立 94 維靜態特徵向量（給 MLP 使用）：
      [gap_1, ..., gap_47, freq_1, ..., freq_47]
    """
    gap_dict      = build_number_current_gap(df_upto)
    gap_features  = np.array([gap_dict[n]           for n in WHITE_NUMBERS], dtype=np.float32)
    freq          = compute_white_ball_frequency(df_upto, window=freq_window)
    freq_features = np.array([freq.get(n, 0)         for n in WHITE_NUMBERS], dtype=np.float32)
    return np.concatenate([gap_features, freq_features])


def _build_label_vector(row: pd.Series) -> np.ndarray:
    """將一期開獎記錄轉換為 47 維二元標籤向量。"""
    appeared = {int(row[c]) for c in WHITE_COLS}
    return np.array([1 if n in appeared else 0 for n in WHITE_NUMBERS], dtype=np.float32)


def _draw_to_binary(row: pd.Series) -> np.ndarray:
    """將一期開獎轉為 47 維 one-hot 二元向量（給 LSTM 使用）。"""
    vec      = np.zeros(N_WHITE, dtype=np.float32)
    appeared = [int(row[c]) - WHITE_BALL_MIN for c in WHITE_COLS]
    vec[appeared] = 1.0
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# MLPPredictor — 多層感知機（PyTorch 實作）
# ─────────────────────────────────────────────────────────────────────────────

class _MLPNet:
    """PyTorch MLP 內部網路，由 MLPPredictor 管理。"""

    def __init__(self, input_dim: int, hidden1: int, hidden2: int, output_dim: int):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("MLPPredictor 需要 PyTorch。請執行：pip install torch")

        import torch.nn as nn

        class Net(nn.Module):
            """
            架構：Linear(94→128) → BatchNorm → ReLU → Dropout(0.3)
                  → Linear(128→64) → BatchNorm → ReLU → Dropout(0.3)
                  → Linear(64→47)
            輸出為 logits（未經 Sigmoid），Loss 使用 BCEWithLogitsLoss。
            """
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden1),
                    nn.BatchNorm1d(hidden1),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden1, hidden2),
                    nn.BatchNorm1d(hidden2),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden2, output_dim),
                )

            def forward(self, x):
                return self.net(x)

        self._Net = Net
        self._torch = torch

    def build(self):
        return self._Net()


class MLPPredictor:
    """
    多層感知機彩票預測器（PyTorch 版）。

    架構：
      Input(94) → Linear(128) → BN → ReLU → Dropout
               → Linear(64)  → BN → ReLU → Dropout
               → Linear(47)  → [predict 時加 Sigmoid]
    Loss: BCEWithLogitsLoss（等同 47 個獨立二元交叉熵）

    Rolling Update 機制：
      partial_fit(new_df) 以現有模型為起點，只對新一期資料做
      fine_tune_epochs 次梯度下降，速度極快（毫秒級）。
    """

    def __init__(
        self,
        hidden1:           int   = 128,
        hidden2:           int   = 64,
        lr:                float = 1e-3,
        epochs:            int   = 100,
        batch_size:        int   = 64,
        fine_tune_epochs:  int   = 5,
        freq_window:       int   = 30,
        lookback:          int   = 100,
        device:            str   = "auto",
    ):
        """
        Args:
            hidden1:          第一隱藏層神經元數
            hidden2:          第二隱藏層神經元數
            lr:               Adam 學習率
            epochs:           初次訓練的 epoch 數
            batch_size:       Mini-batch 大小
            fine_tune_epochs: Rolling update 每次的 epoch 數（太多 → 遺忘，太少 → 學不到）
            freq_window:      近期頻率計算視窗
            lookback:         建構訓練樣本的最少前置期數
            device:           "auto" / "cpu" / "cuda" / "mps"
        """
        self.hidden1          = hidden1
        self.hidden2          = hidden2
        self.lr               = lr
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.fine_tune_epochs = fine_tune_epochs
        self.freq_window      = freq_window
        self.lookback         = lookback

        try:
            import torch
            self._torch = torch
            if device == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(device)
        except ImportError:
            raise ImportError("MLPPredictor 需要 PyTorch。請執行：pip install torch")

        self._model      = None
        self._optimizer  = None
        self._loss_fn    = None
        self._df         = None
        self._is_fitted  = False

        logger.info(f"MLPPredictor 使用裝置：{self._device}")

    def _build_dataset(self, df: pd.DataFrame):
        """從 DataFrame 建立特徵矩陣與標籤矩陣（PyTorch Tensor）。"""
        import torch
        min_periods = max(self.freq_window + 5, self.lookback)
        X_list, y_list = [], []

        for i in range(min_periods, len(df)):
            fv    = _build_feature_vector(df.iloc[:i], self.freq_window)
            label = _build_label_vector(df.iloc[i])
            X_list.append(fv)
            y_list.append(label)

        if not X_list:
            return None, None

        X = torch.tensor(np.array(X_list), dtype=torch.float32).to(self._device)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).to(self._device)
        return X, y

    def _train_loop(self, X, y, epochs: int):
        """內部訓練迴圈（Mini-batch Gradient Descent）。"""
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        dataset    = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                self._optimizer.zero_grad()
                logits = self._model(X_batch)
                loss   = self._loss_fn(logits, y_batch)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                logger.debug(f"  MLP Epoch {epoch+1}/{epochs}  loss={epoch_loss/len(dataloader):.4f}")

    def fit(self, df: pd.DataFrame) -> "MLPPredictor":
        """
        從頭訓練 MLP 模型。

        Args:
            df: 開獎資料 DataFrame（已按 draw_date 升序）
        """
        import torch
        import torch.nn as nn

        df          = df.sort_values("draw_date").reset_index(drop=True)
        self._df    = df.copy()
        input_dim   = N_WHITE * 2   # 94 維

        # ── 建立網路 ────────────────────────────────────────────────────────
        class Net(nn.Module):
            def __init__(self, h1, h2):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(h1, h2),        nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(h2, N_WHITE),
                )
            def forward(self, x):
                return self.net(x)

        self._model    = Net(self.hidden1, self.hidden2).to(self._device)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-4)
        self._loss_fn   = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([42.0 / 5.0] * N_WHITE).to(self._device)
            # pos_weight：正樣本（出現）的損失乘以 8.4，補償類別不平衡
        )

        logger.info(f"MLP：建立訓練資料（共 {len(df)} 期）...")
        X, y = self._build_dataset(df)
        if X is None:
            raise ValueError("訓練資料不足")

        logger.info(f"MLP：開始訓練（{self.epochs} epochs，{len(X)} 樣本）...")
        self._train_loop(X, y, self.epochs)

        self._is_fitted = True
        logger.info("MLP 訓練完成")
        return self

    def partial_fit(self, df_expanded: pd.DataFrame) -> "MLPPredictor":
        """
        Rolling Update：以新資料對現有模型做微調。

        Args:
            df_expanded: 包含新一期資料的完整 DataFrame
                         （不需要重新建構全部特徵，只用最後幾筆）
        """
        if not self._is_fitted:
            return self.fit(df_expanded)

        df = df_expanded.sort_values("draw_date").reset_index(drop=True)
        self._df = df.copy()

        # 只對最後 min(lookback, len(df)) 筆做 fine-tuning
        recent = df.tail(min(self.lookback, len(df))).reset_index(drop=True)
        X, y   = self._build_dataset(recent)
        if X is None or len(X) < 5:
            return self

        self._train_loop(X, y, self.fine_tune_epochs)
        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        以最後一期資料為輸入，輸出 47 顆球的機率，選 top-n_white。

        Returns:
            {"white_balls": [int,...], "mega_balls": [], "proba": {號碼: 機率}}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        import torch

        fv = _build_feature_vector(self._df, self.freq_window)
        x  = torch.tensor(fv, dtype=torch.float32).unsqueeze(0).to(self._device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x)[0]
            proba  = torch.sigmoid(logits).cpu().numpy()

        top_idx   = np.argsort(proba)[::-1][:n_white]
        top_white = sorted([WHITE_NUMBERS[i] for i in top_idx])
        proba_map = {WHITE_NUMBERS[i]: float(proba[i]) for i in range(N_WHITE)}

        logger.info(f"MLP 推薦：{top_white}（最高機率：{proba[top_idx[0]]:.4f}）")
        return {"white_balls": top_white, "mega_balls": [], "proba": proba_map}


# ─────────────────────────────────────────────────────────────────────────────
# LSTMPredictor — 長短期記憶網路（PyTorch 實作）
# ─────────────────────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    LSTM 時序預測器。

    輸入：過去 seq_len=20 期的開獎序列
          每期編碼為 47 維二元向量（出現=1，未出現=0）
          → shape = (batch, seq_len, 47)

    架構：
      LSTM(47 → 128, layers=2, dropout=0.3)
      → 取最後一個時間步的隱藏狀態 h_T
      → Linear(128 → 64) → ReLU
      → Linear(64 → 47)  → [predict 時加 Sigmoid]

    Loss：BCEWithLogitsLoss（相同的類別不平衡加權）

    為何選用 LSTM 而不是普通 RNN？
      普通 RNN 在反向傳播時會遇到「梯度消失問題」（Vanishing Gradient），
      對 20 步前的資訊幾乎無法學習。LSTM 的 Cell State 像一條「資訊高速公路」，
      允許梯度無衰減地流過長時間序列。
    """

    def __init__(
        self,
        seq_len:           int   = 20,
        hidden_size:       int   = 128,
        num_layers:        int   = 2,
        dropout:           float = 0.3,
        lr:                float = 1e-3,
        epochs:            int   = 80,
        batch_size:        int   = 32,
        fine_tune_epochs:  int   = 5,
        device:            str   = "auto",
    ):
        """
        Args:
            seq_len:         使用過去幾期作為輸入序列（建議 10–30）
            hidden_size:     LSTM 隱藏狀態維度
            num_layers:      LSTM 堆疊層數（2 層能捕捉更複雜的時序模式）
            dropout:         LSTM 層間的 Dropout 比率（防止過擬合）
            lr:              Adam 學習率
            epochs:          初次訓練的 epoch 數
            batch_size:      Mini-batch 大小
            fine_tune_epochs: Rolling update 每次的 epoch 數
            device:          "auto" / "cpu" / "cuda" / "mps"
        """
        self.seq_len          = seq_len
        self.hidden_size      = hidden_size
        self.num_layers       = num_layers
        self.dropout          = dropout
        self.lr               = lr
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.fine_tune_epochs = fine_tune_epochs

        try:
            import torch
            self._torch = torch
            if device == "auto":
                if torch.cuda.is_available():
                    self._device = torch.device("cuda")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                else:
                    self._device = torch.device("cpu")
            else:
                self._device = torch.device(device)
        except ImportError:
            raise ImportError("LSTMPredictor 需要 PyTorch。請執行：pip install torch")

        self._model     = None
        self._optimizer = None
        self._loss_fn   = None
        self._df        = None
        self._is_fitted = False

        logger.info(f"LSTMPredictor 使用裝置：{self._device}")

    def _build_lstm_model(self):
        """建立 LSTM 網路並移至設備。"""
        import torch.nn as nn
        seq_len      = self.seq_len
        hidden_size  = self.hidden_size
        num_layers   = self.num_layers
        dropout      = self.dropout

        class LSTMNet(nn.Module):
            """
            LSTM 網路架構：

              輸入序列 (batch, seq_len, 47)
                ↓
              LSTM(input=47, hidden=128, layers=2, dropout=0.3)
                ↓ 取最後時間步 h_T: (batch, 128)
              Linear(128 → 64) → ReLU
                ↓
              Linear(64 → 47)   ← 輸出 logits（BCEWithLogitsLoss）

            為何取最後時間步？
              LSTM 的最後一個隱藏狀態 h_T 已「消化」了整個輸入序列的資訊，
              相當於「看完過去 20 期後，我現在的判斷是什麼」。
            """
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size  = N_WHITE,
                    hidden_size = hidden_size,
                    num_layers  = num_layers,
                    dropout     = dropout if num_layers > 1 else 0.0,
                    batch_first = True,    # batch 維度在第一位：(batch, seq, feature)
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, N_WHITE),
                )

            def forward(self, x):
                # x: (batch, seq_len, 47)
                lstm_out, _  = self.lstm(x)
                # 只取最後一個時間步的輸出：(batch, hidden_size)
                last_hidden  = lstm_out[:, -1, :]
                return self.head(last_hidden)   # (batch, 47)

        return LSTMNet().to(self._device)

    def _build_sequences(self, df: pd.DataFrame):
        """
        將 DataFrame 轉換為 LSTM 訓練資料。

        Returns:
            X: (n_samples, seq_len, 47) 的輸入序列
            y: (n_samples, 47)          的標籤向量
        """
        import torch

        # 先把每期轉成 47 維二元向量
        binary_draws = np.array([_draw_to_binary(df.iloc[i]) for i in range(len(df))])

        X_list, y_list = [], []
        for i in range(self.seq_len, len(df)):
            # 輸入：第 [i-seq_len, i) 期的開獎向量序列
            seq   = binary_draws[i - self.seq_len : i]   # shape (seq_len, 47)
            # 標籤：第 i 期的開獎向量（這是模型要預測的）
            label = binary_draws[i]                       # shape (47,)
            X_list.append(seq)
            y_list.append(label)

        if not X_list:
            return None, None

        X = torch.tensor(np.array(X_list), dtype=torch.float32).to(self._device)
        y = torch.tensor(np.array(y_list), dtype=torch.float32).to(self._device)
        return X, y

    def _train_loop(self, X, y, epochs: int):
        """Mini-batch 訓練迴圈。"""
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        dataset    = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in dataloader:
                self._optimizer.zero_grad()
                logits = self._model(X_batch)
                loss   = self._loss_fn(logits, y_batch)
                loss.backward()
                # Gradient Clipping：防止 LSTM 中的梯度爆炸（Exploding Gradient）
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                self._optimizer.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                logger.debug(f"  LSTM Epoch {epoch+1}/{epochs}  loss={epoch_loss/len(dataloader):.4f}")

    def fit(self, df: pd.DataFrame) -> "LSTMPredictor":
        """
        從頭訓練 LSTM 模型。

        Args:
            df: 開獎資料 DataFrame（已按 draw_date 升序）
        """
        import torch

        df          = df.sort_values("draw_date").reset_index(drop=True)
        self._df    = df.copy()
        self._model = self._build_lstm_model()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=1e-4
        )
        self._loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([42.0 / 5.0] * N_WHITE).to(self._device)
        )

        logger.info(f"LSTM：建立序列資料（seq_len={self.seq_len}，共 {len(df)} 期）...")
        X, y = self._build_sequences(df)
        if X is None:
            raise ValueError(f"訓練資料不足（需要 > {self.seq_len} 期）")

        logger.info(f"LSTM：開始訓練（{self.epochs} epochs，{len(X)} 樣本）...")
        self._train_loop(X, y, self.epochs)

        self._is_fitted = True
        logger.info("LSTM 訓練完成")
        return self

    def partial_fit(self, df_expanded: pd.DataFrame) -> "LSTMPredictor":
        """
        Rolling Update：保留現有模型參數，以最新資料微調。

        Catastrophic Forgetting 警告：
          若 fine_tune_epochs 過多或 lr 過大，模型可能「遺忘」早期學到的規律。
          建議 fine_tune_epochs ≤ 10，lr 建議降為初始值的 1/10。
        """
        if not self._is_fitted:
            return self.fit(df_expanded)

        df = df_expanded.sort_values("draw_date").reset_index(drop=True)
        self._df = df.copy()

        # 只用最近的序列做 fine-tuning（避免處理全部資料）
        recent_n = self.seq_len + 50   # 足夠建立約 50 個序列樣本
        recent   = df.tail(recent_n).reset_index(drop=True)
        X, y     = self._build_sequences(recent)
        if X is None or len(X) < 5:
            return self

        # 微調時使用較小學習率，緩解 Catastrophic Forgetting
        for g in self._optimizer.param_groups:
            g["lr"] = self.lr * 0.1

        self._train_loop(X, y, self.fine_tune_epochs)

        # 恢復原始學習率
        for g in self._optimizer.param_groups:
            g["lr"] = self.lr

        return self

    def predict(self, n_white: int = 5, n_mega: int = 1) -> dict:
        """
        用最近 seq_len 期的序列預測下一期，回傳機率最高的 n_white 顆。

        Returns:
            {"white_balls": [int,...], "mega_balls": [], "proba": {號碼: 機率}}
        """
        if not self._is_fitted:
            raise RuntimeError("請先呼叫 fit()")

        import torch

        # 取最後 seq_len 期作為輸入序列
        recent       = self._df.tail(self.seq_len)
        binary_draws = np.array([_draw_to_binary(recent.iloc[i]) for i in range(len(recent))])
        x = torch.tensor(binary_draws, dtype=torch.float32).unsqueeze(0).to(self._device)
        # x.shape = (1, seq_len, 47)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x)[0]
            proba  = torch.sigmoid(logits).cpu().numpy()

        top_idx   = np.argsort(proba)[::-1][:n_white]
        top_white = sorted([WHITE_NUMBERS[i] for i in top_idx])
        proba_map = {WHITE_NUMBERS[i]: float(proba[i]) for i in range(N_WHITE)}

        logger.info(f"LSTM 推薦：{top_white}（最高機率：{proba[top_idx[0]]:.4f}）")
        return {"white_balls": top_white, "mega_balls": [], "proba": proba_map}

    def get_attention_scores(self) -> Optional[pd.DataFrame]:
        """
        回傳輸入序列中每一期的「相對重要性」估計。
        使用輸出對輸入序列的梯度絕對值作為 proxy（近似 Saliency Map）。

        Returns:
            pd.DataFrame：index=時間步（0=最舊，seq_len-1=最新），
                          values=各時間步的平均梯度絕對值（歸一化）
        """
        if not self._is_fitted:
            return None

        import torch

        recent       = self._df.tail(self.seq_len)
        binary_draws = np.array([_draw_to_binary(recent.iloc[i]) for i in range(len(recent))])
        x = torch.tensor(binary_draws, dtype=torch.float32).unsqueeze(0).to(self._device)
        x.requires_grad_(True)

        self._model.eval()
        logits = self._model(x)
        loss   = logits.sum()
        loss.backward()

        saliency = x.grad.abs().squeeze(0).mean(dim=1).cpu().detach().numpy()
        saliency = saliency / (saliency.sum() + 1e-8)

        draw_dates = recent["draw_date"].values if "draw_date" in recent.columns else range(self.seq_len)
        return pd.DataFrame({
            "step":       range(self.seq_len),
            "draw_date":  draw_dates,
            "importance": saliency,
        })
