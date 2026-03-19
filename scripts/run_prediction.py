"""
run_prediction.py — 執行完整預測流程
=======================================
依序執行：
  1. 讀取資料庫中的所有開獎資料
  2. 訓練所有模型（RF + XGBoost + 基準）
  3. 評估模型在測試集上的表現
  4. 輸出下一期號碼推薦

使用方式：
  python scripts/run_prediction.py

選項：
  --no-xgb     跳過 XGBoost 訓練（較快）
  --save        訓練後存檔模型
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config.settings import DB_PATH, LOG_LEVEL, LOG_FORMAT
from src.database.repository import DrawRepository
from src.models.classifier import LotteryClassifier
from src.models.predictor import print_prediction_report

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SuperLotto Plus 號碼預測")
    parser.add_argument("--no-xgb", action="store_true", help="跳過 XGBoost（加快速度）")
    parser.add_argument("--save",   action="store_true", help="訓練後存檔模型")
    return parser.parse_args()


def main():
    args = parse_args()

    if not DB_PATH.exists():
        logger.error("找不到資料庫！請先執行：init_db.py → scrape_all.py → build_features.py")
        sys.exit(1)

    # ── 1. 讀取資料 ──────────────────────────────────────────────────────────
    repo = DrawRepository(DB_PATH)
    draw_count = repo.get_draw_count()
    logger.info(f"資料庫共有 {draw_count} 筆開獎資料")

    if draw_count < 100:
        logger.warning(f"資料量太少（{draw_count} 筆），模型可靠性很低")

    df = repo.get_all_draws()

    # ── 2. 訓練模型 ──────────────────────────────────────────────────────────
    clf = LotteryClassifier()
    clf.fit(df, train_xgb=not args.no_xgb)

    if args.save:
        clf.save_models()

    # ── 3. 評估 ──────────────────────────────────────────────────────────────
    eval_results = clf.evaluate()

    # ── 4. 輸出預測報表 ──────────────────────────────────────────────────────
    prediction = clf.predict_next()
    print_prediction_report(prediction, eval_results)


if __name__ == "__main__":
    main()
