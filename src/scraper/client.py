"""
client.py — CA Lottery API HTTP 客戶端
========================================
封裝所有 HTTP 請求細節，讓其他模組只需呼叫 fetch_page() 即可。

設計重點：
  • 使用 Session 複用 TCP 連線（效能）
  • 自動重試：指數退避（Exponential Backoff）策略
    - 首次失敗等 2 秒、再失敗等 4 秒、再等 8 秒，最多重試 3 次
    - 針對 429（太多請求）/ 5xx（伺服器錯誤）自動觸發
  • 禮貌性限速：每次請求間隔至少 REQUEST_DELAY 秒
    - 這是對網站的基本禮儀，也避免被封 IP
"""

import time
import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    API_BASE_URL,
    GAME_ID,
    PAGE_SIZE,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class LotteryApiClient:
    """
    CA Lottery JSON API 的 HTTP 客戶端。

    API 端點格式：
        GET /api/DrawGameApi/DrawGamePastDrawResults/{game_id}/{page}/{page_size}

    回應格式（JSON）：
        {
          "DrawGameName": "SuperLotto Plus",
          "PreviousDraws": [ {...draw1}, {...draw2}, ... ],
          "TotalDraws": 2691
        }
    """

    # 模擬瀏覽器的 HTTP headers，減少被伺服器封鎖的機率
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        # Referer 告訴伺服器「我是從哪個頁面發出這個 API 請求的」
        "Referer": "https://www.calottery.com/draw-games/superlotto-plus",
    }

    def __init__(
        self,
        game_id: int = GAME_ID,
        page_size: int = PAGE_SIZE,
        delay: float = REQUEST_DELAY,
        timeout: int = REQUEST_TIMEOUT,
    ):
        """
        Args:
            game_id:   CA Lottery 遊戲 ID（SuperLotto Plus = 8）
            page_size: 每頁筆數（建議 100）
            delay:     每次請求前等待秒數（禮貌限速）
            timeout:   請求逾時秒數
        """
        self.game_id   = game_id
        self.page_size = page_size
        self.delay     = delay
        self.timeout   = timeout
        self.session   = self._build_session()

    def _build_session(self) -> requests.Session:
        """
        建立帶有自動重試機制的 requests Session。

        Retry 參數說明：
          total=3         → 最多重試 3 次
          backoff_factor=2 → 等待時間 = 2^(重試次數-1) 秒（2s, 4s, 8s）
          status_forcelist → 遇到這些 HTTP 狀態碼時自動重試
        """
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            # 不重試 POST（我們只用 GET，保險起見設定）
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)

        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://",  adapter)
        session.headers.update(self.HEADERS)
        return session

    def fetch_page(self, page: int) -> dict[str, Any]:
        """
        抓取指定頁碼的開獎資料。

        Args:
            page: 頁碼（從 1 開始）

        Returns:
            API 回傳的 JSON dict，包含 PreviousDraws 清單

        Raises:
            requests.HTTPError: HTTP 狀態碼為 4xx/5xx 且重試後仍失敗
            requests.Timeout:   請求逾時
        """
        url = f"{API_BASE_URL}/{self.game_id}/{page}/{self.page_size}"

        # 禮貌性等待：在發出請求前先暫停，避免過於頻繁
        time.sleep(self.delay)

        logger.debug(f"正在請求：{url}")
        response = self.session.get(url, timeout=self.timeout)

        # raise_for_status() 會在 4xx/5xx 時拋出 HTTPError 例外
        response.raise_for_status()

        data = response.json()
        draw_count = len(data.get("PreviousDraws", []))
        total = data.get("TotalDrawCount", "未知")
        logger.debug(f"第 {page} 頁：取得 {draw_count} 筆，總計 {total} 筆")

        return data

    def fetch_total_count(self) -> int:
        """
        取得歷史開獎總筆數（透過抓第 1 頁取得 TotalDrawCount）。

        Returns:
            int: 總開獎期數
        """
        data = self.fetch_page(page=1)
        return data.get("TotalDrawCount", 0)

    def close(self):
        """釋放 HTTP Session 資源（長時間執行的程式應該在結束時呼叫）"""
        self.session.close()
        logger.debug("HTTP Session 已關閉")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
