"""スクレイパー共通基盤（S1）。規約遵守の丁寧な取得・リトライ・エンコーディング・欠損検知。

方針（CLAUDE.md開発ルール4/課題G）:
  * アクセス間隔は SCRAPE_MIN_INTERVAL_SEC 以上（プロセス内でホスト単位に自動スロットリング）。
  * 一時的失敗（5xx/接続エラー）は指数バックオフで数回リトライ。4xxは即中断。
  * GambooBET等の日本語サイトは encoding を apparent_encoding で自動判定。

このモジュールは取得のみを担い、ページ固有のパースは各スクレイパー（gamboo_odds 等）が行う。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from config.settings import SCRAPE_MIN_INTERVAL_SEC

_USER_AGENT = "girls-keirin-ai/0.1 (personal research; contact via project)"
_last_access: dict[str, float] = {}   # host -> 最終アクセス時刻(monotonic)
_default_interval = SCRAPE_MIN_INTERVAL_SEC  # 実行時に set_default_interval() で変更可


def set_default_interval(sec: float) -> None:
    """全 fetch() のホスト間隔の既定値を変更する（バックフィルの高速化用）。"""
    global _default_interval
    _default_interval = max(0.0, float(sec))


def _throttle(host: str, min_interval: float) -> None:
    """ホスト単位で前回アクセスから min_interval 秒空ける。"""
    now = time.monotonic()
    wait = min_interval - (now - _last_access.get(host, 0.0))
    if wait > 0:
        time.sleep(wait)
    _last_access[host] = time.monotonic()


@dataclass
class FetchResult:
    url: str
    status: int
    text: str
    encoding: str


def fetch(
    url: str,
    *,
    min_interval: float | None = None,     # None のとき _default_interval（実行時変更可）を使う
    max_retries: int = 3,
    timeout: int = 25,
    session: requests.Session | None = None,
) -> FetchResult:
    """1URLを丁寧に取得する。4xxは即例外、5xx/接続断は指数バックオフでリトライ。"""
    if min_interval is None:
        min_interval = _default_interval
    host = urlparse(url).netloc
    sess = session or requests
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        _throttle(host, min_interval)
        try:
            r = sess.get(url, headers={"User-Agent": _USER_AGENT}, timeout=timeout)
        except requests.RequestException as e:      # 接続断・タイムアウト
            last_exc = e
            time.sleep(min_interval * (2 ** attempt))
            continue
        if 400 <= r.status_code < 500:
            raise RuntimeError(f"client error {r.status_code} for {url}")
        if r.status_code >= 500:
            last_exc = RuntimeError(f"server error {r.status_code}")
            time.sleep(min_interval * (2 ** attempt))
            continue
        r.encoding = r.apparent_encoding or r.encoding
        return FetchResult(url=url, status=r.status_code, text=r.text, encoding=r.encoding)
    raise RuntimeError(f"fetch failed after {max_retries} retries: {url}") from last_exc


def detect_missing_trifecta(odds: dict[tuple, float], field_size: int) -> list[tuple]:
    """N車立てで期待される三連単 N*(N-1)*(N-2) 点に対し、欠損している組合せを返す（欠損検知）。"""
    from itertools import permutations
    cars = range(1, field_size + 1)
    expected = set(permutations(cars, 3))
    return sorted(expected - set(odds.keys()))
