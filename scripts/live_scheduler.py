"""ローカル常駐スケジューラ（ばんえいAI同様のPC常駐運用）。

PC起動時に自動起動し:
  1) 朝（起動時／日付が変わったら）当日のガールズ予測を算出して dashboard/data.json を生成
  2) ダッシュボードをローカル配信（http://127.0.0.1:8787）＝PCではリアルタイム閲覧
  3) 各レースの発走30分前〜締切まで **1分周期**でオッズを再取得しEVを更新（ローカルは即反映）
  4) GitHub Pages（スマホ/遠隔用）へは数分間隔でpush（Pagesのビルド回数上限のため）

  python scripts/live_scheduler.py                 # 常駐起動
  python scripts/live_scheduler.py --once          # 1回だけ更新して終了（動作確認）
  python scripts/live_scheduler.py --no-serve --no-push   # 更新のみ

ローカルDB(data/keirin.sqlite)と学習済みモデルを使う。git push はローカルのgit認証を使用。
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import socketserver
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JST = timezone(timedelta(hours=9))
DASH = ROOT / "dashboard"
DATA_JSON = DASH / "data.json"
PY = sys.executable

WINDOW_MIN = 30        # 発走何分前から1分更新を始めるか
LIVE_SLEEP = 60        # 更新窓内のループ間隔(秒)=1分
IDLE_SLEEP = 300       # 更新対象が無いときのループ間隔(秒)
PUSH_INTERVAL = 420    # Pagesへpushする最短間隔(秒)。Pagesビルド上限(約10回/時)を守る
PORT = 8787


def _run(cmd: list[str], timeout: int = 600) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


def _log(msg: str) -> None:
    print(f"[{datetime.now(JST):%m-%d %H:%M:%S}] {msg}", flush=True)


def serve_dashboard() -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DASH))
    handler.log_message = lambda *a, **k: None  # アクセスログ抑制
    with socketserver.TCPServer(("127.0.0.1", PORT), handler) as httpd:
        httpd.serve_forever()


def morning_build() -> None:
    _log("当日の予測を算出中（build_predictions --predict）…")
    rc, out = _run([PY, "scripts/build_predictions.py", "--db", "data/keirin.sqlite", "--predict"])
    _log(("朝の予測生成 完了" if rc == 0 else "朝の予測生成 失敗\n" + out[-500:]))


def live_refresh() -> None:
    rc, out = _run([PY, "scripts/refresh_predictions.py", "--only-near", str(WINDOW_MIN)])
    if rc == 0:
        last = out.strip().splitlines()[-1] if out.strip() else ""
        _log("オッズ更新 " + last)
    else:
        _log("オッズ更新 失敗: " + out[-300:])


def _next_deadline_min(now: datetime) -> float | None:
    """data.json の当日レースの締切のうち、まだ来ていない最短の「分」を返す。無ければNone。"""
    if not DATA_JSON.exists():
        return None
    try:
        races = json.loads(DATA_JSON.read_text(encoding="utf-8")).get("predictions", {}).get("races", [])
    except Exception:
        return None
    mins = []
    for r in races:
        dl = r.get("deadline")
        if dl and ":" in dl:
            h, m = (int(x) for x in dl.split(":"))
            d = now.replace(hour=h, minute=m, second=0, microsecond=0)
            mm = (d - now).total_seconds() / 60
            if mm > -5:
                mins.append(mm)
    return min(mins) if mins else None


def git_push() -> None:
    rc, out = _run(["git", "diff", "--quiet", "--", "dashboard/data.json"])
    if rc == 0:
        return  # 変更なし
    _run(["git", "add", "dashboard/data.json"])
    _run(["git", "commit", "-m", "chore: live odds refresh"])
    _run(["git", "pull", "--rebase"])
    rc, out = _run(["git", "push"])
    _log("Pagesへ反映 " + ("完了" if rc == 0 else "失敗: " + out[-200:]))


def main() -> None:
    ap = argparse.ArgumentParser(description="ローカル常駐 予測/オッズ更新スケジューラ")
    ap.add_argument("--once", action="store_true", help="1回だけ更新して終了")
    ap.add_argument("--no-serve", action="store_true", help="ローカル配信をしない")
    ap.add_argument("--no-push", action="store_true", help="Pagesへpushしない")
    args = ap.parse_args()

    if args.once:
        morning_build(); live_refresh()
        if not args.no_push:
            git_push()
        return

    if not args.no_serve:
        threading.Thread(target=serve_dashboard, daemon=True).start()
        _log(f"ダッシュボードをローカル配信中: http://127.0.0.1:{PORT}/")

    served_date = None
    last_push = 0.0
    _log("常駐スケジューラ開始（Ctrl+Cで停止）")
    while True:
        now = datetime.now(JST)
        if now.date() != served_date:            # 日付が変わったら朝の予測を作る
            morning_build()
            served_date = now.date()
            if not args.no_push:
                git_push(); last_push = time.time()

        nd = _next_deadline_min(now)              # 最短の未到来締切（分）
        if nd is not None and nd <= WINDOW_MIN + 5:
            live_refresh()                        # 更新窓内→1分更新
            if not args.no_push and time.time() - last_push >= PUSH_INTERVAL:
                git_push(); last_push = time.time()
            time.sleep(LIVE_SLEEP)
        else:
            time.sleep(IDLE_SLEEP)                # 窓外→ゆっくり待機


if __name__ == "__main__":
    main()
