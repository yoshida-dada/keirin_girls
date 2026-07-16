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
import http.server
import json
import os
import socketserver
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # src.notify を import するため
JST = timezone(timedelta(hours=9))
DASH = ROOT / "dashboard"
DATA_JSON = DASH / "data.json"
NOTIFIED_PATH = ROOT / "data" / "notified.json"
PY = sys.executable

WINDOW_MIN = 30        # 発走何分前から1分更新を始めるか
EARLY_WINDOW = 120     # 締切何分前から粗い間隔でオッズ時系列を取り始めるか（ソフトなオッズ捕捉）
LIVE_SLEEP = 60        # 更新窓内のループ間隔(秒)=1分
IDLE_SLEEP = 300       # 更新対象が無いときのループ間隔(秒)=5分（早期スナップショットにも使う）
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
    line = f"[{datetime.now(JST):%m-%d %H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        # Windowsのcp932コンソール/リダイレクトに載らない文字(例: �)でも落とさない
        enc = (sys.stdout.encoding or "utf-8")
        sys.stdout.buffer.write((line + "\n").encode(enc, "replace"))
        sys.stdout.flush()


class DashHandler(http.server.SimpleHTTPRequestHandler):
    """dashboard を配信しつつ POST /push/subscribe で通知購読を受け付ける。"""

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(DASH), **k)

    def log_message(self, *a, **k):  # アクセスログ抑制
        pass

    def _json(self, code: int, obj: dict) -> None:
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")   # localhost閲覧からの購読POSTを許可
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path.rstrip("/") != "/push/subscribe":
            self._json(404, {"ok": False}); return
        try:
            n = int(self.headers.get("Content-Length", 0) or 0)
            sub = json.loads(self.rfile.read(n).decode("utf-8"))
            from src.notify.webpush import add_subscription
            cnt = add_subscription(sub)
            self._json(200, {"ok": True, "count": cnt})
            _log(f"通知購読を登録（計{cnt}件）")
        except Exception as e:
            self._json(400, {"ok": False, "error": str(e)})


def serve_dashboard() -> None:
    with socketserver.TCPServer(("127.0.0.1", PORT), DashHandler) as httpd:
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


def live_snapshot() -> None:
    """締切 EARLY_WINDOW 分前〜の三連単オッズを軽量取得して時系列蓄積（予測はしない）。"""
    rc, out = _run([PY, "scripts/snapshot_odds.py", "--within", str(EARLY_WINDOW)])
    last = out.strip().splitlines()[-1] if out.strip() else ""
    _log(("オッズ時系列 " + last) if rc == 0 else "オッズ時系列 失敗: " + out[-200:])


def live_results() -> None:
    rc, out = _run([PY, "scripts/fetch_results.py"])
    if rc == 0:
        last = out.strip().splitlines()[-1] if out.strip() else ""
        _log("結果取得 " + last)
    else:
        _log("結果取得 失敗: " + out[-300:])


def _pending_results(now: datetime) -> bool:
    """締切+20分を過ぎたのに結果未取得のレースが data.json にあるか。"""
    if not DATA_JSON.exists():
        return False
    try:
        races = json.loads(DATA_JSON.read_text(encoding="utf-8")).get("predictions", {}).get("races", [])
    except Exception:
        return False
    for r in races:
        if r.get("result"):
            continue
        dl = r.get("deadline")
        if dl and ":" in str(dl):
            h, m = (int(x) for x in str(dl).split(":"))
            d = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if (now - d).total_seconds() >= 20 * 60:
                return True
    return False


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


def _load_notified(today: str) -> set[str]:
    try:
        d = json.loads(NOTIFIED_PATH.read_text(encoding="utf-8"))
        if d.get("date") == today:
            return set(d.get("keys", []))
    except Exception:
        pass
    return set()


def _save_notified(today: str, keys: set[str]) -> None:
    try:
        NOTIFIED_PATH.write_text(json.dumps({"date": today, "keys": sorted(keys)},
                                            ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _notify_text(r: dict, dl: str) -> tuple[str, str]:
    venue, rno, rtype = r.get("venue", "?"), r.get("race_no", "?"), r.get("race_type", "")
    riders = sorted(r.get("riders", []), key=lambda x: -(x.get("win_prob") or 0))
    top = riders[0] if riders else {}
    p = int(round(100 * (top.get("win_prob") or 0)))
    home = "(地元)" if top.get("home") else ""
    tri = r.get("top_trifecta") or []
    honmei = tri[0].get("combo") if tri else ""
    title = f"🚲 {venue} R{rno} まもなく締切{dl}"
    body = f"[{rtype}] ◎{top.get('car','')} {top.get('name','')}{home} {p}%"
    if honmei:
        body += f" / 本命 {honmei}"
    return title, body


def notify_lead(now: datetime) -> None:
    """締切 NOTIFY_LEAD_MIN 分前になった当日レースを、スマホへ1回だけ通知する（全ガールズレース）。"""
    try:
        from src.notify.webpush import send_all, enabled, load_subs
    except Exception:
        return
    if not enabled() or not load_subs():
        return
    try:
        lead = int(os.environ.get("NOTIFY_LEAD_MIN", "10"))
    except ValueError:
        lead = 10
    if not DATA_JSON.exists():
        return
    try:
        races = json.loads(DATA_JSON.read_text(encoding="utf-8")).get("predictions", {}).get("races", [])
    except Exception:
        return
    today = now.strftime("%Y-%m-%d")
    done = _load_notified(today)
    changed = False
    for r in races:
        dl = r.get("deadline")
        if not (dl and ":" in str(dl)):
            continue
        key = f"{r.get('venue')}|{r.get('race_no')}"
        if key in done:
            continue
        h, m = (int(x) for x in str(dl).split(":"))
        d = now.replace(hour=h, minute=m, second=0, microsecond=0)
        mins = (d - now).total_seconds() / 60
        if 0 < mins <= lead:                      # 締切lead分前〜締切まで（過ぎたら通知しない）
            title, body = _notify_text(r, dl)
            ok, total = send_all(title, body, url="./", tag=key)
            _log(f"発走前通知 {key} 締切{dl} → {ok}/{total}件")
            done.add(key)
            changed = True
    if changed:
        _save_notified(today, done)


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
    # ログ出力をUTF-8化（cp932に無い文字での常駐クラッシュを防ぐ／ログも文字化けしない）
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
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

        if _pending_results(now):                 # 締切+20分経過レースの結果取得＋反映
            live_results()
            if not args.no_push and time.time() - last_push >= PUSH_INTERVAL:
                git_push(); last_push = time.time()

        notify_lead(now)                          # 締切N分前になったレースをスマホへ通知（1回だけ）

        nd = _next_deadline_min(now)              # 最短の未到来締切（分）
        if nd is not None and nd <= WINDOW_MIN + 5:
            live_refresh()                        # 締切30分前〜→1分更新（予測+オッズ、時系列も保存）
            if not args.no_push and time.time() - last_push >= PUSH_INTERVAL:
                git_push(); last_push = time.time()
            time.sleep(LIVE_SLEEP)
        elif nd is not None and nd <= EARLY_WINDOW:
            live_snapshot()                       # 締切120分前〜30分→5分間隔でオッズ時系列を軽量取得
            time.sleep(IDLE_SLEEP)
        else:
            time.sleep(IDLE_SLEEP)                # 窓外→ゆっくり待機


if __name__ == "__main__":
    main()
