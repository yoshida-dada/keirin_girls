"""PWA Web Push 送信。VAPIDでスマホのプッシュ購読へ通知を送る（別アプリ不要のOSネイティブ通知）。

購読情報は `data/push_subs.json`（gitignore対象・スマホの購読エンドポイント＝準秘匿）に貯める。
スケジューラが発走前に `send_all(title, body, url, tag)` を呼び、全購読へ配信。410/404で失効した
購読は自動削除する。VAPID鍵は `.env`（VAPID_PRIVATE_KEY/PUBLIC_KEY/SUBJECT）から読む。

  from src.notify.webpush import send_all, add_subscription, load_subs
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
SUBS_PATH = ROOT / "data" / "push_subs.json"
_LOCK = threading.Lock()


def _vapid() -> tuple[str, str]:
    """(private_key_b64url, 'mailto:...') を返す。未設定なら ('','')。"""
    priv = os.environ.get("VAPID_PRIVATE_KEY", "").strip()
    subj = os.environ.get("VAPID_SUBJECT", "mailto:admin@example.com").strip()
    return priv, subj


def enabled() -> bool:
    return os.environ.get("NOTIFY_ENABLED", "1").strip() not in ("0", "false", "False", "")


def load_subs() -> list[dict]:
    if not SUBS_PATH.exists():
        return []
    try:
        return json.loads(SUBS_PATH.read_text(encoding="utf-8")).get("subs", [])
    except Exception:
        return []


def _save_subs(subs: list[dict]) -> None:
    SUBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUBS_PATH.write_text(json.dumps({"subs": subs}, ensure_ascii=False, indent=2), encoding="utf-8")


def _endpoint(sub: dict) -> str:
    return (sub or {}).get("endpoint", "")


def add_subscription(sub: dict) -> int:
    """購読を追加（endpointで重複排除）。保存後の購読数を返す。"""
    if not sub or not _endpoint(sub):
        return len(load_subs())
    with _LOCK:
        subs = load_subs()
        eps = {_endpoint(s) for s in subs}
        if _endpoint(sub) not in eps:
            subs.append(sub)
            _save_subs(subs)
        return len(subs)


def send_all(title: str, body: str, url: str = "./", tag: str | None = None) -> tuple[int, int]:
    """全購読へ Web Push 送信。(成功数, 総数) を返す。失効購読は自動削除。"""
    priv, subj = _vapid()
    subs = load_subs()
    if not (priv and subs and enabled()):
        return 0, len(subs)
    from pywebpush import webpush, WebPushException

    payload = json.dumps({"title": title, "body": body, "url": url, "tag": tag or "keirin"})
    ok = 0
    dead: list[str] = []
    for s in subs:
        try:
            webpush(subscription_info=s, data=payload,
                    vapid_private_key=priv, vapid_claims={"sub": subj}, timeout=10)
            ok += 1
        except WebPushException as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code in (404, 410):          # 購読失効 → 掃除
                dead.append(_endpoint(s))
        except Exception:
            pass
    if dead:
        with _LOCK:
            remain = [s for s in load_subs() if _endpoint(s) not in dead]
            _save_subs(remain)
    return ok, len(subs)
