"""PWA Web Push 用の VAPID 鍵ペアを生成する（1回だけ実行）。

出力:
  - 秘密鍵/公開鍵を標準出力に表示 → `.env` の VAPID_PRIVATE_KEY / VAPID_PUBLIC_KEY に貼る
  - dashboard/push_config.json に公開鍵を書き込む（フロントの applicationServerKey）
公開鍵は秘匿情報ではない（配布して良い）。秘密鍵は .env のみ・絶対にコミットしない。

  python scripts/gen_vapid.py            # 生成して表示＋push_config.json更新
  python scripts/gen_vapid.py --force    # 既存の push_config.json を上書き
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

ROOT = Path(__file__).resolve().parent.parent
PUSH_CONFIG = ROOT / "dashboard" / "push_config.json"


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def generate() -> tuple[str, str]:
    """(private_b64url, public_b64url) を返す。private=32byteスカラー, public=65byte非圧縮点。"""
    priv = ec.generate_private_key(ec.SECP256R1())
    priv_bytes = priv.private_numbers().private_value.to_bytes(32, "big")
    pub_bytes = priv.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return _b64url(priv_bytes), _b64url(pub_bytes)


def main() -> None:
    ap = argparse.ArgumentParser(description="VAPID鍵生成（PWA Web Push）")
    ap.add_argument("--force", action="store_true", help="push_config.jsonを上書き")
    args = ap.parse_args()

    if PUSH_CONFIG.exists() and not args.force:
        try:
            cur = json.loads(PUSH_CONFIG.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
        if cur.get("vapidPublicKey"):
            print("既に push_config.json に公開鍵があります。作り直すと既存の購読は全て無効化されます。")
            print("本当に再生成するなら --force を付けてください。")
            sys.exit(1)

    priv, pub = generate()
    PUSH_CONFIG.write_text(json.dumps(
        {"vapidPublicKey": pub, "subscribeUrl": "./push/subscribe"},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print("VAPID鍵を生成しました。以下を .env に貼ってください（秘密鍵はコミット禁止）:\n")
    print(f"VAPID_PRIVATE_KEY={priv}")
    print(f"VAPID_PUBLIC_KEY={pub}")
    print(f"VAPID_SUBJECT=mailto:あなたのメール")
    print("\n公開鍵は dashboard/push_config.json に書き込みました（フロントが読む）。")
    print("=" * 60)


if __name__ == "__main__":
    main()
