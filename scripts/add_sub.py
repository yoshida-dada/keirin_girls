"""スマホで作成した Web Push 購読情報を data/push_subs.json に登録する。

GitHub Pages(静的)からダッシュボードを開いて「🔔 通知」を有効化すると、購読情報の
JSON文字列が表示される（クリップボードにもコピーされる）。それをPCに送り、この
スクリプトに渡すと通知先として登録される。

  python scripts/add_sub.py '{"endpoint":"https://...","keys":{...}}'
  echo '<購読JSON>' | python scripts/add_sub.py     # 標準入力でも可
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.notify.webpush import add_subscription, SUBS_PATH


def main() -> None:
    raw = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not raw:
        print("購読JSONを引数か標準入力で渡してください。")
        sys.exit(1)
    try:
        sub = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"JSONとして読めません: {e}")
        sys.exit(1)
    if not sub.get("endpoint"):
        print("endpoint がありません。購読JSONを確認してください。")
        sys.exit(1)
    n = add_subscription(sub)
    print(f"登録しました。現在の購読数: {n}（{SUBS_PATH}）")


if __name__ == "__main__":
    main()
