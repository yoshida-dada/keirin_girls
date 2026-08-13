"""確定済みレースへ line_strength を後付けする（data_men.json）。

build_predictions は**確定済みレースの予測を丸ごと据え置く**（発走後にモデルが変わると
事前に出した予測が書き換わり、的中実績が遡って良く見えるため）。そのため機能を足しても
過去レースには反映されない。

line_strength は**公開済みの win_prob だけから再構成できる**:
  男子は紐補正を掛けず素のPlackett-Luceなので、三連単分布は win_prob の関数として一意。
  つまりこれは新しい予測ではなく、既に出した確率の別の見せ方でしかない。
  → 予測値を一切変えずに後付けしてよい。

  python scripts/backfill_line_strength.py                 # dashboard/data_men.json を更新
  python scripts/backfill_line_strength.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.line_strength import build_lines
from src.model.plackett_luce import all_trifecta_probs

DEFAULT = ROOT / "dashboard" / "data_men.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="確定済みレースへ line_strength を後付け")
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    p = Path(args.path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    races = (doc.get("predictions") or {}).get("races") or []
    added = skipped = nolines = 0
    for r in races:
        if r.get("is_girls") is not False:
            continue
        if r.get("line_strength"):
            skipped += 1
            continue
        lines = r.get("lines") or []
        if not lines:
            nolines += 1
            continue
        riders = r.get("riders") or []
        strengths = {rd["car"]: rd.get("win_prob") or 0.0 for rd in riders if rd.get("car")}
        if not strengths or abs(sum(strengths.values()) - 1.0) > 0.02:
            nolines += 1          # 1着確率が揃っていない（ベースライン推論等）ものは触らない
            continue
        probs = all_trifecta_probs(strengths)      # 男子は素のPL＝公開済み確率から一意に定まる
        r["line_strength"] = build_lines(
            lines, strengths, probs,
            names={rd["car"]: rd.get("name") for rd in riders},
            scores={rd["car"]: rd.get("score") for rd in riders},
            legs={rd["car"]: rd.get("narabi_leg") for rd in riders},
            classes={rd["car"]: rd.get("class_rank") for rd in riders})
        added += 1

    print(f"追加 {added} / 既存 {skipped} / 対象外(ライン無し等) {nolines}  … 全{len(races)}レース")
    if args.dry_run:
        return
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"更新: {p}  {p.stat().st_size/1024/1024:.2f}MB")


if __name__ == "__main__":
    main()
