"""既存レースへ万車券率(upset_prob)を後付けする（data_men.json / data.json）。

build_predictions は確定済みレースの予測を辞書ごと据え置くので、新しい指標を足しても
過去レースには入らない。万車券率は**公開済みの win_prob だけから一意に決まる**
（三連単分布は win_prob の関数）ので、予測値を変えずに後付けしてよい。

  python scripts/backfill_upset.py                    # 男子
  python scripts/backfill_upset.py --path dashboard/data.json
  python scripts/backfill_upset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.plackett_luce import all_trifecta_probs
from src.model.upset import man_prob

DEFAULT = ROOT / "dashboard" / "data_men.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="万車券率を後付け")
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="既存の値も計算し直す（しきい値を推定し直した後はこれを使う）")
    args = ap.parse_args()

    p = Path(args.path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    races = (doc.get("predictions") or {}).get("races") or []
    added = skipped = bad = 0
    vals = []
    for r in races:
        if r.get("upset_prob") is not None and not args.force:
            skipped += 1
            continue
        st = {rd["car"]: rd.get("win_prob") or 0.0
              for rd in (r.get("riders") or []) if rd.get("car") is not None}
        # 1着確率が揃っていない（ベースライン推論など）ものは触らない。
        # 揃っていないまま合計すると万車券率が過大に出る
        if len(st) < 3 or abs(sum(st.values()) - 1.0) > 0.02:
            bad += 1
            continue
        u = man_prob(all_trifecta_probs(st), is_girls=bool(r.get("is_girls")),
                     field_size=r.get("field_size") or len(st))
        if u is None:          # 検証を通していない層（男子9車）は出さない
            r.pop("upset_prob", None)      # 旧しきい値で入った値が残らないよう消す
            bad += 1
            continue
        r["upset_prob"] = u
        vals.append(u)
        added += 1

    if vals:
        vals.sort()
        print(f"付与した万車券率: 中央値 {vals[len(vals)//2]*100:.1f}% "
              f"/ 最小 {vals[0]*100:.1f}% / 最大 {vals[-1]*100:.1f}% / 平均 "
              f"{sum(vals)/len(vals)*100:.1f}%")
    print(f"付与 {added} / 既存 {skipped} / 対象外 {bad}  … 全{len(races)}レース")
    if args.dry_run:
        print("dry-run: 書き込みなし")
        return
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"更新: {p}")


if __name__ == "__main__":
    main()
