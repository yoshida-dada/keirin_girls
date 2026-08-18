"""既存レースへ万車券率(upset_prob)を後付けする（data_men.json / data.json）。

build_predictions は確定済みレースの予測を辞書ごと据え置くので、新しい指標を足しても
過去レースには入らない。ここで後付けする。

**本番と同じ分布を可能な限り再現してから計算する**。しきい値は本番の分布
（男子＝分岐混合）に合わせて推定してあるので、別の分布に当てると値がずれる。
  ・lines と dev_branches があるレース → **分岐混合を再構成**（保存値は4桁丸めだが誤差は小）
  ・無いレース（ガールズ・並び予想なし）→ 素のPL
以前は無条件に素のPLで組み直しており、混合で出した値を --force で上書きして
中央値2.23pt・最大11.68pt ずらした（2026-08-18）。

既定では upset_prob がある行は触らない。--force は**再構成した分布で計算し直す**
（壊れた値の修復用）。

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
from src.model.development_branches import branch_mixture
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
    kinds: dict = {}
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
        # 本番と同じ分布を再現する。分岐混合で出したレースは混合を組み直す
        # （素のPLで代用すると別の分布に別のしきいを当てることになる）
        dist, kind = all_trifecta_probs(st), "PL"
        lines, dv = r.get("lines"), r.get("dev_branches")
        if lines and dv and (dv.get("branches") or []):
            pB = {b["b_car"]: b["prob"] for b in dv["branches"] if b.get("b_car") is not None}
            mix, _ = branch_mixture(st, lines, pB)
            if mix:
                dist, kind = mix, "混合"
        kinds[kind] = kinds.get(kind, 0) + 1
        u = man_prob(dist, is_girls=bool(r.get("is_girls")),
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
    if kinds:
        print("  分布の内訳: " + " / ".join(f"{k} {v}R" for k, v in sorted(kinds.items())))
    if args.dry_run:
        print("dry-run: 書き込みなし")
        return
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"更新: {p}")


if __name__ == "__main__":
    main()
