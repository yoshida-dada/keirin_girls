"""確定済みレースへ line_strength を後付けする（data_men.json）。

build_predictions は**確定済みレースの予測を丸ごと据え置く**（発走後にモデルが変わると
事前に出した予測が書き換わり、的中実績が遡って良く見えるため）。そのため機能を足しても
過去レースには反映されない。

line_strength は**公開済みの win_prob だけから再構成できる**:
  三連単分布は win_prob の関数として一意なので、新しい予測ではなく既に出した確率の
  別の見せ方でしかない。→ 予測値を一切変えずに後付けしてよい。

**素のPLで再構成する**（紐補正を掛けない）。対象は A-3 以前に予測された確定済みレースで、
その `top_trifecta` は補正なしで作られているため。ここで補正を掛けると line_strength だけが
表示中の三連単確率と別の分布になる。A-3以降に予測されたレースは predict_race が
補正込みで line_strength を入れているので、この関数は触らない（既存はスキップする）。

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

    b_add, b_skip = refresh_branches(doc)
    print(f"line_strength: 追加 {added} / 既存 {skipped} / 対象外 {nolines}  … 全{len(races)}レース")
    print(f"dev_branches : conditional追加 {b_add} / 既存 {b_skip}")
    if args.dry_run:
        return
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"更新: {p}  {p.stat().st_size/1024/1024:.2f}MB")


def refresh_branches(doc: dict) -> tuple[int, int]:
    """既存 dev_branches に conditional を足し、買い目を現行の係数で組み直す。

    **P(B) は保存済みの値をそのまま使う**（当時の展開AIの出力を尊重する）。再計算するのは
    「その分岐を条件にした着順分布」の形だけで、これは公開済みの win_prob とライン構成から
    一意に決まる＝新しい予測ではなく表示派生。予測本体（win_prob / top_trifecta）には触らない。

    dev_branches は的中実績の集計に使っていないので、係数を直したら全レースで揃える方が
    読み手に一貫する（一部のレースだけ条件付き着順が出ない状態にしない）。
    """
    from src.model.development_branches import (branch_trifecta, conditional_orders,
                                                formation_types, _formation,
                                                FORMATION_BUDGET)
    added = skipped = 0
    for r in (doc.get("predictions") or {}).get("races") or []:
        B = r.get("dev_branches")
        lines = r.get("lines") or []
        if not B or not lines or not B.get("branches"):
            continue
        if B["branches"][0].get("form_types"):
            skipped += 1
            continue
        riders = r.get("riders") or []
        st = {rd["car"]: rd.get("win_prob") or 0.0 for rd in riders if rd.get("car")}
        if not st or abs(sum(st.values()) - 1.0) > 0.02:
            continue
        names = {rd["car"]: rd.get("name") for rd in riders}
        fav = max(st, key=st.get)      # ◎＝モデル1着確率トップ
        for b in B["branches"]:
            dist = branch_trifecta(st, b["b_car"], lines)
            if not dist:
                continue
            b["win"] = {int(c): round(sum(p for (a, _, _), p in dist.items() if a == c), 4)
                        for c in st}
            b["formation"] = _formation(dist, budget=FORMATION_BUDGET)
            b["conditional"] = conditional_orders(dist, lines, names)
            b["form_types"] = formation_types(dist, fav)
        added += 1
    return added, skipped


if __name__ == "__main__":
    main()
