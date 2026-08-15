"""確定済みレースへ競り(seri)を後付けする（data_men.json）。

build_predictions は**確定済みレースの予測を丸ごと据え置く**（発走後にモデルが変わると
事前に出した予測が書き換わり、的中実績が遡って良く見えるため）。そのため競り対応を
入れても過去レースには反映されず、西武園2R が5車の直列ラインのまま表示されていた。

競りは **DBの narabi.seri_group から引くだけ**で、予測値には一切触らない:
  ・`seri` を race に足す（表示のカッコ用）
  ・line_strength の cars の pos / pos_label / seri / seri_side を引き直す
  ・p_win / p_12 / p_top3_any / settle_prob は**そのまま**。競りはライン内の位置の
    解釈を変えるだけで、どのラインに属するかは変えないため、これらは影響を受けない。
→ 公開済みの確率を書き換えないので、的中実績が遡って変わることはない。

  python scripts/backfill_seri_json.py
  python scripts/backfill_seri_json.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR
from src.model.line_strength import seri_sides, _pos_label
from src.features.line_features import positions_with_seri

DEFAULT = ROOT / "dashboard" / "data_men.json"


def load_seri(db_path) -> dict[tuple, list[list[int]]]:
    """(race_date, venue_code, race_no) → [[車番,...], ...]。

    グループ内は narabi.position 順＝並び予想の内→外の順に並べる（seri_sides がこの順を使う）。
    """
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    rows = c.execute(
        "SELECT r.race_date, r.venue_code, r.race_no, n.seri_group, n.car_number"
        "  FROM narabi n JOIN races r ON r.race_id = n.race_id"
        " WHERE n.seri_group IS NOT NULL"
        " ORDER BY r.race_date, r.venue_code, r.race_no, n.seri_group, n.position").fetchall()
    c.close()
    out: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for d, v, no, g, car in rows:
        out[(d, str(v), int(no))][g].append(car)
    return {k: [v[g] for g in sorted(v)] for k, v in out.items()}


def patch_line_strength(ls: dict, lines: list, seri: list) -> int:
    """line_strength の位置ラベルだけを競り込みで引き直す。確率は触らない。"""
    if not ls:
        return 0
    side = seri_sides(lines, seri)
    inseri = {c for g in seri for c in g}
    n = 0
    by_id = {L.get("line_id"): L for L in ls.get("lines") or []}
    for li, mem in enumerate(lines):
        L = by_id.get(li)
        if not L:
            continue
        gs = [g for g in seri if all(c in mem for c in g)]
        pos = positions_with_seri(mem, gs)
        for car in L.get("cars") or []:
            c = car.get("car")
            if c not in pos:
                continue
            car["pos"] = pos[c]
            car["pos_label"] = _pos_label(pos[c], len(mem))
            car["seri"] = c in inseri
            car["seri_side"] = side.get(c)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="確定済みレースへ競りを後付け")
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seri_map = load_seri(args.db)
    print(f"DBに競りを持つレース: {len(seri_map):,}")

    p = Path(args.path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    races = (doc.get("predictions") or {}).get("races") or []
    filled = already = nomatch = nokey = 0
    for r in races:
        key = (r.get("date"), str(r.get("venue_code")), r.get("race_no"))
        if None in key:
            nokey += 1
            continue
        if r.get("seri"):
            already += 1
            continue
        s = seri_map.get(key)
        if not s:
            # DBに競りが無い＝競りのないレース。空配列を入れて「確認済み」にする
            # （キー欠落と「競り無し」を表示側で区別できるようにする）
            r.setdefault("seri", [])
            nomatch += 1
            continue
        lines = r.get("lines") or []
        # ラインに含まれない車番が混じる競りは扱わない（欠場等でズレている）
        s = [g for g in s if any(all(c in mem for c in g) for mem in lines)]
        if not s:
            r.setdefault("seri", [])
            nomatch += 1
            continue
        r["seri"] = s
        cars = patch_line_strength(r.get("line_strength"), lines, s)
        filled += 1
        print(f"  {r['date']} {r.get('venue')}{r.get('race_no')}R  競り={s}  "
              f"（{cars}名の位置を引き直し）")

    print(f"\n競り付与 {filled} / 既存 {already} / 競り無し {nomatch} / キー不足 {nokey}"
          f"  … 全{len(races)}レース")
    if args.dry_run:
        print("dry-run: 書き込みなし")
        return
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    print(f"更新: {p}")


if __name__ == "__main__":
    main()
