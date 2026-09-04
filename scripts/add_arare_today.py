"""本日の全レース(完了含む)に穴フォーメーション(F3-4-5)を注入する一回限りスクリプト。

refresh_predictions は締切5分超過の完了レースをスキップするため、完了レースに arare が付かない。
再フェッチ不要: 各レースに保存済みの combos([a,b,c,odds,prob,ev]) の prob(本番分布)から
各着マージナル上位3-4-5を再計算して arare を注入。result 等は保持(上書きしない)。

  PYTHONIOENCODING=utf-8 python scripts/add_arare_today.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASH = ROOT / "dashboard"


def arare_form(combos, n1=3, n2=4, n3=5):
    probs = {(int(c[0]), int(c[1]), int(c[2])): c[4]
             for c in (combos or []) if len(c) >= 5 and c[4] is not None}
    if not probs:
        return None
    p1, p2, p3 = defaultdict(float), defaultdict(float), defaultdict(float)
    for (a, b, c), p in probs.items():
        p1[a] += p; p2[b] += p; p3[c] += p
    A = [c for c, _ in sorted(p1.items(), key=lambda kv: -kv[1])[:n1]]
    B = [c for c, _ in sorted(p2.items(), key=lambda kv: -kv[1])[:n2]]
    C = [c for c, _ in sorted(p3.items(), key=lambda kv: -kv[1])[:n3]]
    cs = [(a, b, c) for a in A for b in B for c in C if len({a, b, c}) == 3]
    return {"first": A, "second": B, "third": C, "points": len(cs)} if cs else None


def main():
    today = date.today().isoformat()
    for path in (DASH / "data.json", DASH / "data_men.json"):
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        rs = doc.get("predictions", {}).get("races", [])
        n = done = 0
        for r in rs:
            if r.get("date") != today:
                continue
            n += 1
            a = arare_form(r.get("combos"))
            if a:
                r["arare"] = a
                done += 1
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.name}: 本日{n}R 穴付与{done}R")


if __name__ == "__main__":
    main()
