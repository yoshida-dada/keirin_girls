"""穴フォーメーション(1着-2着-3着, mix各着マージナル上位)の万車券カバーを検証（男子7車, as-of）。

ユーザー希望のフォーメーション記法で穴を出すため、mixの各着マージナル上位から矩形フォーメーションを
作り、検証済み mix24(joint上位24) とカバー率が同等かを確認。1ヶ月前まで学習→直近1ヶ月をas-of。

  PYTHONIOENCODING=utf-8 python scripts/validate_arare_formation.py --days 30
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.model.backstretch import load_backstretch
from src.model.train_gbdt import train_gbdt
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.development_branches import build_branches, branch_mixture

DB = str(DATA_DIR / "keirin_men.sqlite")
MAN = 10000


def _load():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    tri = {rid: (tuple(int(x) for x in cb.split("-")), p)
           for rid, cb, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    lines = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
    return tri, lines


def _merged(br):
    cs = set()
    for f in (br or {}).get("merged", {}).get("forms", []):
        cs |= {(a, b, c) for a in f["first"] for b in f["second"] for c in f["third"]
               if len({a, b, c}) == 3}
    return cs


def _formation(probs, n1, n2, n3):
    p1, p2, p3 = defaultdict(float), defaultdict(float), defaultdict(float)
    for (a, b, c), p in probs.items():
        p1[a] += p; p2[b] += p; p3[c] += p
    A = [c for c, _ in sorted(p1.items(), key=lambda kv: -kv[1])[:n1]]
    B = [c for c, _ in sorted(p2.items(), key=lambda kv: -kv[1])[:n2]]
    C = [c for c, _ in sorted(p3.items(), key=lambda kv: -kv[1])[:n3]]
    combos = {(a, b, c) for a in A for b in B for c in C if len({a, b, c}) == 3}
    return combos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    tri, lines = _load()
    model, _, _ = load_for(False)
    bs = load_backstretch(is_girls=False)
    base = load_samples(DB, field_size=[7], features=PL_FEATURES_FULL)
    samples = [s for s in augment_samples(base, DB, model.feature_names)
               if s.race_id in tri and s.race_id in lines]
    samples.sort(key=lambda s: (s.date, s.race_id))
    days = sorted({s.date for s in samples})
    tset = set(days[-args.days:])
    train = [s for s in samples if s.date not in tset]
    test = [s for s in samples if s.date in tset]
    print(f"男子7車 as-of 学習{len(train)}R → 検証直近{args.days}日 {len(test)}R\n")
    m = train_gbdt(train)

    forms = {"F 3-3-5": (3, 3, 5), "F 3-4-4": (3, 4, 4), "F 2-4-5": (2, 4, 5), "F 3-4-5": (3, 4, 5)}
    strat = ["現行", "mix24"] + list(forms)
    agg = {k: {"pts": 0, "hit": 0, "ret": 0, "man": 0} for k in strat}
    n_man = 0
    for s in test:
        rid = s.race_id
        st = m.strengths(s.X, s.car_numbers)
        pB = bs.strengths(s.X, s.car_numbers) if bs else None
        ln = lines[rid]
        mix, _ = branch_mixture(st, ln, pB)
        probs = mix or corrected_trifecta_probs(st, {}, MEN_PARAMS, lines=ln)
        mixr = [k for k, _ in sorted(probs.items(), key=lambda kv: -kv[1])]
        buys = {"現行": _merged(build_branches(st, ln, pB)), "mix24": set(mixr[:24])}
        for nm, (n1, n2, n3) in forms.items():
            buys[nm] = _formation(probs, n1, n2, n3)
        combo, pay = tri[rid]; combo = tuple(combo)
        is_man = pay >= MAN
        if is_man:
            n_man += 1
        for k in strat:
            b = buys[k]
            agg[k]["pts"] += len(b)
            if combo in b:
                agg[k]["hit"] += 1; agg[k]["ret"] += pay
                if is_man:
                    agg[k]["man"] += 1
    N = len(test)
    print(f"検証{N}レース（万車券{n_man}本）\n")
    print(f"{'買い目':<12}{'平均点数':>9}{'的中率':>9}{'回収率':>9}{'万車券カバー':>14}")
    for k in strat:
        d = agg[k]
        roi = d["ret"] / (d["pts"] * 100) * 100 if d["pts"] else 0
        print(f"{k:<12}{d['pts']/N:>8.1f}{d['hit']/N*100:>8.1f}%{roi:>8.1f}%"
              f"{d['man']:>7}/{n_man} ({d['man']/n_man*100:.0f}%)")
    print("\n※フォーメーションの万車券カバーが mix24 と同等なら、その形で表示採用。")


if __name__ == "__main__":
    main()
