"""穴買い目: 単一矩形(3-4-4) vs 複数フォーメーション(頭ごとに紐最適化・3着広げ) の比較（男子7車, as-of）。

矩形=全頭で同じ2着/3着。複数=各1着頭ごとにjoint mixで2着/3着を選び直しunion(数パターンの2-3-4等)。
どちらが万車券カバー効率(カバー/点)が良いかを1ヶ月as-oで測る。

  PYTHONIOENCODING=utf-8 python scripts/validate_arare_multi.py --days 30
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
from src.model.development_branches import branch_mixture

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


def _rect(probs, n1, n2, n3):
    p1, p2, p3 = defaultdict(float), defaultdict(float), defaultdict(float)
    for (a, b, c), p in probs.items():
        p1[a] += p; p2[b] += p; p3[c] += p
    A = [c for c, _ in sorted(p1.items(), key=lambda kv: -kv[1])[:n1]]
    B = [c for c, _ in sorted(p2.items(), key=lambda kv: -kv[1])[:n2]]
    C = [c for c, _ in sorted(p3.items(), key=lambda kv: -kv[1])[:n3]]
    return {(a, b, c) for a in A for b in B for c in C if len({a, b, c}) == 3}


def _multi(probs, nh, n2, n3):
    """各1着頭ごとにjoint mixで2着/3着を選び直したフォーメーションをunion（数パターン）。"""
    p1 = defaultdict(float)
    for (a, b, c), p in probs.items():
        p1[a] += p
    heads = [c for c, _ in sorted(p1.items(), key=lambda kv: -kv[1])[:nh]]
    combos = set()
    for h in heads:
        p2, p3 = defaultdict(float), defaultdict(float)
        for (a, b, c), p in probs.items():
            if a == h:
                p2[b] += p; p3[c] += p
        B = [c for c, _ in sorted(p2.items(), key=lambda kv: -kv[1])[:n2]]
        C = [c for c, _ in sorted(p3.items(), key=lambda kv: -kv[1])[:n3]]
        combos |= {(h, b, c) for b in B for c in C if len({h, b, c}) == 3}
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

    cand = {
        "矩形3-4-4": lambda pr: _rect(pr, 3, 4, 4),
        "矩形3-4-5(3着広)": lambda pr: _rect(pr, 3, 4, 5),
        "複数 頭3×2-3-4": lambda pr: _multi(pr, 3, 3, 4),
        "複数 頭3×2-2-5(3着広)": lambda pr: _multi(pr, 3, 2, 5),
        "複数 頭2×2-3-5(3着広)": lambda pr: _multi(pr, 2, 3, 5),
    }
    strat = list(cand)
    agg = {k: {"pts": 0, "hit": 0, "ret": 0, "man": 0} for k in strat}
    n_man = 0
    for s in test:
        rid = s.race_id
        st = m.strengths(s.X, s.car_numbers)
        pB = bs.strengths(s.X, s.car_numbers) if bs else None
        ln = lines[rid]
        mix, _ = branch_mixture(st, ln, pB)
        probs = mix or corrected_trifecta_probs(st, {}, MEN_PARAMS, lines=ln)
        combo, pay = tri[rid]; combo = tuple(combo)
        is_man = pay >= MAN
        if is_man:
            n_man += 1
        for k in strat:
            b = cand[k](probs)
            agg[k]["pts"] += len(b)
            if combo in b:
                agg[k]["hit"] += 1; agg[k]["ret"] += pay
                if is_man:
                    agg[k]["man"] += 1
    N = len(test)
    print(f"検証{N}レース（万車券{n_man}本）\n")
    print(f"{'買い目':<22}{'平均点数':>9}{'的中率':>9}{'回収率':>9}{'万車券カバー':>14}{'カバー/10点':>11}")
    for k in strat:
        d = agg[k]
        roi = d["ret"] / (d["pts"] * 100) * 100 if d["pts"] else 0
        eff = d["man"] / (d["pts"] / N) / 10 if d["pts"] else 0   # 万車券カバー本数/(10点)
        print(f"{k:<22}{d['pts']/N:>8.1f}{d['hit']/N*100:>8.1f}%{roi:>8.1f}%"
              f"{d['man']:>7}/{n_man} ({d['man']/n_man*100:.0f}%){eff:>10.1f}")
    print("\n※カバー/10点=万車券を10点あたり何本拾うか(効率)。矩形と複数でどちらが効率的かを比較。")


if __name__ == "__main__":
    main()
