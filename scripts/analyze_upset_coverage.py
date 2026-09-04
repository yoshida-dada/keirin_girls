"""【A】現行買い目(build_branches統合)の的中率を万車券率帯別に診断（男子7車）。

ユーザー観察「万車券率≤20%はカバー、≥30%は取りこぼし」を数値で確定する。本番モデルで
現行と同じ統合買い目(merged)を再現し、実払戻(payouts_trifecta)で決済。万車券率(man_prob,
較正済み)で帯別に 的中率/点数/回収率 を集計。これが荒れ用買い目の基準値になる。
※本番モデル(in-sample)＝現行システムが出す買い目の再現。候補比較(C)はas-of walk-forwardで行う。

  PYTHONIOENCODING=utf-8 python scripts/analyze_upset_coverage.py
"""
from __future__ import annotations

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
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.development_branches import build_branches, branch_mixture
from src.model.upset import man_prob

DB = str(DATA_DIR / "keirin_men.sqlite")


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


def _bucket(u):
    if u is None:
        return None
    if u < 0.15:
        return "①<15%"
    if u < 0.20:
        return "②15-20%"
    if u < 0.25:
        return "③20-25%"
    if u < 0.30:
        return "④25-30%"
    if u < 0.35:
        return "⑤30-35%"
    return "⑥≥35%"


ORDER = ["①<15%", "②15-20%", "③20-25%", "④25-30%", "⑤30-35%", "⑥≥35%"]


def main():
    tri, lines = _load()
    model, _, _ = load_for(False)
    bs = load_backstretch(is_girls=False)
    base = load_samples(DB, field_size=[7], features=PL_FEATURES_FULL)
    samples = [s for s in augment_samples(base, DB, model.feature_names)
               if s.race_id in tri and s.race_id in lines]
    print(f"男子7車 {len(samples)}レース（本番モデルで現行買い目を再現・実払戻で決済）\n")

    agg = defaultdict(lambda: {"n": 0, "hit": 0, "pts": 0, "ret": 0, "u": 0.0})
    for s in samples:
        rid = s.race_id
        st = model.strengths(s.X, s.car_numbers)
        pB = bs.strengths(s.X, s.car_numbers) if bs else None
        ln = lines[rid]
        mix, _ = branch_mixture(st, ln, pB)
        probs = mix or corrected_trifecta_probs(st, {}, MEN_PARAMS, lines=ln)
        u = man_prob(probs, is_girls=False, field_size=7)
        bk = _bucket(u)
        if bk is None:
            continue
        merged = _merged(build_branches(st, ln, pB))
        if not merged:
            continue
        combo, pay = tri[rid]
        hit = tuple(combo) in merged
        d = agg[bk]
        d["n"] += 1; d["hit"] += int(hit); d["pts"] += len(merged)
        d["ret"] += pay if hit else 0; d["u"] += u

    print(f"{'万車券率帯':<10}{'レース数':>8}{'平均万車率':>10}{'現行的中率':>10}{'平均点数':>9}{'回収率':>8}")
    tot = defaultdict(int)
    for bk in ORDER:
        d = agg.get(bk)
        if not d or d["n"] == 0:
            continue
        n = d["n"]
        roi = d["ret"] / (d["pts"] * 100) * 100 if d["pts"] else 0
        print(f"{bk:<10}{n:>8}{d['u']/n*100:>9.1f}%{d['hit']/n*100:>9.1f}%"
              f"{d['pts']/n:>9.1f}{roi:>7.1f}%")
        for k in ("n", "hit", "pts", "ret"):
            tot[k] += d[k]
    if tot["n"]:
        roi = tot["ret"] / (tot["pts"] * 100) * 100
        print(f"{'全体':<10}{tot['n']:>8}{'':>10}{tot['hit']/tot['n']*100:>9.1f}%"
              f"{tot['pts']/tot['n']:>9.1f}{roi:>7.1f}%")
    print("\n※荒れ帯(≥30%)で的中率が急落＝現行買い目が荒れ決着を拾えていない、を確認。"
          "これが荒れ用買い目(B/C)の改善対象・基準カバー率。")


if __name__ == "__main__":
    main()
