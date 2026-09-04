"""過去1ヶ月・全レースで 現行 / 効率(mix18) / バランス(mix24) の的中と万車券カバーを比較（男子7車）。

万車券率ゲート無し=全レースにmixで広げた買い目を適用。リーク回避のため1ヶ月前までで着順モデルを
学習し直近1ヶ月をas-of予測。万車券(払戻≥1万円)を各買い目が何本カバーできるかを主眼に集計。

  PYTHONIOENCODING=utf-8 python scripts/validate_month_coverage.py --days 30
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
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
MAN = 10000       # 万車券のしきい(円)
EFF, BAL = 18, 24


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
    test_days = set(days[-args.days:])
    train = [s for s in samples if s.date not in test_days]
    test = [s for s in samples if s.date in test_days]
    print(f"男子7車 as-of: 学習{len(train)}R(〜{days[-args.days-1]}) → 検証 直近{args.days}日"
          f"({days[-args.days]}〜{days[-1]}) {len(test)}R\n")
    m = train_gbdt(train)

    strat = ["現行", "効率mix18", "バランスmix24"]
    agg = {k: {"pts": 0, "hit": 0, "ret": 0} for k in strat}
    man = {k: 0 for k in strat}     # 万車券カバー数
    n_man = 0
    for s in test:
        rid = s.race_id
        st = m.strengths(s.X, s.car_numbers)
        pB = bs.strengths(s.X, s.car_numbers) if bs else None
        ln = lines[rid]
        mix, _ = branch_mixture(st, ln, pB)
        probs = mix or corrected_trifecta_probs(st, {}, MEN_PARAMS, lines=ln)
        mixr = [k for k, _ in sorted(probs.items(), key=lambda kv: -kv[1])]
        current = _merged(build_branches(st, ln, pB))
        buys = {"現行": current, "効率mix18": set(mixr[:EFF]), "バランスmix24": set(mixr[:BAL])}
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
                    man[k] += 1

    N = len(test)
    print(f"検証{N}レース（うち万車券 {n_man}本 = {n_man/N*100:.1f}%）\n")
    print(f"{'戦略':<16}{'平均点数':>9}{'全体的中率':>10}{'回収率':>9}{'万車券カバー':>14}")
    for k in strat:
        d = agg[k]
        roi = d["ret"] / (d["pts"] * 100) * 100 if d["pts"] else 0
        print(f"{k:<16}{d['pts']/N:>8.1f}{d['hit']/N*100:>9.1f}%{roi:>8.1f}%"
              f"{man[k]:>7}/{n_man} ({man[k]/n_man*100:.0f}%)")
    print("\n※万車券カバー=払戻≥1万円のレースのうち買い目が的中した本数。"
          "全体回収率は落ちても万車券カバーが増えれば『荒れを拾う』目的は達成。")


if __name__ == "__main__":
    main()
