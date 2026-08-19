"""買い目パッケージ検証: 三連単K点 + 三連複2点 の的中頻度・回収率を実測(男子)。

黒字ゾーンは無い(全戦略<100%)前提で、「三連単(高配当の上振れ) + 三連複2点(的中保険)」の
現実的パッケージのリターンプロフィールを測る。ROIは券種平均で~72%だが、三連複2点で
「何か当たる率」がどれだけ上がるか、種別(勝ち上がり)別にどう組むのが良いかを見る。

  PYTHONIOENCODING=utf-8 python scripts/analyze_buy_package.py
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
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.rider_narabi import compute_narabi_features
from src.model.feature_sets import men_features
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.backtest.walkforward import fold_boundaries


def _role(name):
    if not name:
        return "他"
    for kw, lab in [("準決", "堅"), ("決勝", "堅"), ("予選", "標"),
                    ("選抜", "荒"), ("一般", "荒"), ("特選", "標")]:
        if kw in name:
            return lab
    return "標"


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    role = {rid: _role(nm) for rid, nm in c.execute("SELECT race_id,race_name FROM races")}
    tri, trio = {}, {}
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        tri[rid] = (tuple(int(x) for x in combo.split("-")), pay)
    for rid, combo, pay in c.execute("SELECT race_id,combo,payout FROM payouts_trio"):
        trio[rid] = (frozenset(int(x) for x in combo.replace("=", "-").split("-")), pay)
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
    return role, tri, trio, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    role, tri, trio, lines = _load(args.db)

    base = load_samples(args.db, field_size=7, features=PL_FEATURES_FULL)
    samples = augment_samples(base, args.db, men_features())
    narabi = compute_narabi_features(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    recs = []
    for a, b, c in bounds:
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            if s.race_id not in tri or s.race_id not in trio:
                continue
            st = model.strengths(s.X, s.car_numbers)
            npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
            dist = corrected_trifecta_probs(st, npos, MEN_PARAMS, lines=lines.get(s.race_id))
            tri_rank = [o for o, _ in sorted(dist.items(), key=lambda kv: -kv[1])]
            trp = defaultdict(float)
            for (x, y, z), p in dist.items():
                trp[frozenset((x, y, z))] += p
            trio_rank = [o for o, _ in sorted(trp.items(), key=lambda kv: -kv[1])]
            recs.append({"role": role.get(s.race_id, "標"), "tri_rank": tri_rank,
                         "trio_rank": trio_rank, "tri": tri[s.race_id], "trio": trio[s.race_id]})
    print(f"男子7車・三連単/三連複あり {len(recs)}レース\n")

    def pkg(rs, kt, kc):
        n = len(rs)
        st = kt * 100 + kc * 100
        ret = th = ch = any_h = 0
        for r in rs:
            trib = set(r["tri_rank"][:kt]); trib_hit = r["tri"][0] in trib
            triob = set(r["trio_rank"][:kc]); triob_hit = r["trio"][0] in triob
            ret += (r["tri"][1] if trib_hit else 0) + (r["trio"][1] if triob_hit else 0)
            th += trib_hit; ch += triob_hit; any_h += (trib_hit or triob_hit)
        return (ret/(st*n)*100, th/n*100, ch/n*100, any_h/n*100)

    def show(rs, label):
        if len(rs) < 100:
            return
        print(f"■ {label}  n={len(rs)}")
        print(f"   {'パッケージ':<22}{'回収率':>8}{'三連単的中':>10}{'三連複的中':>10}{'何か当たる':>10}")
        for kt, kc in [(6, 0), (6, 2), (8, 2), (10, 2), (12, 2)]:
            roi, th, ch, an = pkg(rs, kt, kc)
            lbl = f"三連単{kt}" + (f"+三連複{kc}" if kc else "")
            print(f"   {lbl:<22}{roi:>7.1f}%{th:>9.1f}%{ch:>9.1f}%{an:>9.1f}%")
        print()

    show(recs, "全体")
    show([r for r in recs if r["role"] == "堅"], "堅い戦(準決勝/決勝)")
    show([r for r in recs if r["role"] == "荒"], "荒れ戦(選抜/一般)")
    show([r for r in recs if r["role"] == "標"], "標準(予選/特選)")


if __name__ == "__main__":
    main()
