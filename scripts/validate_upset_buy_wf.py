"""【B/C】荒れ帯(万車券率≥30%)の買い目候補を as-of walk-forward で比較（男子7車）。

Aで現行買い目の荒れ帯的中率が崖(≥30%で30-37%)と確定。ここは荒れ帯だけを対象に、
現行(build_branches merged) vs 穴に振った候補（EVでなくmix分布で widen / 人気薄先行頭）を
的中率・点数・回収率(真値)で比較する。各foldで着順モデルを再学習(as-of, リーク排除)。
backstretch(P(B))は本番を流用(分岐混合の重み・影響小)。判定: 1点あたりの的中増と回収率で選ぶ。

  PYTHONIOENCODING=utf-8 python scripts/validate_upset_buy_wf.py --folds 5 --thr 0.30
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
from src.model.upset import man_prob
from src.backtest.walkforward import fold_boundaries

DB = str(DATA_DIR / "keirin_men.sqlite")


def _load():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    tri = {rid: (tuple(int(x) for x in cb.split("-")), p)
           for rid, cb, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    leg = {(rid, car): lt for rid, car, lt in
           c.execute("SELECT race_id,car_number,leg_type FROM entries")}
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
    return tri, leg, lines


def _merged(br):
    cs = set()
    for f in (br or {}).get("merged", {}).get("forms", []):
        cs |= {(a, b, c) for a in f["first"] for b in f["second"] for c in f["third"]
               if len({a, b, c}) == 3}
    return cs


def _candidates(current, mix_ranked, st, ln, leg, rid):
    """荒れ用の買い目候補（EVでなくモデル確率/mixで振る）。"""
    def fill(base, extra, n):
        out = list(base)
        for ab in extra:
            if ab not in out:
                out.append(ab)
            if len(out) >= n:
                break
        return set(out)
    cur = set(current)
    out = {"現行": cur, "mix18": set(mix_ranked[:18]), "mix24": set(mix_ranked[:24]),
           "現行∪mix+8": fill(cur, mix_ranked, len(cur) + 8),
           "現行∪mix+14": fill(cur, mix_ranked, len(cur) + 14)}
    # 人気薄の別ライン先行(逃/両)を1着に追加（ハイ荒れの主決着）。model勝率が低い逃頭を狙う
    heads = [c for c in st if leg.get((rid, c)) in ("逃", "両")]
    heads = sorted(heads, key=lambda c: st.get(c, 0))[:2]      # 勝率下位2の先行
    tops = [c for c in sorted(st, key=lambda c: -st.get(c, 0))][:4]
    add = set()
    for h in heads:
        for b in tops:
            for cc in tops:
                if len({h, b, cc}) == 3:
                    add.add((h, b, cc))
    out["現行∪人気薄先行頭"] = cur | add
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--thr", type=float, default=0.30)      # 荒れ帯のしきい(万車券率)
    args = ap.parse_args()
    tri, leg, lines = _load()
    model, _, _ = load_for(False)
    bs = load_backstretch(is_girls=False)
    base = load_samples(DB, field_size=[7], features=PL_FEATURES_FULL)
    samples = [s for s in augment_samples(base, DB, model.feature_names)
               if s.race_id in tri and s.race_id in lines]
    samples.sort(key=lambda s: (s.date, s.race_id))
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"男子7車 as-of walk-forward {len(bounds)}fold  荒れ帯 万車券率≥{args.thr:.0%} のみ比較\n")

    names = ["現行", "mix18", "mix24", "現行∪mix+8", "現行∪mix+14", "現行∪人気薄先行頭"]
    agg = {nm: {"n": 0, "hit": 0, "pts": 0, "ret": 0} for nm in names}
    n_arare = 0
    for a, b, c in bounds:
        m = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            rid = s.race_id
            st = m.strengths(s.X, s.car_numbers)
            pB = bs.strengths(s.X, s.car_numbers) if bs else None
            ln = lines[rid]
            mix, _ = branch_mixture(st, ln, pB)
            probs = mix or corrected_trifecta_probs(st, {}, MEN_PARAMS, lines=ln)
            u = man_prob(probs, is_girls=False, field_size=7)
            if u is None or u < args.thr:
                continue                       # 荒れ帯のみ
            n_arare += 1
            current = _merged(build_branches(st, ln, pB))
            mix_ranked = [k for k, _ in sorted((mix or probs).items(), key=lambda kv: -kv[1])]
            combo, pay = tri[rid]; combo = tuple(combo)
            for nm, picks in _candidates(current, mix_ranked, st, ln, leg, rid).items():
                d = agg[nm]
                d["n"] += 1; d["pts"] += len(picks)
                if combo in picks:
                    d["hit"] += 1; d["ret"] += pay

    print(f"荒れ帯レース数(as-of): {n_arare}\n")
    print(f"{'候補':<18}{'平均点数':>9}{'的中率':>9}{'回収率':>9}{'現行比的中':>11}{'的中増/+1点':>12}")
    base_hit = agg["現行"]["hit"] / agg["現行"]["n"] * 100 if agg["現行"]["n"] else 0
    base_pts = agg["現行"]["pts"] / agg["現行"]["n"] if agg["現行"]["n"] else 0
    for nm in names:
        d = agg[nm]
        if d["n"] == 0:
            continue
        n = d["n"]
        hit = d["hit"] / n * 100
        pts = d["pts"] / n
        roi = d["ret"] / (d["pts"] * 100) * 100 if d["pts"] else 0
        dhit = hit - base_hit
        per = (dhit / (pts - base_pts)) if (pts - base_pts) > 0.01 else 0
        print(f"{nm:<18}{pts:>8.1f}{hit:>8.1f}%{roi:>8.1f}%{dhit:>+10.1f}pt{per:>+11.2f}")
    print("\n※現行比で的中率が上がり、かつ回収率が大きく落ちない候補が有効。"
          "『的中増/+1点』が高い＝1点あたり効率よく荒れを拾える。回収率は真値(as-of)。")


if __name__ == "__main__":
    main()
