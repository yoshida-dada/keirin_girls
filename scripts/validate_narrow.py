"""予測的中率10〜30%になるまで買い目を絞ったとき、回収率が最も良い形を探す（男子7車）。

全点均等買いは ROI 58.0%（上限75%）と分かっている。**絞ればどうか**を測る。

**買い目の作り方**: モデルの三連単分布（本番と同じ紐補正込み）を確率の降順に並べ、
累積確率が目標（10/15/20/25/30%）に届くまで買う。こうすると「予測的中率」を
直接指定できる。並べ替えの基準は2通り:
  prob順 … 確率が高い順。当たりやすい順に買う
  EV順  … 確率×オッズ が高い順。同じ予測的中率でも配当の高い目に寄せる
        （※穴狙い検証(4.21.1)では EV順が prob順を30pt下回った。ここでも確認する）

**多重比較の扱い（ここが要）**: 目標5通り × 並べ替え2通り × 層5通り = 50セルを走査する。
95%区間を50回引けば、真にエッジが無くても平均2.5セルは「下限>100%」になる。
過去に単一分割の114%を掴みかけた前例（men_keirin_plan.md 4.15）と同じ罠なので、
**セル数で補正した区間**を主基準に置く。

**事前登録した採否基準（後から緩めない）**:
  主基準: レース単位ブートストラップの **Bonferroni補正済み区間（1-0.05/50）の下限 > 100%**
  副基準: 5foldのうち **4fold以上で単独ROI > 100%**（1本の高配当で持ち上がった形を弾く）
  両方を満たすセルだけ「絞り込みの候補」と呼ぶ。片方だけなら候補としない。

  PYTHONIOENCODING=utf-8 python scripts/validate_narrow.py
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.feature_sets import men_features
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS
from src.model.race_type import classify_race
from src.model.upset import threshold_for
from src.backtest.walkforward import fold_boundaries

STAKE = 100
TARGETS = [0.10, 0.15, 0.20, 0.25, 0.30]
ORDERS = ["prob順", "EV順"]


def load_market(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    odds: dict = defaultdict(dict)
    for rid, combo, o in c.execute("SELECT race_id,combo,odds FROM odds_final_trifecta"):
        odds[rid][combo] = o
    pay = {r: (combo, p) for r, combo, p in
           c.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    nar: dict = defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi"
            " WHERE line_id IS NOT NULL"):
        nar[rid][car] = (li, pi)
    c.close()
    return odds, pay, nar


def pick(probs: dict, odds: dict, target: float, order: str) -> list[str]:
    """累積確率が target に届くまで買う。order で並べ替え基準を変える。"""
    if order == "prob順":
        seq = sorted(probs.items(), key=lambda kv: -kv[1])
    else:
        seq = sorted(probs.items(), key=lambda kv: -(kv[1] * odds.get(kv[0], 0.0)))
    out, cum = [], 0.0
    for k, p in seq:
        out.append(k)
        cum += p
        if cum >= target:
            break
    return out


def boot(rows, alpha, n_boot=3000, seed=0):
    """rows=[(賭け金, 払戻)]。レース単位でリサンプル。alpha は両側の合計。"""
    if not rows:
        return None, None, None
    rnd = random.Random(seed)
    s = sum(a for a, _ in rows)
    point = sum(b for _, b in rows) / s if s else 0.0
    n = len(rows)
    vals = []
    for _ in range(n_boot):
        ss = rr = 0.0
        for _ in range(n):
            a, b = rows[rnd.randrange(n)]
            ss += a; rr += b
        if ss:
            vals.append(rr / ss)
    vals.sort()
    lo = vals[max(0, int(alpha / 2 * len(vals)))]
    hi = vals[min(len(vals) - 1, int((1 - alpha / 2) * len(vals)))]
    return point, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser(description="絞り込み買い目の回収率探索")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    odds_all, pay, nar = load_market(args.db)
    thr = threshold_for(False, 7)
    raw = load_samples(args.db, field_size=[7], features=PL_FEATURES_FULL)
    smp = augment_samples(raw, args.db, men_features())
    print(f"サンプル {len(smp):,}（男子7車）")

    # rows[(目標, 並べ替え, 層)] = [(賭け金, 払戻)] / fold別も持つ
    rows: dict = defaultdict(list)
    frows: dict = defaultdict(lambda: defaultdict(list))
    ups: list[float] = []
    recs = []
    for fi, (a, b, c2) in enumerate(fold_boundaries(len(smp), n_folds=args.folds,
                                                    warmup_frac=0.40, window="expanding")):
        model = train_gbdt(smp[a:b])
        for s in smp[b:c2]:
            if s.race_id not in pay or s.race_id not in odds_all:
                continue
            st = model.strengths(s.X, s.car_numbers)
            if not st:
                continue
            npos = {c: p for c, (_l, p) in (nar.get(s.race_id) or {}).items()}
            pr = corrected_trifecta_probs(st, npos or None, MEN_PARAMS)
            probs = {f"{k[0]}-{k[1]}-{k[2]}": v for k, v in pr.items()}
            up = sum(p for p in probs.values() if p <= thr)
            ups.append(up)
            recs.append((fi, s.race_id, st, probs, up))

    ups.sort()
    q = lambda f: ups[int(f * (len(ups) - 1))]
    hi_cut, lo_cut = q(.70), q(.30)
    print(f"万車券率 30%点 {lo_cut*100:.1f}% / 70%点 {hi_cut*100:.1f}%")

    for fi, rid, st, probs, up in recs:
        rt = classify_race(st).label
        od = odds_all[rid]
        win, yen = pay[rid]
        strata = ["全体", f"型:{rt}"]
        strata.append("万車券率:高" if up >= hi_cut else
                      ("万車券率:低" if up < lo_cut else "万車券率:中"))
        for t in TARGETS:
            for o in ORDERS:
                buy = pick(probs, od, t, o)
                stake = STAKE * len(buy)
                ret = yen if win in buy else 0
                for lab in strata:
                    rows[(t, o, lab)].append((stake, ret))
                    frows[(t, o, lab)][fi].append((stake, ret))

    cells = len(rows)
    alpha = 0.05 / cells                      # Bonferroni
    print(f"\n走査セル数 {cells} → 補正後の区間水準 {(1-alpha)*100:.3f}%")
    print(f"\n{'目標':>5}{'並べ替え':>8}{'層':>12}{'点数':>7}{'予測':>7}{'実測':>7}"
          f"{'ROI':>8}{'補正区間':>22}{'fold勝ち':>8}")
    hits = []
    for (t, o, lab), rr in sorted(rows.items()):
        pts = sum(a for a, _ in rr) / len(rr) / STAKE
        act = sum(1 for _, x in rr if x > 0) / len(rr)
        p, lo, hi = boot(rr, alpha)
        wins = sum(1 for f in frows[(t, o, lab)].values()
                   if sum(x for _, x in f) > sum(a for a, _ in f))
        ok = lo is not None and lo > 1.0 and wins >= 4
        if ok:
            hits.append((t, o, lab, p, lo, hi, wins))
        print(f"{t*100:>4.0f}%{o:>8}{lab:>12}{pts:>7.1f}{t*100:>6.0f}%{act*100:>6.1f}%"
              f"{p*100:>7.1f}%{f'[{lo*100:.1f}–{hi*100:.1f}%]':>22}{wins:>6}/5"
              + ("  ★" if ok else ""))

    print("\n事前基準（補正区間の下限>100% かつ 4/5fold以上でROI>100%）を満たすセル: "
          f"{len(hits)}")
    for h in hits:
        print(f"  ★ {h[0]*100:.0f}% {h[1]} {h[2]}  ROI {h[3]*100:.1f}% "
              f"[{h[4]*100:.1f}–{h[5]*100:.1f}%] {h[6]}/5fold")
    if not hits:
        print("  → 絞り込みでも黒字ゾーンは出ない。控除率25%（上限75%）を越えられていない。")


if __name__ == "__main__":
    main()
