"""買い目の検証: 型別（◎頭/◎2着/◎3着/◎抜き）の予測が実測と合うかを walk-forward で見る。

**◎を1着に固定する前提を検証する。** 男子の◎1着的中は43%＝**57%は◎が勝たない**ので、
◎頭だけ出すのは買い方として偏っている。◎を2着・3着に置く型、◎を外す型も併せて出し、
それぞれについて「予測した確率」と「実際に起きた割合」を突き合わせる。

見る2つ:
  1. **型の出現確率**の較正 … P(◎1着) / P(◎2着) / P(◎3着) / P(◎3着圏外)
     ここがずれていれば型の選び方そのものが誤る
  2. **買い目のカバー率**の較正 … その型が起きたとき、組んだ買い目が実際に当たった割合
     予測カバーより実測が低ければ、買い目が絵に描いた餅になる

**本番と同じ条件で測る**: 主導権は展開AIの予測を使い、真のBは使わない。
分岐の混合分布 Σ_b P(B=b)·P(順位|B=b) から型を組む＝実際に賭けるときと同じ情報量。

**事前登録した採否基準（後から緩めない）**:
  主基準: 型の出現確率の平均絶対誤差が **3pt以内**
  副基準: 各型のカバー率の誤差が **5pt以内**（|予測−実測|）
  外れたら**買い目として出さない**。当たらない買い目を並べる方が無いより悪い。

  PYTHONIOENCODING=utf-8 python scripts/validate_formations.py
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
from src.model.feature_sets import men_features
from src.model.development_branches import branch_trifecta, formation_types, FORM_KINDS
from src.backtest.walkforward import fold_boundaries

FADE_GRID = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4]


def _fit_fade(tr, model, bmodel) -> float:
    """train fold で ◎2着/◎3着 の出現率に合うよう fav_fade を選ぶ。"""
    cache = []
    a2 = a3 = 0
    for s, d, lines, bt, order in tr:
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        fav = max(st, key=st.get)
        cache.append((st, bt, lines, fav))
        a2 += int(order[1] == fav)
        a3 += int(order[2] == fav)
    if not cache:
        return 1.0
    n = len(cache)
    t2, t3 = a2 / n, a3 / n
    best, bd = 1.0, 1e9
    for ff in FADE_GRID:
        p2 = p3 = 0.0
        for st, b, lines, fav in cache:
            dd = branch_trifecta(st, b, lines, fav_fade=ff)
            p2 += sum(p for k, p in dd.items() if k[1] == fav)
            p3 += sum(p for k, p in dd.items() if k[2] == fav)
        err = abs(p2 / n - t2) + abs(p3 / n - t3)
        if err < bd:
            best, bd = ff, err
    return best


def _ctx(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    nb = defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        nb[rid][car] = (li, pi)
    pos, sbm = defaultdict(dict), defaultdict(dict)
    for rid, p, car, s in c.execute("SELECT race_id,position,car_number,sb FROM results"):
        pos[rid][p] = car
        sbm[rid][car] = s
    c.close()
    return nb, pos, sbm


def _lines(d: dict) -> list[list[int]]:
    mem = defaultdict(list)
    for car, (li, pi) in d.items():
        mem[li].append((pi, car))
    return [[c for _, c in sorted(v)] for _, v in sorted(mem.items())]


def _kind_of(order3, fav) -> str:
    """実際の着順から、それがどの型だったかを判定する。"""
    if order3[0] == fav:
        return "◎頭"
    if len(order3) > 1 and order3[1] == fav:
        return "◎2着"
    if len(order3) > 2 and order3[2] == fav:
        return "◎3着"
    return "◎抜き"


def main() -> None:
    ap = argparse.ArgumentParser(description="買い目（型別）の検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--budget", type=int, default=18)
    ap.add_argument("--emit", action="store_true",
                    help="実測の的中率・回収率を表示用の定数として保存する")
    args = ap.parse_args()

    # 回収率も測る。表示に「実測的中率」を出す以上、当たったときいくら返ったかも
    # 併記しないと「よく当たる買い目＝良い買い目」と読まれてしまう
    c0 = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    payout = {r: (combo, p) for r, combo, p in
              c0.execute("SELECT race_id,combo,payout FROM payouts_trifecta")}
    c0.close()

    nb, pos, sbm = _ctx(args.db)
    raw = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = augment_samples(raw, args.db, men_features())

    rows = []
    for s in samples:
        d, P = nb.get(s.race_id), pos.get(s.race_id)
        if not d or not P or 1 not in P or 2 not in P or 3 not in P:
            continue
        bs = [x for x in d if sbm[s.race_id].get(x) and "B" in str(sbm[s.race_id][x])]
        if len(bs) != 1:
            continue
        rows.append((s, d, _lines(d), bs[0], (P[1], P[2], P[3])))
    print(f"対象 {len(rows):,}レース（B一意・3着まで確定）\n")

    bounds = fold_boundaries(len(rows), n_folds=args.folds, warmup_frac=0.40,
                             window="expanding")
    # 型ごとに: 予測確率の和 / 実際の出現数 / 予測カバーの和 / 実際に買い目が当たった数
    agg = {k: {"p": 0.0, "n": 0, "cov_p": 0.0, "cov_n": 0, "pts": 0,
               "stake": 0, "ret": 0} for k in FORM_KINDS}
    total = 0
    for fi, (a, b2, c) in enumerate(bounds):
        tr, te = rows[a:b2], rows[b2:c]
        model = train_gbdt([r[0] for r in tr])
        # 展開AI（主導権予測）も fold 内で学習する
        btr = []
        for s, d, lines, bt, _o in tr:
            t = type(s)(**{**s.__dict__})
            t.order = [bt] + [x for x in s.car_numbers if x != bt]
            btr.append(t)
        bmodel = train_gbdt(btr)
        fade = _fit_fade(tr, model, bmodel)
        print(f"  fold{fi}: fav_fade={fade}")

        for s, d, lines, bt, order in te:
            st = model.strengths(s.X, s.car_numbers)
            pb = bmodel.strengths(s.X, s.car_numbers)
            if not st or not pb:
                continue
            fav = max(st, key=st.get)
            # 本番と同じ: 主導権は予測。上位分岐の混合分布を作る
            mix = defaultdict(float)
            for b, p in sorted(pb.items(), key=lambda kv: -kv[1])[:3]:
                if p < 0.05:
                    continue
                for k, v in branch_trifecta(st, b, lines, fav_fade=fade).items():
                    mix[k] += p * v
            z = sum(mix.values())
            if z <= 0:
                continue
            mix = {k: v / z for k, v in mix.items()}
            types = {t["kind"]: t for t in formation_types(mix, fav, budget=args.budget,
                                                           min_prob=0.0)}
            actual_kind = _kind_of(order, fav)
            total += 1
            for k in FORM_KINDS:
                t = types.get(k)
                if t is None:
                    continue
                agg[k]["p"] += t["scenario_prob"]
                agg[k]["cov_p"] += t["scenario_prob"] * t["cover"]
                f = t["formation"]
                agg[k]["pts"] += f["points"]
                agg[k]["stake"] += 100 * f["points"]
                if k == actual_kind:
                    agg[k]["n"] += 1
                    hit = (order[0] in f["first"] and order[1] in f["second"]
                           and order[2] in f["third"])
                    agg[k]["cov_n"] += int(hit)
                    if hit:
                        pay = payout.get(s.race_id)
                        if pay and pay[1]:
                            agg[k]["ret"] += pay[1]

    print(f"{'型':>7}{'予測確率':>10}{'実測':>8}{'誤差':>8}"
          f"{'予測カバー':>11}{'実測カバー':>11}{'誤差':>8}{'平均点数':>9}")
    errs_p, errs_c = [], []
    for k in FORM_KINDS:
        v = agg[k]
        pp = v["p"] / total * 100
        aa = v["n"] / total * 100
        cp = (v["cov_p"] / v["p"] * 100) if v["p"] > 0 else 0.0
        ca = (v["cov_n"] / v["n"] * 100) if v["n"] else 0.0
        pts = v["pts"] / total
        errs_p.append(abs(pp - aa))
        if v["n"] >= 100:
            errs_c.append(abs(cp - ca))
        print(f"{k:>7}{pp:>9.1f}%{aa:>7.1f}%{pp-aa:>+8.1f}"
              f"{cp:>10.1f}%{ca:>10.1f}%{cp-ca:>+8.1f}{pts:>9.1f}")
    mae_p = sum(errs_p) / len(errs_p)
    mae_c = sum(errs_c) / len(errs_c) if errs_c else 99.0
    print(f"\nn={total:,}  型の出現確率 平均絶対誤差 {mae_p:.2f}pt / "
          f"カバー率 平均絶対誤差 {mae_c:.2f}pt")
    print("\n事前基準の判定:")
    print(f"  主基準（型の出現確率 3pt以内）: {'充足' if mae_p <= 3.0 else '不充足'}")
    print(f"  副基準（カバー率 5pt以内）: {'充足' if mae_c <= 5.0 else '不充足'}")
    print(f"\n→ {'採用（買い目として出す）' if mae_p <= 3.0 and mae_c <= 5.0 else '不採用'}")

    # ---- 表示用の実測値 ----
    # ダッシュボードに「予測的中率」だけ出すと当たる気にさせるので、**実測の的中率と
    # 回収率をセットで**出せるようにする。回収率は控除率25%のため上限75%。
    print(f"\n{'型':>7}{'実測的中率(無条件)':>20}{'平均点数':>9}{'回収率':>9}")
    emit = {}
    for k in FORM_KINDS:
        v = agg[k]
        hit_abs = v["cov_n"] / total if total else 0.0
        roi = v["ret"] / v["stake"] if v["stake"] else 0.0
        print(f"{k:>7}{hit_abs*100:>19.1f}%{v['pts']/total:>9.1f}{roi*100:>8.1f}%")
        emit[k] = {"hit": round(hit_abs, 4), "roi": round(roi, 4),
                   "points": round(v["pts"] / total, 1), "n": total}
    if args.emit:
        import json
        p = Path(__file__).resolve().parent.parent / "src" / "model" / "formation_stats_men.json"
        p.write_text(json.dumps(
            {"note": "walk-forward out-of-sample の実測。hit=無条件の的中率, "
                     "roi=回収率(控除率25%のため上限75%)。買い目の推奨ではない。",
             "kinds": emit}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n保存: {p}")


if __name__ == "__main__":
    main()
