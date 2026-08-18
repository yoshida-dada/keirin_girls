"""万車券率のしきい値を全履歴から推定して保存する。

検証（`scripts/validate_upset_prob.py`）で採用した推定量 `pls`:
  三連単分布のうち p <= しきい値 の目の確率を合計する。しきい値は
  「予測平均が実測の万車券率に一致する値」を二分探索で求める。**車立てごとに別々に**
  求める（組合せ数が 210(7車) と 504(9車) で倍以上違い、同じしきいは当てられない）。

検証は walk-forward（前のfoldだけで推定→次のfoldで評価）で行い、ここでは
**全履歴で推定し直す**（手法の妥当性は検証で確認済み。本番は全データを使う）。

事前基準を満たしたのは以下だけ。**満たさなかった層には出さない**:
  男子7車     ECE 0.0148 / Brier 0.1905(定数0.1970) / ρ0.99  → 採用
  ガールズ7車 ECE 0.0211 / Brier 0.1334(定数0.1420) / ρ0.93  → 採用
  男子9車     ECE 0.0523 → 不採用（n=1,130で十分位113件、ECEのノイズ下限が約0.038。
              0.03というしきいはこの標本では到達不能だった。基準は動かさず不採用）

  PYTHONIOENCODING=utf-8 python scripts/deploy_upset.py
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
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.feature_sets import men_features, girls_features
from src.model.plackett_luce import all_trifecta_probs
from src.model.development_branches import branch_mixture
from src.model.upset import P_MAN, THRESHOLD_PATH


def _lines_of(d: dict) -> list[list[int]]:
    """{車番:(line_id,pos)} → ライン構成のリスト。"""
    mem: dict[int, list] = defaultdict(list)
    for car, (li, pi) in d.items():
        mem[li].append((pi, car))
    return [[c for _, c in sorted(v)] for _, v in sorted(mem.items())]

# 検証で事前基準を満たした層だけ。ここに無い層には万車券率を出さない
APPROVED = {("men", 7), ("girls", 7)}


def fit_threshold(dists, ys) -> float:
    if not dists:
        return P_MAN
    target = sum(ys) / len(ys)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        m = sum(sum(p for p in d.values() if p <= mid) for d in dists) / len(dists)
        if m < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def run(db: str, sex: str) -> dict:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    lab = {r: int(p >= 10000) for r, p in
           c.execute("SELECT race_id,payout FROM payouts_trifecta WHERE payout IS NOT NULL")}
    c.close()
    girls = sex == "girls"
    feats = girls_features() if girls else men_features()
    raw = load_samples(db, field_size=[7] if girls else [7, 9], features=PL_FEATURES_FULL)
    smp = augment_samples(raw, db, feats)
    model = train_gbdt(smp)          # 全履歴で学習（しきい値推定のためだけに使う）
    # **本番と同じ分布でしきい値を推定する**。以前は素のPLで推定した値を、本番では
    # 紐補正版に当てていた（分布の形が違えばしきいはズレる）。男子は本表示が
    # 分岐混合になったので、ここも混合で推定する。
    bmodel, nar = None, {}
    if not girls:
        btr = []
        c2 = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        sb = defaultdict(dict)
        for rid, car, s in c2.execute("SELECT race_id,car_number,sb FROM results"):
            sb[rid][car] = s
        for rid, car, li, pi in c2.execute(
                "SELECT race_id,car_number,line_id,pos_in_line FROM narabi"
                " WHERE line_id IS NOT NULL"):
            nar.setdefault(rid, {})[car] = (li, pi)
        c2.close()
        for s in smp:
            bs = [x for x in (nar.get(s.race_id) or {})
                  if sb.get(s.race_id, {}).get(x) and "B" in str(sb[s.race_id][x])]
            if len(bs) == 1:
                t = type(s)(**{**s.__dict__})
                t.order = [bs[0]] + [x for x in s.car_numbers if x != bs[0]]
                btr.append(t)
        bmodel = train_gbdt(btr) if btr else None
        print(f"  展開AI 学習 {len(btr):,}レース")

    def _dist(s, st):
        """本番と同じ経路で三連単分布を作る（混合が作れなければ素のPL）。"""
        if bmodel is not None and nar.get(s.race_id):
            pb = bmodel.strengths(s.X, s.car_numbers)
            ln = _lines_of(nar[s.race_id])
            mix, _ = branch_mixture(st, ln, pb)
            if mix:
                return mix
        return all_trifecta_probs(st)

    by_fs: dict[int, list] = {}
    for s in smp:
        if s.race_id not in lab:
            continue
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        by_fs.setdefault(len(s.car_numbers), []).append((_dist(s, st), lab[s.race_id]))

    out = {}
    for fs, rows in sorted(by_fs.items()):
        ds = [d for d, _ in rows]
        ys = [y for _, y in rows]
        t = fit_threshold(ds, ys)
        pred = sum(sum(p for p in d.values() if p <= t) for d in ds) / len(ds)
        ok = (sex, fs) in APPROVED
        print(f"  {sex} {fs}車 n={len(rows):,} 実測{sum(ys)/len(ys)*100:.2f}% "
              f"→ しきい値 {t:.6f}（既定 {P_MAN:.6f}） 予測平均{pred*100:.2f}% "
              f"{'採用' if ok else '**不採用（検証の事前基準を満たさない層）**'}")
        if ok:
            out[str(fs)] = round(t, 6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="万車券率のしきい値を推定して保存")
    ap.add_argument("--men-db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--girls-db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc = {}
    for sex, db in (("men", args.men_db), ("girls", args.girls_db)):
        if not Path(db).exists():
            print(f"{sex}: DB無し（{db}）")
            continue
        print(f"{sex}:")
        doc[sex] = run(db, sex)

    print(f"\n{json.dumps(doc, ensure_ascii=False)}")
    if args.dry_run:
        print("dry-run: 書き込みなし")
        return
    p = Path(THRESHOLD_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {p}")


if __name__ == "__main__":
    main()
