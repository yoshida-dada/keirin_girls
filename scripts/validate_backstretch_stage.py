"""二段階モデルv1: 展開AI(最終バック先頭B確率)→着順AIの特徴量、の効果を検証。

Stage1(展開AI): pre-race特徴から「最終バック先頭(sb='B')を取る選手」を lambdarank で学習し、
  各選手の P(B) を出す（train_gbdt を order=[B取得車] で流用）。
Stage2(着順AI): 通常の着順モデルに Stage1 の P(B) を1特徴として追加し、無し(baseline)と比較。
  リーク防止のため P(B) は「test=train窓で学習したStage1で予測」「train=窓内2分割クロスフィット」。

比較指標(walk-forward hold-out): top1(1着)/2着top3/三連単top10/三連単log-loss。
Stage1自体のB的中(argmax P(B)==実B)も出す（b_count argmax 51.3%が基準）。

  PYTHONIOENCODING=utf-8 python scripts/validate_backstretch_stage.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import copy
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS, compute_narabi_features
from src.model.himo_adjust import corrected_trifecta_probs
from src.backtest.walkforward import fold_boundaries


def _b_taker(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    byrace = {}
    for rid, car, v in c.execute("SELECT race_id,car_number,sb FROM results"):
        if v and "B" in v:
            byrace.setdefault(rid, []).append(car)
    c.close()
    return {rid: cs[0] for rid, cs in byrace.items() if len(cs) == 1}   # B一意のみ


def _as_border(samples, btk):
    """Stage1学習用: order を [B取得車] に差し替えたサンプル（B一意のレースのみ）。"""
    out = []
    for s in samples:
        b = btk.get(s.race_id)
        if b is None or b not in s.car_numbers:
            continue
        s2 = copy.copy(s)
        s2.order = [b]
        out.append(s2)
    return out


def _with_pb(samples, pb_by_race):
    """各サンプルXに P(B) 列を追加した新サンプル。"""
    out = []
    for s in samples:
        pb = pb_by_race.get(s.race_id)
        s2 = copy.copy(s)
        col = np.array([[pb.get(c, 0.0)] for c in s.car_numbers]) if pb else np.zeros((len(s.car_numbers), 1))
        s2.X = np.hstack([s.X, col])
        s2.feature_names = list(s.feature_names) + ["p_backstretch"]
        out.append(s2)
    return out


def _predict_pb(model, samples):
    return {s.race_id: model.strengths(s.X, s.car_numbers) for s in samples}


def _tri10(dist, order3):
    top = [k for k, _ in sorted(dist.items(), key=lambda kv: -kv[1])[:10]]
    return int(tuple(order3) in top)


def _second_top3(dist, a2):
    marg = {}
    for (a, b, c), p in dist.items():
        marg[b] = marg.get(b, 0.0) + p
    return int(a2 in [x for x, _ in sorted(marg.items(), key=lambda kv: -kv[1])[:3]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
    samples = augment_samples(base, args.db, feats)
    btk = _b_taker(args.db)
    narabi = compute_narabi_features(args.db)
    npos = {s.race_id: {c: narabi.get((s.race_id, c), {}).get("narabi_pos") for c in s.car_numbers}
            for s in samples}
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    agg = {k: 0 for k in ("n", "b_hit", "b_den",
                          "base_top1", "aug_top1", "base_2t3", "aug_2t3",
                          "base_t10", "aug_t10", "base_ll", "aug_ll")}
    perfold = []
    for fi, (a, b, c) in enumerate(bounds):
        tr, te = samples[a:b], samples[b:c]
        # Stage1: 全train窓で学習→testのP(B)
        s1_full = train_gbdt(_as_border(tr, btk))
        pb_te = _predict_pb(s1_full, te)
        # train窓の P(B) は2分割クロスフィット（リーク防止）
        m = len(tr) // 2
        s1_a = train_gbdt(_as_border(tr[m:], btk))   # 後半で学習→前半予測
        s1_b = train_gbdt(_as_border(tr[:m], btk))   # 前半で学習→後半予測
        pb_tr = {}
        pb_tr.update(_predict_pb(s1_a, tr[:m]))
        pb_tr.update(_predict_pb(s1_b, tr[m:]))

        # Stage2: baseline vs +P(B)
        base_model = train_gbdt(tr)
        aug_model = train_gbdt(_with_pb(tr, pb_tr))
        te_aug = _with_pb(te, pb_te)

        f = {k: 0 for k in agg}
        for s, sa in zip(te, te_aug):
            o3 = tuple(s.order[:3])
            np_ = npos[s.race_id]
            stb = base_model.strengths(s.X, s.car_numbers)
            sta = aug_model.strengths(sa.X, sa.car_numbers)
            db_ = corrected_trifecta_probs(stb, np_)
            da_ = corrected_trifecta_probs(sta, np_)
            w = s.order[0]
            f["base_top1"] += int(max(stb, key=stb.get) == w)
            f["aug_top1"] += int(max(sta, key=sta.get) == w)
            f["base_2t3"] += _second_top3(db_, o3[1]); f["aug_2t3"] += _second_top3(da_, o3[1])
            f["base_t10"] += _tri10(db_, o3); f["aug_t10"] += _tri10(da_, o3)
            pbll = db_.get(o3, 0.0); f["base_ll"] += -math.log(pbll) if pbll > 0 else 50.0
            pall = da_.get(o3, 0.0); f["aug_ll"] += -math.log(pall) if pall > 0 else 50.0
            f["n"] += 1
            bt = btk.get(s.race_id)
            if bt is not None and bt in s.car_numbers:
                f["b_den"] += 1
                f["b_hit"] += int(max(pb_te[s.race_id], key=pb_te[s.race_id].get) == bt)
        for k in agg:
            agg[k] += f[k]
        perfold.append(f)

    n = agg["n"]
    print(f"二段階v1検証 walk-forward {len(bounds)}fold / test {n}レース\n")
    print(f"【Stage1(展開AI) 最終バック先頭B的中】 argmax P(B)==実B: "
          f"{agg['b_hit']/max(1,agg['b_den'])*100:.1f}% （基準 b_count argmax 51.3%）\n")
    print("【Stage2(着順AI) baseline vs +P(B)特徴】")
    def line(lbl, bk, ak, pct=True, inv=False):
        bv, av = agg[bk]/n, agg[ak]/n
        if pct:
            print(f"  {lbl:<14} {bv*100:6.2f}% → {av*100:6.2f}%  (Δ{(av-bv)*100:+.2f})")
        else:
            print(f"  {lbl:<14} {bv:7.4f} → {av:7.4f}  (Δ{av-bv:+.4f}{' 改善' if av<bv else ''})")
    line("1着 top1的中", "base_top1", "aug_top1")
    line("2着 top3的中", "base_2t3", "aug_2t3")
    line("三連単 top10", "base_t10", "aug_t10")
    line("三連単 log-loss", "base_ll", "aug_ll", pct=False)
    print("\n  per-fold Δ(+P(B)−baseline):")
    for i, f in enumerate(perfold):
        m = f["n"]
        print(f"    fold{i} n={m}: top1 {(f['aug_top1']-f['base_top1'])/m*100:+.2f} / "
              f"2着top3 {(f['aug_2t3']-f['base_2t3'])/m*100:+.2f} / "
              f"tri10 {(f['aug_t10']-f['base_t10'])/m*100:+.2f} / "
              f"ll {(f['aug_ll']-f['base_ll'])/m:+.4f}")


if __name__ == "__main__":
    main()
