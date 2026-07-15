"""混戦◎嫌い戦略(F3=◎を2着固定 ほか)の walk-forward 多fold頑健性検証。

単一分割で F3(◎2着固定・9点/R) が ROI均等114%だったが、ドッチング82%との乖離＝単発/大穴依存の
疑いがある。複数の時系列fold（各foldで train学習→未来をout-of-sample検証）で再現するかを見る。
各foldで 均等・ドッチング両方のROIを出し、fold間のばらつきで「本物のエッジか単発か」を判定する。

  PYTHONIOENCODING=utf-8 python scripts/validate_chaos_walkforward.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.backtest.walkforward import walk_forward_folds
from src.backtest.bucket_analysis import build_records
from src.backtest.selection import group_by_race, _first_place_probs, settle_full


def _marks(recs):
    fp = _first_place_probs(recs)
    ranked = [c for c, _ in sorted(fp.items(), key=lambda kv: -kv[1])]
    return ranked if len(ranked) >= 5 else None


def picks_f3(recs):
    """F3: 1着∈{○,▲,△} → 2着=◎ → 3着∈{○,▲,△,×}（9点/R）。"""
    r = _marks(recs)
    if not r:
        return []
    hon, maru, sanko, sankaku, batsu = r[:5]
    heads = (maru, sanko, sankaku)          # ○▲△
    thirds = (maru, sanko, sankaku, batsu)  # ○▲△×
    want = {(h, hon, c) for h in heads for c in thirds if len({h, hon, c}) == 3}
    rec_of = {x.combo: x for x in recs}
    return [rec_of[w] for w in want if w in rec_of]


def picks_anti(recs):
    """参考: 1着≠◎ 総流し（≈179点/R）。"""
    r = _marks(recs)
    if not r:
        return []
    return [x for x in recs if x.combo[0] != r[0]]


def apply(recs_by_race, fn):
    out = {}
    for rid, rs in recs_by_race.items():
        if rs and rs[0].race_type == "混戦":
            p = fn(rs)
            if p:
                out[rid] = p
    return out


def main():
    ap = argparse.ArgumentParser(description="混戦◎嫌いの walk-forward 検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    samples = augment_samples(base, args.db, feats31)

    print(f"walk-forward {args.folds}fold（expanding, warmup40%）／31特徴lambdarank／混戦のみ\n")
    print(f"{'fold':>4}{'検証期間':>26}{'混戦R':>7}"
          f"{'  F3 点/R 的中 ROI均等 ROIドッチ':>34}{'  |1着≠◎ ROI均等/ドッチ':>26}")
    agg = {"f3": [], "anti": []}
    for i, (tr, te) in enumerate(walk_forward_folds(samples, n_folds=args.folds,
                                                    warmup_frac=0.40, window="expanding")):
        model = train_gbdt(tr)
        recs = build_records(args.db, model, [s.race_id for s in te])
        by = group_by_race(recs)
        s3 = settle_full(apply(by, picks_f3))
        sa = settle_full(apply(by, picks_anti))
        d0, d1 = te[0].date, te[-1].date
        if s3:
            agg["f3"].append(s3)
        if sa:
            agg["anti"].append(sa)
        f3s = (f"{s3['pts']:>6.1f}{s3['n_hits']:>5}{s3['roi_eq']*100:>8.1f}%{s3['roi_du']*100:>8.1f}%"
               if s3 else f"{'(該当なし)':>27}")
        anti = (f"{sa['roi_eq']*100:>10.1f}%{sa['roi_du']*100:>8.1f}%" if sa else f"{'—':>18}")
        n_ch = s3['n_races'] if s3 else 0
        print(f"{i:>4}{d0+'〜'+d1:>26}{n_ch:>7}   {f3s}   |{anti}")

    def summ(name, rows):
        if not rows:
            print(f"\n{name}: fold該当なし"); return
        eq = [r["roi_eq"] for r in rows]; du = [r["roi_du"] for r in rows]
        hits = sum(r["n_hits"] for r in rows); nb = sum(r["n_bets"] for r in rows)
        # 全fold合算ROI（重み=stake）
        eq_all = sum(r["roi_eq"] * r["n_bets"] * 100 for r in rows) / sum(r["n_bets"] * 100 for r in rows)
        print(f"\n{name}: fold数{len(rows)} 合算的中{hits} "
              f"ROI均等 min{min(eq)*100:.1f}% / max{max(eq)*100:.1f}% / 合算{eq_all*100:.1f}%  "
              f"ROIドッチ min{min(du)*100:.1f}% / max{max(du)*100:.1f}%")

    summ("F3(◎2着固定 9点)", agg["f3"])
    summ("1着≠◎ 総流し", agg["anti"])
    print("\n判定: 各foldでROI均等が安定して>100%かつドッチも追随 → 本物。"
          "fold間で乱高下/ドッチ<100% → 単発・大穴依存で実運用不可。")


if __name__ == "__main__":
    main()
