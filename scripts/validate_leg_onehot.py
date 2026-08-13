"""脚質 one-hot の純増を walk-forward で検証（男子・本番39特徴 vs +脚質7列）。

**なぜ再検証するか**: 以前「脚質1列(ln_leg)」を検証して不採用にした（tri10 -0.28pt・0/5fold）が、
原因は情報ではなく**符号化**だった。`LEG_AGGR_MEN` は 先行/押え先/カマシ を**すべて2.0に潰す**。
実測(25,269R)のライン先頭の主導権率は 先行55.2% / 押え先34.7% / カマシ33.0% で20pt違い、
さらに過去B回数の四分位で層別してもなお分かれる（最少帯26pt差・最多帯19pt差）＝
既存特徴に吸収されていない。one-hot なら順序も間隔も仮定しないのでこの差を表現できる。

**事前登録した採否基準（後から緩めない）**:
  主基準: tri10 が過半fold(3/5以上)で改善、かつ top1 の平均が悪化しない
  副基準: top1 が過半foldで改善
  外れたら不採用。数字を良く見せるための基準の作り直しはしない。

  PYTHONIOENCODING=utf-8 python scripts/validate_leg_onehot.py --db data/keirin_men.sqlite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.feature_sets import men_features
from src.model.plackett_luce import all_trifecta_probs
from src.features.line_features import LEGOH_KEYS
from src.backtest.walkforward import fold_boundaries


def _metrics(model, test) -> tuple[float, float]:
    """(top1的中率, 三連単top10的中率)。"""
    if not test:
        return 0.0, 0.0
    t1 = t10 = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        if max(st, key=st.get) == s.order[0]:
            t1 += 1
        top = sorted(all_trifecta_probs(st).items(), key=lambda kv: -kv[1])[:10]
        t10 += int(tuple(s.order[:3]) in [k for k, _ in top])
    n = len(test)
    return t1 / n * 100, t10 / n * 100


def main() -> None:
    ap = argparse.ArgumentParser(description="脚質one-hotの純増検証")
    ap.add_argument("--db", default="data/keirin_men.sqlite")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base_feats = men_features()                    # 本番39列
    test_feats = base_feats + list(LEGOH_KEYS)     # +脚質7列 = 46列
    base_raw = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    base = augment_samples(base_raw, args.db, base_feats)
    ext = augment_samples(base_raw, args.db, test_feats)
    print(f"サンプル {len(base):,}  基準{len(base[0].feature_names)}列 → "
          f"拡張{len(ext[0].feature_names)}列")
    assert list(ext[0].feature_names) == test_feats, "列順が想定と不一致"

    bounds = fold_boundaries(len(base), n_folds=args.folds, warmup_frac=0.40,
                             window="expanding")
    print(f"\n{'fold':>5}{'n_test':>8}{'top1 基準':>11}{'top1 +脚質':>12}{'Δ':>8}"
          f"{'tri10 基準':>12}{'tri10 +脚質':>13}{'Δ':>8}")
    d1s, d10s = [], []
    for fi, (a, b, c) in enumerate(bounds):
        mb = train_gbdt(base[a:b])
        me = train_gbdt(ext[a:b])
        b1, b10 = _metrics(mb, base[b:c])
        e1, e10 = _metrics(me, ext[b:c])
        d1s.append(e1 - b1)
        d10s.append(e10 - b10)
        print(f"{fi:>5}{c-b:>8}{b1:>10.2f}%{e1:>11.2f}%{e1-b1:>+8.2f}"
              f"{b10:>11.2f}%{e10:>12.2f}%{e10-b10:>+8.2f}")

    n1 = sum(1 for d in d1s if d > 0)
    n10 = sum(1 for d in d10s if d > 0)
    m1 = sum(d1s) / len(d1s)
    m10 = sum(d10s) / len(d10s)
    print(f"\n平均Δ  top1 {m1:+.2f}pt ({n1}/{len(d1s)}fold改善) / "
          f"tri10 {m10:+.2f}pt ({n10}/{len(d10s)}fold改善)")
    ok_main = n10 >= (len(d10s) + 1) // 2 and m1 >= 0
    ok_sub = n1 >= (len(d1s) + 1) // 2
    print(f"\n事前基準の判定:")
    print(f"  主基準（tri10が過半foldで改善 かつ top1平均が悪化しない）: "
          f"{'充足' if ok_main else '不充足'}")
    print(f"  副基準（top1が過半foldで改善）: {'充足' if ok_sub else '不充足'}")
    print(f"\n→ {'採用' if ok_main else '不採用（基準を緩めない）'}")


if __name__ == "__main__":
    main()
