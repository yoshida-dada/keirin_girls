"""競り対応（案Y）の純増を walk-forward で検証する。

基準 = 現行の男子46特徴（競りを直列とみなしたライン位置）
比較 = 47特徴（競りの選手に同じライン内位置＋ln_seri フラグ）

**なぜやるか**: 記者の並び予想の `2先行 ( 7競り 1競り ) 4追込` は「7と1が番手を争う」意味だが、
パーサーがカッコを読み飛ばして5車の直列ラインに潰していた。実測で、記録上の「勝者の番手」が
2着に来る率は 競り無し41.1% に対し **競りあり33.7%（−7.4pt）**。番手を取れなかった方を
番手として扱っている。競りは392/25,383レース（1.54%）。

**事前登録した採否基準（後から緩めない）**:
  主基準: **競りを含むレースだけ**で tri10 が過半fold(3/5以上)で改善
          （全体1.54%なので全体指標では埋もれる。ここを見ないと改善したか分からない）
  副基準: 全体の tri10 が悪化しない（平均Δ >= -0.1pt）
  外れたら不採用。競りレースが良くなっても全体を壊すなら入れない。

  PYTHONIOENCODING=utf-8 python scripts/validate_seri.py
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
from src.features.line_features import SERI_KEYS
from src.model.plackett_luce import all_trifecta_probs
from src.backtest.walkforward import fold_boundaries


def _metrics(model, test) -> tuple[float, float, int]:
    if not test:
        return 0.0, 0.0, 0
    t1 = t10 = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        t1 += int(max(st, key=st.get) == s.order[0])
        top = sorted(all_trifecta_probs(st).items(), key=lambda kv: -kv[1])[:10]
        t10 += int(tuple(s.order[:3]) in [k for k, _ in top])
    n = len(test)
    return t1 / n * 100, t10 / n * 100, n


def main() -> None:
    ap = argparse.ArgumentParser(description="競り対応の純増検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    c = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    seri_races = {r[0] for r in c.execute(
        "SELECT DISTINCT race_id FROM narabi WHERE seri_group IS NOT NULL")}
    c.close()
    print(f"競りを含むレース: {len(seri_races):,}")

    base_feats = men_features()                                  # 46列（現行・競りなし）
    new_feats = base_feats + list(SERI_KEYS)                     # 47列（競り込み）
    raw = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    # 競り補正（ライン内位置の共有）と ln_seri 列はセットで、ln_seri の有無だけで決まる。
    # → base は自動的に「競り補正なし」になる。ここが効いていないと位置補正が両方に入り、
    #   「フラグ1列だけ」を測ることになって検証にならない。
    ext = augment_samples(raw, args.db, new_feats)
    base = augment_samples(raw, args.db, base_feats)
    print(f"サンプル {len(base):,}  基準{len(base[0].feature_names)}列 → "
          f"拡張{len(ext[0].feature_names)}列")

    bounds = fold_boundaries(len(base), n_folds=args.folds, warmup_frac=0.40,
                             window="expanding")
    print(f"\n{'fold':>5}{'n_test':>8}{'競りn':>7}"
          f"{'全体tri10 基準':>15}{'拡張':>8}{'Δ':>7}"
          f"{'競りtri10 基準':>15}{'拡張':>8}{'Δ':>7}")
    d_all, d_seri = [], []
    for fi, (a, b, c2) in enumerate(bounds):
        mb = train_gbdt(base[a:b])
        me = train_gbdt(ext[a:b])
        tb, te = base[b:c2], ext[b:c2]
        _b1, b10, n = _metrics(mb, tb)
        _e1, e10, _ = _metrics(me, te)
        sb = [s for s in tb if s.race_id in seri_races]
        se = [s for s in te if s.race_id in seri_races]
        _sb1, sb10, sn = _metrics(mb, sb)
        _se1, se10, _ = _metrics(me, se)
        d_all.append(e10 - b10)
        if sn:
            d_seri.append(se10 - sb10)
        print(f"{fi:>5}{n:>8}{sn:>7}{b10:>14.2f}%{e10:>7.2f}%{e10-b10:>+7.2f}"
              f"{sb10:>14.2f}%{se10:>7.2f}%{se10-sb10:>+7.2f}")

    n_seri_ok = sum(1 for d in d_seri if d > 0)
    m_all = sum(d_all) / len(d_all)
    m_seri = sum(d_seri) / len(d_seri) if d_seri else 0.0
    print(f"\n平均Δ  全体 {m_all:+.2f}pt / 競りレース {m_seri:+.2f}pt "
          f"({n_seri_ok}/{len(d_seri)}fold改善)")
    ok_main = n_seri_ok >= (len(d_seri) + 1) // 2
    ok_sub = m_all >= -0.1
    print("\n事前基準の判定:")
    print(f"  主基準（競りレースの tri10 が過半foldで改善）: {'充足' if ok_main else '不充足'}")
    print(f"  副基準（全体の tri10 が悪化しない）: {'充足' if ok_sub else '不充足'}")
    print(f"\n→ {'採用' if ok_main and ok_sub else '不採用（基準を緩めない）'}")


if __name__ == "__main__":
    main()
