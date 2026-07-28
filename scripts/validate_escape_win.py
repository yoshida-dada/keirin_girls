"""逃げ切り率(t_escwin_rel)の純増を walk-forward で検証（36特徴 vs 37特徴）。

背景: 既存の逃げ残率 escape_survival は P(3着以内|B取得) で、上位選手では1.0に飽和し
「逃げ切る本命」と「捕まっても粘る本命」を区別できない（実測: B66本でtop3率1.00・逃切率0.12、
両者のSpearman順位相関0.625＝別情報）。そこで分子を「逃げ決まり手で1着」に絞った
  escape_win_rate = (逃げ切り勝ち + k*通算勝率) / (B取得走 + k)
をレース内相対化した t_escwin_rel を追加し、着順予測が改善するかを見る。

評価は全体（top1/tri10/ece/logloss）に加えて、狙いが効くべき部分集合を分けて出す:
  - 逃げ決着R : 実際の勝ち決まり手が「逃」だったレース（逃げ切りを当てられるようになったか）
  - 非逃げ決着R: それ以外（垂れて負ける側を当てられるようになったか）
部分集合は結果で切る事後分割なので**診断用**（ROI主張には使えない）。foldごとの符号一致で頑健性を見る。

  PYTHONIOENCODING=utf-8 python scripts/validate_escape_win.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate
from src.model.feature_augment import augment_samples
from src.model.plackett_luce import all_trifecta_probs
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS
from src.features.bank_features import BANK_KEYS
from src.backtest.walkforward import fold_boundaries

NEW_COL = "t_escwin_rel"
OLD_COL = "t_escape_rel"


def _with_escwin(samples, db: str, replace: bool):
    """逃げ切り率(レース内相対)の列を足す/差し替えた samples を返す。

    t_escwin_rel は不採用で TACTIC_NAMES に無いため、ここで as-of の raw 値から作る。
    相対化は tactics_features._rel と同一関数＝本番と同じ値になる。
    """
    import copy
    from src.features.rider_tactics import compute_pre_race_tactics
    from src.features.tactics_features import _rel

    tac = compute_pre_race_tactics(db)
    out = []
    for s in samples:
        cars = list(s.car_numbers)
        col = np.array(_rel([tac.get((s.race_id, c), {}).get("escape_win_rate") for c in cars]),
                       dtype=float).reshape(-1, 1)
        s2 = copy.copy(s)
        names = list(s.feature_names)
        if replace:                       # 逃げ残率を逃げ切り率へ差し替え（列数は不変）
            i = names.index(OLD_COL)
            X = s.X.copy()
            X[:, i] = col[:, 0]
            names[i] = NEW_COL
            s2.X = X
        else:                             # 追加（列数+1）
            s2.X = np.hstack([s.X, col])
            names = names + [NEW_COL]
        s2.feature_names = names
        out.append(s2)
    return out


def _tri10(model, test) -> float:
    if not test:
        return 0.0
    hit = 0
    for s in test:
        ranked = sorted(all_trifecta_probs(model.strengths(s.X, s.car_numbers)).items(),
                        key=lambda kv: -kv[1])[:10]
        hit += int(tuple(s.order[:3]) in [k for k, _ in ranked])
    return hit / len(test)


def _escape_races(db: str) -> set[str]:
    """勝ち決まり手が「逃」だったレースID（決まり手は1・2着のみ記録される疎データ）。"""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    ids = {rid for rid, in c.execute(
        "SELECT race_id FROM results WHERE position=1 AND kimarite LIKE '%逃%'")}
    c.close()
    return ids


def _upset_gap(model, test) -> float:
    """波乱確率の較正ギャップ（実−予測, pt）。0に近いほど良い。"""
    if not test:
        return 0.0
    pred = act = 0.0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        fav = max(st, key=st.get)
        pred += 1 - st[fav]
        act += int(s.order[0] != fav)
    return (act - pred) / len(test) * 100


def main() -> None:
    ap = argparse.ArgumentParser(description="逃げ切り率特徴の検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mode", choices=["add", "replace"], default="add",
                    help="add=36→37で追加 / replace=36のまま逃げ残率を逃げ切り率へ差し替え")
    args = ap.parse_args()

    # ベースラインは**現行デプロイと同じ構成**（バンク交互作用込みの38特徴）。
    # t_escwin_rel は不採用のため TACTIC_NAMES から外してあるので、この検証スクリプト内で
    # 逃げ切り率を相対化して列を作る（本番の既定に不採用特徴を戻さないための措置）。
    feats = (list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
             + list(NARABI_KEYS) + list(BANK_KEYS))
    base = load_samples(args.db, features=PL_FEATURES_FULL)
    s36 = augment_samples(base, args.db, feats)
    s37 = _with_escwin(s36, args.db, replace=(args.mode == "replace"))
    f36, fvar = list(s36[0].feature_names), list(s37[0].feature_names)
    assert s36[0].X.shape[1] == len(f36) and s37[0].X.shape[1] == len(fvar), "列数不一致"
    esc_ids = _escape_races(args.db)

    bounds = fold_boundaries(len(base), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    label = (f"+{NEW_COL}" if args.mode == "add" else f"{OLD_COL}→{NEW_COL} 置換")
    print(f"walk-forward {len(bounds)}fold / 全{len(base)}レース"
          f"（{len(f36)}特徴 vs {len(fvar)}特徴＝{label}）")
    print(f"逃げ決着レース: {len(esc_ids)}件\n")

    agg = []
    hdr = (f"{'fold':>4}{'testR':>7}{'  全体top1 36→37':>19}{'  全体tri10 36→37':>20}"
           f"{'   全体ece 36→37':>20}{'  逃げ決着top1 36→37':>22}")
    print(hdr)
    for i, (a, b, c) in enumerate(bounds):
        m36, m37 = train_gbdt(s36[a:b]), train_gbdt(s37[a:b])
        te36, te37 = s36[b:c], s37[b:c]
        e36, e37 = evaluate(m36.strengths, te36), evaluate(m37.strengths, te37)
        t36, t37 = _tri10(m36, te36), _tri10(m37, te37)
        ke36 = [s for s in te36 if s.race_id in esc_ids]
        ke37 = [s for s in te37 if s.race_id in esc_ids]
        kn36 = [s for s in te36 if s.race_id not in esc_ids]
        kn37 = [s for s in te37 if s.race_id not in esc_ids]
        x36, x37 = evaluate(m36.strengths, ke36), evaluate(m37.strengths, ke37)
        y36, y37 = evaluate(m36.strengths, kn36), evaluate(m37.strengths, kn37)
        agg.append({
            "top1": e37["top1_acc"] - e36["top1_acc"], "tri10": t37 - t36,
            "ece": e37["ece"] - e36["ece"], "ll": e37["logloss"] - e36["logloss"],
            "esc_top1": x37.get("top1_acc", 0) - x36.get("top1_acc", 0),
            "non_top1": y37.get("top1_acc", 0) - y36.get("top1_acc", 0),
            "gap36": _upset_gap(m36, te36), "gap37": _upset_gap(m37, te37),
        })
        print(f"{i:>4}{len(te36):>7}{e36['top1_acc']*100:>10.1f}→{e37['top1_acc']*100:.1f}%"
              f"{t36*100:>11.1f}→{t37*100:.1f}%{e36['ece']:>11.4f}→{e37['ece']:.4f}"
              f"{x36.get('top1_acc',0)*100:>12.1f}→{x37.get('top1_acc',0)*100:.1f}%")

    n = len(agg)
    if not n:
        print("fold なし")
        return

    def avg(k):
        return sum(r[k] for r in agg) / n

    def wins(k, better_low=False):
        return sum(1 for r in agg if (r[k] < 0 if better_low else r[k] > 0))

    print(f"\n【{label} の純増（変更後−現行）／{n}fold平均・+勝ちfold数】")
    print(f"  全体 top1      {avg('top1')*100:+.2f}pt   勝ち {wins('top1')}/{n}")
    print(f"  全体 tri10     {avg('tri10')*100:+.2f}pt   勝ち {wins('tri10')}/{n}")
    print(f"  全体 logloss   {avg('ll'):+.4f}    改善 {wins('ll', True)}/{n}（負=改善）")
    print(f"  全体 ece       {avg('ece'):+.5f}   改善 {wins('ece', True)}/{n}（負=改善）")
    print(f"  逃げ決着R top1 {avg('esc_top1')*100:+.2f}pt   勝ち {wins('esc_top1')}/{n}  ←狙いの本丸")
    print(f"  非逃げ決着 top1{avg('non_top1')*100:+.2f}pt   勝ち {wins('non_top1')}/{n}")
    print(f"  波乱確率gap    36={avg('gap36'):+.2f}pt → 37={avg('gap37'):+.2f}pt（0に近いほど良）")
    print("\n判定基準: 逃げ決着R top1 が過半foldで+ かつ 全体 top1/tri10 が悪化しない → 採用検討。"
          "\n          単一foldの大きな+は無視（過去にF3で再現しなかった前例あり）。")


if __name__ == "__main__":
    main()
