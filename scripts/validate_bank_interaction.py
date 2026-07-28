"""「逃げ力×バンク」交互作用の純増を walk-forward で検証（36特徴 vs 37/38特徴）。

動機: 実測でバンク長は決まり手を大きく変える（逃げ率 333m 35.3% / 400m 19.3% / 500m 12.8%、
ペースで層別しても消えない）。ところが**モデルはバンクを一切知らない** — バンクはレース定数で、
順位モデル（PL/lambdarank）はシフト不変なので race一定の列は設計上除外されているため。
唯一バンク情報を注入できる形が「レース内で変動する量 × バンクの逃げ有利度」の交互作用。

  bank_w   = (そのバンクの逃げ率 − 全体逃げ率) / 10     … 333:+1.39 / 400:-0.21 / 500:-0.86
  x_esc_bank  = t_escape_rel × bank_w    （逃げ残率が短走路で効く、を表現）
  x_lead_bank = t_lead_rel   × bank_w    （主導権指数が短走路で効く）

過去に不採用となった「選手×バンク長**適性**」（その選手のバンク別成績）とは別物で、
こちらは「短走路では逃げという決着自体が残りやすい」という決まり手構造を表す。

**事前登録した採否基準（後から緩めない）**:
  主基準: 全体 tri10 が >=3/5 fold で改善、かつ 全体 top1 の平均が悪化しない
  副基準: 333m 部分集合の top1 が >=3/5 fold で改善
  外れたら打ち切り。V1(1列)を優先し、V2(2列)は探索的な参考。

  PYTHONIOENCODING=utf-8 python scripts/validate_bank_interaction.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate
from src.model.feature_augment import augment_samples
from src.model.plackett_luce import all_trifecta_probs
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS
from src.features import venue_meta as vm
from src.backtest.walkforward import fold_boundaries

STATS = Path(__file__).resolve().parent.parent / "src" / "model" / "kimarite_stats.json"


def _bank_weight() -> dict[int, float]:
    """バンク長 → 逃げ有利度。実測の逃げ率と全体との差を1/10スケールで。"""
    d = json.loads(STATS.read_text(encoding="utf-8"))
    g = d["global"]["kim"]["逃"]
    out = {}
    for b, cell in (d.get("bank") or {}).items():
        if cell.get("kim"):
            out[int(b)] = (cell["kim"]["逃"] - g) / 10.0
    return out


def _venue_of(db: str) -> dict[str, str]:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    out = {rid: v for rid, v in c.execute("SELECT race_id,venue_code FROM races")}
    c.close()
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


def _add_inter(samples, venue, bw, cols: list[str]):
    """指定した相対化特徴 × バンク重み の列を末尾に足した samples を返す。"""
    out = []
    for s in samples:
        s2 = copy.copy(s)
        names = list(s.feature_names)
        bank = vm.bank_length(venue.get(s.race_id, "") or "")
        w = bw.get(bank, 0.0) if bank else 0.0
        extra = []
        for cname in cols:
            i = names.index(cname)
            extra.append(s.X[:, i] * w)
        s2.X = np.hstack([s.X, np.array(extra, dtype=float).T])
        s2.feature_names = names + [f"x_{c}_bank" for c in cols]
        out.append(s2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="逃げ力×バンク交互作用の検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    bw = _bank_weight()
    venue = _venue_of(args.db)
    print("バンク重み(逃げ有利度): " + "  ".join(f"{k}m {v:+.2f}" for k, v in sorted(bw.items())))

    feats = [f for f in (list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
                         + list(NARABI_KEYS)) if f != "t_escwin_rel"]
    base = load_samples(args.db, features=PL_FEATURES_FULL)
    s0 = augment_samples(base, args.db, feats)
    v1 = _add_inter(s0, venue, bw, ["t_escape_rel"])
    v2 = _add_inter(s0, venue, bw, ["t_escape_rel", "t_lead_rel"])
    print(f"基準 {s0[0].X.shape[1]}特徴 / V1 {v1[0].X.shape[1]} / V2 {v2[0].X.shape[1]}")

    bank_of = {s.race_id: vm.bank_length(venue.get(s.race_id, "") or "") for s in s0}
    bounds = fold_boundaries(len(base), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"\nwalk-forward {len(bounds)}fold / 全{len(base)}レース\n")

    for tag, sv in (("V1 逃げ残率×バンク", v1), ("V2 +主導権指数×バンク", v2)):
        print(f"===== {tag} =====")
        print(f"{'fold':>4}{'testR':>7}{' 全体top1 基準→変更':>22}{' 全体tri10 基準→変更':>23}"
              f"{' 333m top1 基準→変更':>23}{'333n':>6}")
        agg = []
        for i, (a, b, c) in enumerate(bounds):
            m0, mv = train_gbdt(s0[a:b]), train_gbdt(sv[a:b])
            t0, tv = s0[b:c], sv[b:c]
            e0, ev = evaluate(m0.strengths, t0), evaluate(mv.strengths, tv)
            r0, rv = _tri10(m0, t0), _tri10(mv, tv)
            k0 = [s for s in t0 if bank_of.get(s.race_id) == 333]
            kv = [s for s in tv if bank_of.get(s.race_id) == 333]
            x0, xv = evaluate(m0.strengths, k0), evaluate(mv.strengths, kv)
            agg.append({"top1": ev["top1_acc"] - e0["top1_acc"], "tri10": rv - r0,
                        "ece": ev["ece"] - e0["ece"], "ll": ev["logloss"] - e0["logloss"],
                        "b333": xv.get("top1_acc", 0) - x0.get("top1_acc", 0)})
            print(f"{i:>4}{len(t0):>7}{e0['top1_acc']*100:>13.1f}→{ev['top1_acc']*100:.1f}%"
                  f"{r0*100:>14.1f}→{rv*100:.1f}%"
                  f"{x0.get('top1_acc',0)*100:>14.1f}→{xv.get('top1_acc',0)*100:.1f}%{len(k0):>6}")
        n = len(agg)
        av = lambda k: sum(r[k] for r in agg) / n
        wins = lambda k, low=False: sum(1 for r in agg if (r[k] < 0 if low else r[k] > 0))
        print(f"\n  全体 top1   {av('top1')*100:+.2f}pt  勝ち {wins('top1')}/{n}")
        print(f"  全体 tri10  {av('tri10')*100:+.2f}pt  勝ち {wins('tri10')}/{n}   ←主基準")
        print(f"  全体 logloss{av('ll'):+.4f}   改善 {wins('ll', True)}/{n}")
        print(f"  全体 ece    {av('ece'):+.5f}  改善 {wins('ece', True)}/{n}")
        print(f"  333m top1   {av('b333')*100:+.2f}pt  勝ち {wins('b333')}/{n}   ←副基準")
        # 「過半fold」はfold数に応じて決める（3固定だと7foldで3/7を過半と誤判定する）
        maj = n // 2 + 1
        ok_main = wins("tri10") >= maj and av("top1") >= 0
        ok_sub = wins("b333") >= maj
        print(f"  → 主基準 {'充足' if ok_main else '未充足'}（tri10 {wins('tri10')}/{n} ≥ {maj} かつ top1平均≥0）"
              f" / 副基準 {'充足' if ok_sub else '未充足'}（333m {wins('b333')}/{n} ≥ {maj}）")
        print(f"     {'採用検討' if (ok_main and ok_sub) else ('主基準のみ充足＝要追加確認' if ok_main else '不採用')}\n")


if __name__ == "__main__":
    main()
