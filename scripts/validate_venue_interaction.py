"""交互作用の重みを「バンク3水準」→「会場別（実測逃げ率・縮約あり）」に替える検証。

動機: 投入済みのバンク交互作用は周長3水準の重みなので、同じ400mの
小松島(逃げ率9.4%)と四日市(29.5%)を同一に扱ってしまう。会場差は実在し（400m内で
9.4〜29.5%、各n=67〜292で有意）、カント・みなし直線・幅員・当日条件では説明できない
ことも検証済み（scripts/analyze_bank_specs.py）。individual要因の特定を諦めて、
会場ごとの実測レートを直接重みにする＝カント/風/路面など全要因の総和を取り込む。

  venue_w = (縮約後の会場逃げ率 − 全体逃げ率) / 10
  縮約: (会場の逃げ勝ち数 + k×バンク平均逃げ率) / (会場のレース数 + k)   k=SHRINK_K
        標本の薄い会場はバンク平均へ寄る＝過学習を防ぐ

**リーク防止**: 会場レートは各foldの「テスト開始前まで」のレースだけから算出する
（本番では推論時点までの全履歴を使えるので、この作り方が実運用と一致する）。
これをやらずに全期間から作ると、テストfoldの結果が重みに混入して楽観側に出る。

**事前登録した採否基準（後から緩めない）**:
  主基準: 全体 tri10 が過半foldで改善、かつ 全体 top1 の平均が悪化しない
  副基準: 400m部分集合の tri10 が過半foldで改善
          （バンク重みでは区別できない層＝会場別化の効果が出るべき場所）
  外れたら打ち切り。

  PYTHONIOENCODING=utf-8 python scripts/validate_venue_interaction.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
from collections import defaultdict
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
from src.features.bank_features import BANK_KEYS
from src.features import venue_meta as vm
from src.backtest.walkforward import fold_boundaries

STATS = Path(__file__).resolve().parent.parent / "src" / "model" / "kimarite_stats.json"
SHRINK_K = 80.0          # 会場レートをバンク平均へ縮約する擬似サンプル数


def _load_ctx(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    venue = {rid: v for rid, v in c.execute("SELECT race_id,venue_code FROM races")}
    esc = {rid: (kim == "逃") for rid, kim
           in c.execute("SELECT race_id,kimarite FROM results WHERE position=1")}
    c.close()
    return venue, esc


def _bank_rates() -> tuple[dict[int, float], float]:
    d = json.loads(STATS.read_text(encoding="utf-8"))
    g = d["global"]["kim"]["逃"]
    bank = {int(b): cell["kim"]["逃"] for b, cell in (d.get("bank") or {}).items()
            if (cell or {}).get("kim")}
    return bank, g


def _venue_weights(ids: list[str], venue: dict, esc: dict,
                   bank_rate: dict[int, float], glob: float) -> dict[str, float]:
    """指定レース集合（=そのfoldの学習可能部分）だけから会場別重みを作る。"""
    n = defaultdict(int)
    e = defaultdict(int)
    for rid in ids:
        v = venue.get(rid)
        if v is None or rid not in esc:
            continue
        n[v] += 1
        e[v] += int(esc[rid])
    out = {}
    for v in set(list(n) + list(venue.values())):
        bank = vm.bank_length(v or "")
        prior = bank_rate.get(bank, glob) if bank else glob
        rate = (e[v] * 100 + SHRINK_K * prior) / (n[v] + SHRINK_K)   # %単位
        out[v] = (rate - glob) / 10.0
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


def _reweight(samples, idx_esc: int, idx_lead: int, venue: dict, w: dict[str, float],
              bw: dict[str, float]):
    """交互作用列を venue_w で作り直す。元の rel 値 = 既存列 / bank_w で復元する。"""
    out = []
    for s in samples:
        v = venue.get(s.race_id, "")
        b = bw.get(s.race_id, 0.0)      # bw は race_id キー（会場コードで引くと常に0になる）
        s2 = copy.copy(s)
        X = s.X.copy()
        if abs(b) > 1e-9:                      # 既存列から rel を復元して掛け直す
            X[:, idx_esc] = X[:, idx_esc] / b * w.get(v, 0.0)
            X[:, idx_lead] = X[:, idx_lead] / b * w.get(v, 0.0)
        s2.X = X
        out.append(s2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="会場別重み交互作用の検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    venue, esc = _load_ctx(args.db)
    bank_rate, glob = _bank_rates()
    print(f"全体逃げ率 {glob}%  バンク別 " + " ".join(f"{k}m {v}%" for k, v in sorted(bank_rate.items())))

    feats = [f for f in (list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
                         + list(NARABI_KEYS)) if f != "t_escwin_rel"] + list(BANK_KEYS)
    base = load_samples(args.db, features=PL_FEATURES_FULL)
    s0 = augment_samples(base, args.db, feats)      # 現行（バンク3水準）
    names = list(s0[0].feature_names)
    i_esc, i_lead = names.index("x_esc_bank"), names.index("x_lead_bank")
    # 既存列の重み（bank_features と同じ値）を race_id ごとに引けるようにする
    from src.features.bank_features import bank_weight
    bw = {s.race_id: bank_weight(venue.get(s.race_id, "")) for s in s0}

    bank_of = {s.race_id: vm.bank_length(venue.get(s.race_id, "") or "") for s in s0}
    bounds = fold_boundaries(len(base), n_folds=args.folds, warmup_frac=0.40, window="expanding")
    print(f"\nwalk-forward {len(bounds)}fold / 全{len(base)}レース（縮約 k={SHRINK_K:.0f}）\n")
    print(f"{'fold':>4}{'testR':>7}{' 全体top1 現行→会場別':>24}{' 全体tri10 現行→会場別':>25}"
          f"{' 400m tri10 現行→会場別':>26}{'400n':>6}")

    agg = []
    for i, (a, b, c) in enumerate(bounds):
        # テスト開始(b)より前のレースだけで会場重みを作る＝as-of（リーク防止）
        w = _venue_weights([s.race_id for s in s0[:b]], venue, esc, bank_rate, glob)
        sv = _reweight(s0, i_esc, i_lead, venue, w, bw)
        # 自己チェック: 列が実際に変わっているか（変わっていなければ結果が全て0になり、
        # それを「効果なし」と誤読してしまう。実際に一度その取り違えをしたので必ず見る）
        diff = sum(1 for x, y in zip(s0[b:c], sv[b:c])
                   if not np.allclose(x.X[:, [i_esc, i_lead]], y.X[:, [i_esc, i_lead]]))
        if i == 0:
            v400 = [vv for vv in set(venue.values()) if vm.bank_length(vv or "") == 400]
            ws = sorted((w.get(vv, 0.0), vm.venue_name(vv) or vv) for vv in v400)
            print(f"  [fold0] 重みが変化したテストレース {diff}/{len(sv[b:c])}  "
                  f"400m会場の重み範囲 {ws[0][0]:+.3f}({ws[0][1]}) 〜 {ws[-1][0]:+.3f}({ws[-1][1]})"
                  f"  ※現行の400m一律は {bank_rate.get(400, 0) - glob:+.1f}%/10 = "
                  f"{(bank_rate.get(400,0)-glob)/10:+.3f}")
        if diff == 0:
            raise SystemExit("★ 交互作用列が変化していない。重みの適用に失敗している。")
        m0, mv = train_gbdt(s0[a:b]), train_gbdt(sv[a:b])
        t0, tv = s0[b:c], sv[b:c]
        e0, ev = evaluate(m0.strengths, t0), evaluate(mv.strengths, tv)
        r0, rv = _tri10(m0, t0), _tri10(mv, tv)
        k0 = [s for s in t0 if bank_of.get(s.race_id) == 400]
        kv = [s for s in tv if bank_of.get(s.race_id) == 400]
        q0, qv = _tri10(m0, k0), _tri10(mv, kv)
        agg.append({"top1": ev["top1_acc"] - e0["top1_acc"], "tri10": rv - r0,
                    "ece": ev["ece"] - e0["ece"], "ll": ev["logloss"] - e0["logloss"],
                    "t400": qv - q0})
        print(f"{i:>4}{len(t0):>7}{e0['top1_acc']*100:>14.1f}→{ev['top1_acc']*100:.1f}%"
              f"{r0*100:>15.1f}→{rv*100:.1f}%{q0*100:>16.1f}→{qv*100:.1f}%{len(k0):>6}")

    n = len(agg)
    av = lambda k: sum(r[k] for r in agg) / n
    wins = lambda k, low=False: sum(1 for r in agg if (r[k] < 0 if low else r[k] > 0))
    maj = n // 2 + 1
    print(f"\n【会場別重みの純増（会場別−現行）／{n}fold平均・+勝ちfold数】")
    print(f"  全体 top1   {av('top1')*100:+.2f}pt  勝ち {wins('top1')}/{n}")
    print(f"  全体 tri10  {av('tri10')*100:+.2f}pt  勝ち {wins('tri10')}/{n}   ←主基準")
    print(f"  全体 logloss{av('ll'):+.4f}   改善 {wins('ll', True)}/{n}")
    print(f"  全体 ece    {av('ece'):+.5f}  改善 {wins('ece', True)}/{n}")
    print(f"  400m tri10  {av('t400')*100:+.2f}pt  勝ち {wins('t400')}/{n}   ←副基準")
    ok_main = wins("tri10") >= maj and av("top1") >= 0
    ok_sub = wins("t400") >= maj
    print(f"\n  主基準 {'充足' if ok_main else '未充足'}（tri10 {wins('tri10')}/{n} >= {maj} かつ top1平均>=0）"
          f" / 副基準 {'充足' if ok_sub else '未充足'}（400m tri10 {wins('t400')}/{n} >= {maj}）")
    print(f"  → {'採用検討' if (ok_main and ok_sub) else ('主基準のみ＝要追加確認' if ok_main else '不採用')}")


if __name__ == "__main__":
    main()
