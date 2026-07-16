"""並び予想の検証: (A) 並び予想の的中率(事前×事後), (B) 位置取り特徴のモデル寄与。

A) 診断: 予想先頭(position0) が実際に S/B(主導権)を取ったか・1着になったか・上位3内か。
   並び予想がどれだけ当たるか＝どれだけ信じてよいかを数値化。混戦とそれ以外で層別。
B) 特徴評価: narabi特徴(narabi_pos/lead/leg のレース内相対化)を本番31特徴へ足し、time_split で
   top1/logloss/brier/ece/三連単top10 を比較（narabiがある期間内でのみ）。混戦サブセットも見る。

  PYTHONIOENCODING=utf-8 python scripts/analyze_narabi.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import copy
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import evaluate, time_split
from src.model.feature_augment import augment_samples
from src.model.race_type import classify_race
from src.model.plackett_luce import all_trifecta_probs
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import compute_narabi_features, NARABI_KEYS


def _tri10(model, test):
    hit = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        ranked = [k for k, _ in sorted(all_trifecta_probs(st).items(), key=lambda kv: -kv[1])]
        act = tuple(s.order[:3])
        hit += int(act in ranked[:10])
    return round(hit / len(test), 4) if test else 0.0


def _rel(vals):
    present = [v for v in vals if v is not None]
    mean = sum(present) / len(present) if present else 0.0
    return [(v - mean) if v is not None else 0.0 for v in vals]


def diagnostic(db_path, narabi):
    """A) 並び予想の的中率（予想先頭 vs 実S/B取得者 / 1着 / top3）。混戦層別。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    # narabi のあるレースの結果(車→position,sb)を集める
    rids = {rid for (rid, _c) in narabi}
    res = {}
    for rid, car, pos, sb in conn.execute(
            "SELECT race_id, car_number, position, sb FROM results WHERE position IS NOT NULL"):
        if rid in rids:
            res.setdefault(rid, {})[car] = (pos, sb or "")
    conn.close()

    # レースごとに 予想先頭 と 実測を突合
    agg = {"all": [0, 0, 0, 0], "混戦以外近似": [0, 0, 0, 0]}  # [races, lead_takesB, lead_win, lead_top3]
    n = leadB = leadWin = leadTop3 = 0
    for rid, cars in res.items():
        lead = next((c for (r, c) in narabi if r == rid and narabi[(r, c)]["narabi_lead"] == 1.0), None)
        if lead is None or lead not in cars:
            continue
        n += 1
        # 実際にBを取った車（複数可）
        b_cars = {c for c, (_p, sb) in cars.items() if "B" in sb}
        pos_lead = cars[lead][0]
        leadB += int(lead in b_cars)
        leadWin += int(pos_lead == 1)
        leadTop3 += int(pos_lead <= 3)
    print("\n【A) 並び予想の的中率（narabiのある確定レース）】")
    if n:
        print(f"  対象 {n}レース")
        print(f"  予想先頭が実際にB(主導権)を取った率 : {leadB/n*100:.1f}%")
        print(f"  予想先頭が1着になった率            : {leadWin/n*100:.1f}%")
        print(f"  予想先頭がtop3に入った率           : {leadTop3/n*100:.1f}%")
    else:
        print("  narabi×結果 の突合レースがまだありません（バックフィル/蓄積を待つ）。")
    return n


def feature_eval(db_path, narabi):
    """B) narabi特徴の寄与（31 vs 31+narabi）。narabiのある期間内で time_split 比較。"""
    narabi_rids = {rid for (rid, _c) in narabi}
    base = [s for s in load_samples(db_path, features=PL_FEATURES_FULL) if s.race_id in narabi_rids]
    print(f"\n【B) 位置取り特徴の寄与】narabi のあるサンプル {len(base)}レース")
    if len(base) < 200:
        print("  サンプルが少なく評価は参考値（バックフィルを増やすと安定）。")
    if not base:
        return
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    s31 = augment_samples(base, db_path, feats31)

    def add_narabi(samples):
        out = []
        for s in samples:
            s2 = copy.copy(s)
            cols = []
            for key in NARABI_KEYS:
                vals = [narabi.get((s.race_id, c), {}).get(key) for c in s.car_numbers]
                # narabi_lead は相対化しない(0/1)。pos/leg はレース内相対化。
                col = vals if key == "narabi_lead" else _rel(vals)
                cols.append(np.array([v if v is not None else 0.0 for v in col]).reshape(-1, 1))
            s2.X = np.hstack([s.X] + cols)
            s2.feature_names = list(s.feature_names) + NARABI_KEYS
            out.append(s2)
        return out

    s_nb = add_narabi(s31)
    tr0, te0 = time_split(s31, 0.30)
    tr1, te1 = time_split(s_nb, 0.30)
    m0, m1 = train_gbdt(tr0), train_gbdt(tr1)
    r0, r1 = evaluate(m0.strengths, te0), evaluate(m1.strengths, te1)
    print(f"  検証 test {len(te0)}レース")
    print(f"  {'指標':<10}{'31特徴':>12}{'+並び予想':>12}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"  {k:<10}{r0[k]:>12}{r1[k]:>12}")
    print(f"  {'三連単top10':<10}{_tri10(m0, te0):>12}{_tri10(m1, te1):>12}")

    # 混戦サブセット（基準31モデルで分類）
    labels = {s.race_id: classify_race(m0.strengths(s.X, s.car_numbers)).label for s in te0}
    ch0 = [s for s in te0 if labels.get(s.race_id) == "混戦"]
    ch1 = [s for s in te1 if labels.get(s.race_id) == "混戦"]
    if ch0:
        c0, c1 = evaluate(m0.strengths, ch0), evaluate(m1.strengths, ch1)
        print(f"\n  [混戦 {len(ch0)}レース] top1 {c0['top1_acc']}→{c1['top1_acc']} / "
              f"logloss {c0['logloss']}→{c1['logloss']} / ece {c0['ece']}→{c1['ece']} / "
              f"tri10 {_tri10(m0, ch0)}→{_tri10(m1, ch1)}")


def main():
    ap = argparse.ArgumentParser(description="並び予想の検証（的中率＋特徴寄与）")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    args = ap.parse_args()
    narabi = compute_narabi_features(args.db)
    print(f"narabi 収録: {len({r for (r, _c) in narabi})}レース / {len(narabi)}エントリ")
    if not narabi:
        print("narabi が空。scripts/backfill_narabi.py で蓄積してから実行。"); return
    diagnostic(args.db, narabi)
    feature_eval(args.db, narabi)


if __name__ == "__main__":
    main()
