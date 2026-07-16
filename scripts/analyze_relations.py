"""選手間の関係性（同県・同地区・同期）が着順/展開に相関するか、モデルに効くかを検証する。

ユーザー仮説: ガールズはラインが無いが、同県選手は自力ある選手の後ろに入り「ライン的なもの」を作る。
同地区・同期でも連携の可能性。これを検出（A 相関診断）し、特徴化して寄与を測る（B 特徴評価）。

A) 相関/クイックライン検出（entries+results。resultsは診断のみ・学習特徴には着順を使わない）:
   A1 同県ペアの着順隣接率／逃げ×top3 共起がランダム基準より高いか
   A2 有力者(競走得点max)と同県/同地区/同期の選手は、素の期待を上回るか（＝ライン的恩恵）
   A3 有力者(model 1着確率max)と同県の選手は、モデル top3 期待を上回るか（get_recordsはキャッシュ利用）
B) 関係性特徴（RELATION_KEYS）を本番31特徴へ足し、time_split で top1/logloss/brier/ece/三連単top10 を
   比較。混戦サブセットを層別（scripts/analyze_narabi.py の feature_eval と同型）。

  PYTHONIOENCODING=utf-8 python scripts/analyze_relations.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import copy
import math
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.evaluate import time_split, evaluate
from src.model.feature_augment import augment_samples
from src.model.race_type import classify_race
from src.model.plackett_luce import all_trifecta_probs
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_relations import (
    compute_relation_features, RELATION_KEYS, normalize_pref, district_of)


# --------------------------------------------------------------------------
# データ取得（read-only）
# --------------------------------------------------------------------------
def _load_race_tables(db_path: str, field_size: int = 7):
    """field_size 車で結果確定のレースについて entries と results を返す。

    entry[rid] = {car: {"pref","dist","term","score"}}
    result[rid] = {car: {"pos","sb","kimarite"}}
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rids = {r[0] for r in conn.execute(
            "SELECT race_id FROM races WHERE field_size=?", (field_size,))}
        entry: dict[str, dict] = defaultdict(dict)
        for rid, car, pref, term, score in conn.execute(
                "SELECT race_id, car_number, prefecture, term, racing_score FROM entries"):
            if rid in rids:
                entry[rid][car] = {"pref": normalize_pref(pref), "dist": district_of(pref),
                                   "term": term if (term is not None and term != 999) else None,
                                   "score": score}
        result: dict[str, dict] = defaultdict(dict)
        for rid, car, pos, sb, kim in conn.execute(
                "SELECT race_id, car_number, position, sb, kimarite FROM results"
                " WHERE position IS NOT NULL"):
            if rid in rids:
                result[rid][car] = {"pos": pos, "sb": sb or "", "kimarite": kim or ""}
    finally:
        conn.close()
    # 結果が3着以上そろうレースのみ
    races = [rid for rid in entry
             if rid in result and len([1 for v in result[rid].values() if v["pos"]]) >= 3]
    return entry, result, races


def _wilson_or_ci(rate, n):
    """二項比率の 95%CI 半幅（正規近似）。表示補助。"""
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / n)


# --------------------------------------------------------------------------
# A1 同県ペアの着順隣接・逃げ×top3 共起
# --------------------------------------------------------------------------
def diag_pairs(entry, result, races):
    print("\n【A1) 同県ペアの着順相関（entries×results。基準=無関係ランダム）】")
    # 7車の任意ペアが隣接着(|Δpos|==1)になる基準確率 = 6/C(7,2)=0.2857
    base_adj = 6 / math.comb(7, 2)

    same_adj = same_n = 0            # 同県ペア: 隣接着
    diff_adj = diff_n = 0            # 異県ペア: 隣接着（実測基準）
    same_lead_top3 = same_lead_n = 0  # 同県: 片方S/B先頭 かつ 相方top3
    diff_lead_top3 = diff_lead_n = 0

    for rid in races:
        es, rs = entry[rid], result[rid]
        cars = [c for c in es if c in rs and rs[c]["pos"]]
        for a, b in combinations(cars, 2):
            pa, pb = rs[a]["pos"], rs[b]["pos"]
            adj = int(abs(pa - pb) == 1)
            same = (es[a]["pref"] is not None and es[a]["pref"] == es[b]["pref"])
            if same:
                same_n += 1; same_adj += adj
            else:
                diff_n += 1; diff_adj += adj
            # 片方が主導権(S/B) & もう片方が top3（先頭が仲間を連れて来るか）
            for lead, mate in ((a, b), (b, a)):
                if "B" in rs[lead]["sb"] or "S" in rs[lead]["sb"]:
                    hit = int(rs[mate]["pos"] <= 3)
                    if same:
                        same_lead_n += 1; same_lead_top3 += hit
                    else:
                        diff_lead_n += 1; diff_lead_top3 += hit

    if same_n:
        r = same_adj / same_n
        print(f"  同県ペアが隣接着になる率   : {r*100:5.1f}%  (±{_wilson_or_ci(r,same_n)*100:.1f}, n={same_n}) "
              f"| 異県 {diff_adj/diff_n*100:.1f}% (n={diff_n}) | 理論基準 {base_adj*100:.1f}%")
    if same_lead_n:
        r = same_lead_top3 / same_lead_n
        print(f"  同県で片方がS/B先頭時、相方top3率: {r*100:5.1f}%  (n={same_lead_n}) "
              f"| 異県 {diff_lead_top3/diff_lead_n*100:.1f}% (n={diff_lead_n})")
    print("  ※ 同県>異県 なら『同県は着順が近く・先頭が仲間を引き上げる』傾向（=クイックライン）の証拠。")


# --------------------------------------------------------------------------
# A2 有力者と同県/同地区/同期 → 素の期待を上回るか（競走得点で強さを統制）
# --------------------------------------------------------------------------
def diag_ally_benefit(entry, result, races):
    print("\n【A2) 有力者(競走得点max)の仲間は素の期待を上回るか（自身の得点順位で層別統制）】")
    # 非有力者を「自身の競走得点順位(2..7位)」で層別し、仲間/非仲間の top3率 を比較。
    # 得点順位で条件付けることで『元々強い車が仲間になりやすい』交絡を除く。
    # 仲間種別ごとに集計。
    buckets = {"同県": defaultdict(lambda: [0, 0, 0, 0]),   # rank -> [ally_top3, ally_n, other_top3, other_n]
               "同地区": defaultdict(lambda: [0, 0, 0, 0]),
               "同期": defaultdict(lambda: [0, 0, 0, 0])}

    for rid in races:
        es, rs = entry[rid], result[rid]
        cars = [c for c in es if c in rs and rs[c]["pos"]]
        if len(cars) < 4:
            continue
        # 有力者
        top = max(cars, key=lambda c: (es[c]["score"] if es[c]["score"] is not None else -1e9, -c))
        tp, td, tt = es[top]["pref"], es[top]["dist"], es[top]["term"]
        # 得点順位(1=最強)
        ranked = sorted(cars, key=lambda c: -(es[c]["score"] if es[c]["score"] is not None else -1e9))
        rank = {c: i + 1 for i, c in enumerate(ranked)}
        for c in cars:
            if c == top:
                continue
            top3 = int(rs[c]["pos"] <= 3)
            rk = rank[c]
            for kind, (val, tval) in (("同県", (es[c]["pref"], tp)),
                                      ("同地区", (es[c]["dist"], td)),
                                      ("同期", (es[c]["term"], tt))):
                b = buckets[kind][rk]
                if val is not None and val == tval:
                    b[0] += top3; b[1] += 1
                else:
                    b[2] += top3; b[3] += 1

    for kind, per_rank in buckets.items():
        ally_t = sum(b[0] for b in per_rank.values()); ally_n = sum(b[1] for b in per_rank.values())
        oth_t = sum(b[2] for b in per_rank.values()); oth_n = sum(b[3] for b in per_rank.values())
        # 得点順位で重み付けした「統制済み」仲間top3率（非仲間の順位分布に合わせる）
        adj_ally = adj_base = wsum = 0.0
        for rk, b in per_rank.items():
            if b[1] and (b[1] + b[3]):
                w = b[1]                       # 仲間の順位分布で重み付け
                adj_ally += w * (b[0] / b[1])
                adj_base += w * (b[2] / b[3]) if b[3] else 0.0
                wsum += w
        if ally_n and oth_n:
            raw = ally_t / ally_n
            print(f"  {kind:<4}仲間 top3率 {raw*100:5.1f}% (n={ally_n:5d}) vs 非仲間 {oth_t/oth_n*100:5.1f}% "
                  f"(n={oth_n:5d}) | 得点順位統制後 仲間 {adj_ally/wsum*100:.1f}% vs 基準 {adj_base/wsum*100:.1f}%"
                  if wsum else "")
    print("  ※ 統制後で『仲間>基準』なら、強さでは説明できないライン的恩恵の証拠。")


# --------------------------------------------------------------------------
# A3 model 1着確率で見た有力者の仲間 → モデル top3 期待を超えるか（records キャッシュ利用）
# --------------------------------------------------------------------------
def diag_model_residual(db_path, entry):
    print("\n【A3) モデル1着確率最大の車の同県仲間は、モデルtop3期待を上回るか（records=キャッシュ）】")
    try:
        from src.backtest.records_cache import get_records
        from src.backtest.selection import group_by_race, _first_place_probs
        records = get_records(db_path)          # rebuild しない（キャッシュ読込のみ）
    except Exception as e:                       # noqa: BLE001
        print(f"  records 取得失敗（キャッシュ未生成？）: {e}"); return
    races = group_by_race(records)
    print(f"  検証レコード {len(races)}レース")

    def car_top3_prob(recs):
        d = defaultdict(float)
        for r in recs:
            for c in set(r.combo):
                d[c] += r.model_prob
        return d

    ally_res = []; oth_res = []
    for rid, recs in races.items():
        if rid not in entry:
            continue
        es = entry[rid]
        fp = _first_place_probs(recs)
        if not fp:
            continue
        top = max(fp, key=fp.get)
        if top not in es:
            continue
        tp = es[top]["pref"]
        exp3 = car_top3_prob(recs)              # モデルの各車 top3 確率
        actual3 = set(next((r.combo for r in recs if r.is_win), ()))  # 実 top3 = 的中買い目の3車
        if not actual3:
            continue
        for c in es:
            if c == top or c not in exp3:
                continue
            res = int(c in actual3) - exp3[c]   # 実現 - モデル期待（残差）
            if tp is not None and es[c]["pref"] == tp:
                ally_res.append(res)
            else:
                oth_res.append(res)
    if ally_res and oth_res:
        print(f"  有力者の同県仲間 平均残差 {np.mean(ally_res):+.4f} (n={len(ally_res)}) "
              f"vs 非仲間 {np.mean(oth_res):+.4f} (n={len(oth_res)})")
        print("  ※ 仲間の残差が正で非仲間より大きい＝モデルがライン恩恵を過小評価＝関係性特徴に伸びしろ。")
    else:
        print("  サンプル不足。")


# --------------------------------------------------------------------------
# B 特徴評価（31 vs 31+関係性）
# --------------------------------------------------------------------------
def _tri10(model, test):
    hit = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        ranked = [k for k, _ in sorted(all_trifecta_probs(st).items(), key=lambda kv: -kv[1])]
        hit += int(tuple(s.order[:3]) in ranked[:10])
    return round(hit / len(test), 4) if test else 0.0


def _rel(vals):
    """レース内で平均0に相対化（欠損は0）。"""
    present = [v for v in vals if v is not None]
    mean = sum(present) / len(present) if present else 0.0
    return [(v - mean) if v is not None else 0.0 for v in vals]


# 0/1フラグは相対化せず生値、カウント系はレース内相対化する。
_FLAG_KEYS = {"ally_of_top", "ally_of_top_dist", "top_is_allied"}


def feature_eval(db_path):
    print("\n【B) 関係性特徴の寄与（本番31特徴 vs +関係性）】")
    rel = compute_relation_features(db_path)
    base = load_samples(db_path, features=PL_FEATURES_FULL)
    print(f"  学習サンプル {len(base)}レース（field_size=7）")
    if not base:
        return
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    s31 = augment_samples(base, db_path, feats31)

    def add_rel(samples):
        out = []
        for s in samples:
            s2 = copy.copy(s)
            cols = []
            for key in RELATION_KEYS:
                vals = [rel.get((s.race_id, c), {}).get(key) for c in s.car_numbers]
                col = vals if key in _FLAG_KEYS else _rel(vals)
                cols.append(np.array([v if v is not None else 0.0 for v in col]).reshape(-1, 1))
            s2.X = np.hstack([s.X] + cols)
            s2.feature_names = list(s.feature_names) + RELATION_KEYS
            out.append(s2)
        return out

    s_rel = add_rel(s31)
    tr0, te0 = time_split(s31, 0.30)
    tr1, te1 = time_split(s_rel, 0.30)
    m0, m1 = train_gbdt(tr0), train_gbdt(tr1)
    r0, r1 = evaluate(m0.strengths, te0), evaluate(m1.strengths, te1)
    print(f"  検証 test {len(te0)}レース")
    print(f"  {'指標':<12}{'31特徴':>12}{'+関係性':>12}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"  {k:<12}{r0[k]:>12}{r1[k]:>12}")
    print(f"  {'三連単top10':<12}{_tri10(m0, te0):>12}{_tri10(m1, te1):>12}")

    # 混戦サブセット（基準31モデルで分類）
    labels = {s.race_id: classify_race(m0.strengths(s.X, s.car_numbers)).label for s in te0}
    for lab in ("混戦", "標準", "軸堅"):
        ch0 = [s for s in te0 if labels.get(s.race_id) == lab]
        ch1 = [s for s in te1 if labels.get(s.race_id) == lab]
        if len(ch0) >= 30:
            c0, c1 = evaluate(m0.strengths, ch0), evaluate(m1.strengths, ch1)
            print(f"\n  [{lab} {len(ch0)}レース] top1 {c0['top1_acc']}→{c1['top1_acc']} / "
                  f"logloss {c0['logloss']}→{c1['logloss']} / ece {c0['ece']}→{c1['ece']} / "
                  f"tri10 {_tri10(m0, ch0)}→{_tri10(m1, ch1)}")


def main():
    ap = argparse.ArgumentParser(description="関係性(同県/同地区/同期)の相関診断＋特徴寄与")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--skip-model", action="store_true", help="A3(records)をスキップ")
    args = ap.parse_args()

    entry, result, races = _load_race_tables(args.db)
    print(f"診断対象: {len(races)}レース（field_size=7・結果確定）")
    diag_pairs(entry, result, races)
    diag_ally_benefit(entry, result, races)
    if not args.skip_model:
        diag_model_residual(args.db, entry)
    feature_eval(args.db)


if __name__ == "__main__":
    main()
