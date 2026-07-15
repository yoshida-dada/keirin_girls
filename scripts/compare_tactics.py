"""展開特徴（選手展開の相対化版 A / レース展開 B）の有用性検証（読み取り専用）。

基準線 = 本番と同じ 拡張20 + rel_elo = 21特徴。そこへ段階投入(ablation):
    基準21 / +A / +B / +A+B
を lambdarank(train_gbdt) と PL(train_pl) の両方で比較する。

  A = 選手展開特徴（rider_tactics）の**レース内相対化版**（Sub-1 の最重要指摘に沿う）:
        t_lead_rel   = lead_index      − レース内平均
        t_leadsb_rel = lead_index_sb   − レース内平均
        t_sikake_rel = sikake          − レース内平均
        t_lastlap_rel= avg_last_lap    − レース内平均
        t_escape_rel = escape_survival − レース内平均
        t_legchg_rel = leg_change_rate − レース内平均
  B = レース展開特徴（race_dynamics）のうち**レース内で変動する列**:
        lead_margin / sikake_rel / escape_success / last_lap_rel
      ※ pace_mean/max/n600/std・lead_contest は「レース定数」で softmax/group内ランキングに
        寄与しない（Sub-1 の核心）ため、モデルには投入しない（race_dynamics は診断用に返す）。

指標: 1着(top1_acc/logloss/brier/ece) + 三連単Top-k(k=1,3,10)。
レースタイプ層別(軸堅/標準/混戦)を必ず出す（混戦=CHAOSで効く仮説を重点確認）。
採否: **ece を悪化させず** logloss と 三連単top10 が改善する部分集合を「採用推奨」。

  PYTHONIOENCODING=utf-8 python scripts/compare_tactics.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_pl import train_pl
from src.model.evaluate import evaluate, time_split
from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
from src.model.plackett_luce import all_trifecta_probs
from src.model.race_type import classify_race, JIKU, STANDARD, CHAOS
from src.features.rider_tactics import compute_pre_race_tactics
from src.features.race_dynamics import compute_pre_race_dynamics

try:
    from src.model.train_gbdt import train_gbdt
    _HAS_LGB, _LGB_ERR = True, ""
except Exception as e:  # LightGBM 未導入等
    _HAS_LGB, _LGB_ERR = False, str(e)

# --- A: 選手展開特徴の相対化版（rider_tactics のキー → 追加列名） ---
A_KEYS = ["lead_index", "lead_index_sb", "sikake", "avg_last_lap",
          "escape_survival", "leg_change_rate"]
A_NAMES = ["t_lead_rel", "t_leadsb_rel", "t_sikake_rel", "t_lastlap_rel",
           "t_escape_rel", "t_legchg_rel"]
# --- B: レース展開特徴のうちレース内変動列（race_dynamics のキー） ---
B_NAMES = ["lead_margin", "sikake_rel", "escape_success", "last_lap_rel"]


# ---------------------------------------------------------------- データ準備
def augment_elo(samples, pre_elo):
    """各サンプルXにレース内相対Elo(rel_elo)を1列追加（本番基準線と同型・as-of安全）。"""
    out = []
    for s in samples:
        s2 = copy.copy(s)
        elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
        s2.X = np.hstack([s.X, (elos - elos.mean()).reshape(-1, 1)])
        s2.feature_names = list(s.feature_names) + ["rel_elo"]
        out.append(s2)
    return out


def _rel_column(vals: list[float | None]) -> np.ndarray:
    """レース内相対化: present の平均を引く。欠損は0（=レース平均扱い）。eval_rolling と同型。"""
    present = [v for v in vals if v is not None]
    mean = sum(present) / len(present) if present else 0.0
    return np.array([(v - mean) if v is not None else 0.0 for v in vals])


def augment_A(samples, tactics):
    """選手展開特徴（rider_tactics 絶対値）をレース内相対化して列追加。"""
    out = []
    for s in samples:
        s2 = copy.copy(s)
        cols = []
        for key in A_KEYS:
            vals = [tactics.get((s.race_id, c), {}).get(key) for c in s.car_numbers]
            cols.append(_rel_column(vals).reshape(-1, 1))
        s2.X = np.hstack([s.X] + cols)
        s2.feature_names = list(s.feature_names) + A_NAMES
        out.append(s2)
    return out


def augment_B(samples, dynamics):
    """レース展開特徴（race_dynamics のレース内変動列）を列追加。"""
    out = []
    for s in samples:
        s2 = copy.copy(s)
        cols = []
        for key in B_NAMES:
            vals = [dynamics.get((s.race_id, c), {}).get(key, 0.0) for c in s.car_numbers]
            cols.append(np.array([v if v is not None else 0.0 for v in vals]).reshape(-1, 1))
        s2.X = np.hstack([s.X] + cols)
        s2.feature_names = list(s.feature_names) + B_NAMES
        out.append(s2)
    return out


# ---------------------------------------------------------------- 三連単指標
def rank_pos(probs: dict, actual: tuple) -> int:
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    for i, (k, _) in enumerate(ranked, 1):
        if k == actual:
            return i
    return 10 ** 9


def trifecta_topk(model, test):
    """三連単Top-k(k=1,3,10)的中率。PL連鎖(all_trifecta_probs)で210通り確率化。"""
    ks = (1, 3, 10)
    hit = {k: 0 for k in ks}
    n = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        probs = all_trifecta_probs(st)
        pos = rank_pos(probs, tuple(s.order[:3]))
        for k in ks:
            hit[k] += int(pos <= k)
        n += 1
    return {f"tri{k}": round(hit[k] / n, 4) if n else 0.0 for k in ks}


def eval_all(model, test, label):
    r = evaluate(model.strengths, test)
    if r.get("n", 0) == 0:
        return {"name": label, "n": 0}
    r.update(trifecta_topk(model, test))
    r["name"] = label
    return r


# ---------------------------------------------------------------- レースタイプ
def race_type_labels(base_model, base_test):
    """基準21特徴モデルの1着確率で各testレースを 軸堅/標準/混戦 に分類（層別の基準を固定）。"""
    labels = {}
    for s in base_test:
        st = base_model.strengths(s.X, s.car_numbers)
        labels[s.race_id] = classify_race(st).label
    return labels


# ---------------------------------------------------------------- 出力
def print_block(title, rows):
    print(f"\n【{title}】")
    print(f"{'モデル':<20}{'n':>6}{'top1':>8}{'logloss':>9}{'brier':>9}{'ece':>9}"
          f"{'tri1':>8}{'tri3':>8}{'tri10':>8}")
    for r in rows:
        if r.get("n", 0) == 0:
            print(f"{r['name']:<20}{'(該当レースなし)':>40}")
            continue
        print(f"{r['name']:<20}{r['n']:>6}{r['top1_acc']:>8}{r['logloss']:>9}"
              f"{r['brier']:>9}{r['ece']:>9}{r['tri1']:>8}{r['tri3']:>8}{r['tri10']:>8}")


def main():
    ap = argparse.ArgumentParser(description="展開特徴(A:選手相対化 / B:レース展開)の有用性検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    if not _HAS_LGB:
        print(f"[警告] LightGBM を import できません: {_LGB_ERR}  → PLのみ評価")

    print("読み込み: samples(拡張20) + rel_elo(=21特徴 基準線) ...")
    base = load_samples(args.db, features=PL_FEATURES_FULL)
    base21 = augment_elo(base, compute_pre_race_elo(args.db))
    print("as-of 展開特徴を集計: rider_tactics / race_dynamics ...")
    tactics = compute_pre_race_tactics(args.db)
    dynamics = compute_pre_race_dynamics(args.db)

    # 4条件のサンプル（同一 race 順・同一 order。time_split はインデックス整合）
    cond = {
        "基準21": base21,
        "+A": augment_A(base21, tactics),
        "+B": augment_B(base21, dynamics),
        "+A+B": augment_B(augment_A(base21, tactics), dynamics),
    }
    splits = {k: time_split(v, args.test_frac) for k, v in cond.items()}
    n_tr = len(splits["基準21"][0]); n_te = len(splits["基準21"][1])
    print(f"サンプル {len(base21)}R（7車立て） / train {n_tr} / test {n_te}")
    print(f"基準21: {len(base21[0].feature_names)}列")
    print(f"  A 追加列({len(A_NAMES)}): {A_NAMES}")
    print(f"  B 追加列({len(B_NAMES)}): {B_NAMES}")
    print("  ※ race_dynamics のレース定数列(pace_*, lead_contest)はシフト不変のため投入外")

    # ---- 学習（PL / LGB を各条件で）
    algos = [("PL", train_pl)]
    if _HAS_LGB:
        algos.append(("LGB", train_gbdt))
    trained = {}   # (algo, cond) -> model
    for aname, fn in algos:
        for cname, (tr, te) in splits.items():
            print(f"学習: {aname} {cname} ...")
            trained[(aname, cname)] = fn(tr)

    # ---- レースタイプ層別の基準ラベル（基準21 PLモデルの1着確率で分類・固定）
    base_pl = trained[("PL", "基準21")]
    _, base_te = splits["基準21"]
    labels = race_type_labels(base_pl, base_te)
    from collections import Counter
    cnt = Counter(labels.values())
    print(f"\nレースタイプ内訳(test {n_te}R, 基準21 PLで分類): "
          f"軸堅 {cnt.get(JIKU,0)} / 標準 {cnt.get(STANDARD,0)} / 混戦 {cnt.get(CHAOS,0)}")

    # ---- 全体 & レースタイプ層別評価
    for aname, _ in algos:
        rows = [eval_all(trained[(aname, c)], splits[c][1], c) for c in cond]
        print_block(f"[{aname}] 全体 test 評価（段階投入）", rows)

        for lab in (JIKU, STANDARD, CHAOS):
            sub_rows = []
            for c in cond:
                te = splits[c][1]
                te_sub = [s for s in te if labels.get(s.race_id) == lab]
                sub_rows.append(eval_all(trained[(aname, c)], te_sub, c))
            print_block(f"[{aname}] 層別: {lab}（n={sub_rows[0].get('n',0)}R）", sub_rows)

    # ---- LGB(+A+B) 特徴重要度（gain）
    if _HAS_LGB:
        m = trained[("LGB", "+A+B")]
        try:
            gain = m.booster.feature_importance(importance_type="gain")
            fn = m.feature_names
            add_feats = set(A_NAMES) | set(B_NAMES)
            order = np.argsort(gain)[::-1]
            print("\n【LGB(+A+B) 特徴重要度(gain) 上位 & 展開特徴】")
            for rank, i in enumerate(order, 1):
                tag = ""
                if fn[i] in A_NAMES:
                    tag = " <A"
                elif fn[i] in B_NAMES:
                    tag = " <B"
                if rank <= 10 or fn[i] in add_feats:
                    print(f"  {rank:>2}. {fn[i]:<16}{gain[i]:>12.1f}{tag}")
        except Exception as e:
            print(f"  重要度取得失敗: {e}")

    # ---- 採否判定（ece 非悪化 かつ logloss と tri10 改善）
    print("\n【採否判定（基準21比: ece≤基準 かつ logloss改善 かつ tri10改善 → 採用推奨）】")
    for aname, _ in algos:
        base_r = eval_all(trained[(aname, "基準21")], splits["基準21"][1], "基準21")
        for c in ("+A", "+B", "+A+B"):
            r = eval_all(trained[(aname, c)], splits[c][1], c)
            ok = (r["ece"] <= base_r["ece"] + 1e-9 and r["logloss"] < base_r["logloss"]
                  and r["tri10"] > base_r["tri10"])
            d = (f"Δlogloss={r['logloss']-base_r['logloss']:+.4f} "
                 f"Δece={r['ece']-base_r['ece']:+.5f} Δtri10={r['tri10']-base_r['tri10']:+.4f}")
            print(f"  {aname} {c:<5} {'採用推奨' if ok else '見送り'}  {d}")


if __name__ == "__main__":
    main()
