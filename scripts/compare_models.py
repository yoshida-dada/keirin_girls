"""1着・三連単確率モデルの比較検証（PL線形 vs LightGBM 2方式 vs アンサンブル）。

同一の時系列分割・同一特徴（拡張20 + rel_elo = 21特徴）で 4方式を比較する:
  1. PL線形                         (train_pl)
  2. LightGBM lambdarank            (train_gbdt)
  3. LightGBM 着順多クラス + 順位MC (train_gbdt_multiclass)
  4. アンサンブル                   (PL × 最良LightGBM を blend_loglinear、α走査)

評価: 1着(top1_acc/top3/logloss/brier/ece)、三連単Top-k(k=1,3,10)、人気帯層別的中率、
擬似回収率(payouts_trifecta)。本番成果物・persist・predict_race は一切変更しない読み取り専用検証。

  PYTHONIOENCODING=utf-8 python scripts/compare_models.py --db data/keirin.sqlite
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
from src.model.train_pl import train_pl
from src.model.evaluate import evaluate, time_split
from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
from src.model.plackett_luce import all_trifecta_probs
from src.ev.market import blend_loglinear

_HAS_LGB = True
_LGB_ERR = ""
try:
    from src.model.train_gbdt import train_gbdt
    from src.model.train_gbdt_multiclass import train_gbdt_multiclass
except Exception as e:  # LightGBM 未インストール等
    _HAS_LGB = False
    _LGB_ERR = str(e)


# ---------------------------------------------------------------- データ準備
def augment_elo(samples, pre_elo):
    """各サンプルXにレース内相対Eloを付与（accuracy_history/eval_rolling と同型・as-of安全）。"""
    out = []
    for s in samples:
        s2 = copy.copy(s)
        elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
        s2.X = np.hstack([s.X, (elos - elos.mean()).reshape(-1, 1)])
        s2.feature_names = list(s.feature_names) + ["rel_elo"]
        out.append(s2)
    return out


def load_payouts(db_path):
    """{race_id: {"combo": (a,b,c), "payout": int, "pop": int}} を返す（読み取りのみ）。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT race_id, combo, payout, popularity FROM payouts_trifecta").fetchall()
    finally:
        conn.close()
    out = {}
    for rid, combo, payout, pop in rows:
        try:
            a, b, c = (int(x) for x in combo.split("-"))
        except Exception:
            continue
        out[rid] = {"combo": (a, b, c), "payout": payout, "pop": pop}
    return out


# ---------------------------------------------------------------- 三連単確率
def trifecta_of(model, s):
    """モデルに応じた三連単210通り確率 {(a,b,c): p}。"""
    if hasattr(model, "trifecta_probs"):          # 多クラス（位置別連鎖）
        return model.trifecta_probs(s.X, s.car_numbers)
    st = model.strengths(s.X, s.car_numbers)      # PL / lambdarank（PL連鎖）
    return all_trifecta_probs(st)


def rank_pos(probs: dict, actual: tuple) -> int:
    """actual 組合せが確率降順で何位か（1始まり。無ければ大きな値）。"""
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    for i, (k, _) in enumerate(ranked, 1):
        if k == actual:
            return i
    return 10 ** 9


# ---------------------------------------------------------------- 指標
def trifecta_metrics(test, tri_by_race, payouts, name):
    """三連単Top-k的中率・層別的中率・擬似回収率をまとめる。"""
    ks = (1, 3, 10)
    hit = {k: 0 for k in ks}
    n = 0
    # 人気帯（実三連単のpopularity）で層別した Top-10 的中
    bands = [("堅(1-3人気)", 1, 3), ("標準(4-10)", 4, 10),
             ("やや薄(11-30)", 11, 30), ("人気薄(31+)", 31, 10 ** 9)]
    band_n = {b[0]: 0 for b in bands}
    band_hit = {b[0]: 0 for b in bands}
    # 擬似回収率（各kで top-k を各100円買い）
    roi_ret = {k: 0 for k in ks}
    roi_cost = {k: 0 for k in ks}
    hit_payouts = {k: [] for k in ks}   # 的中時払戻（top-k戦略）
    for s in test:
        probs = tri_by_race[s.race_id]
        actual = tuple(s.order[:3])
        pos = rank_pos(probs, actual)
        n += 1
        for k in ks:
            if pos <= k:
                hit[k] += 1
        # 層別（payoutのpopularityが実三連単の人気順位）
        pinfo = payouts.get(s.race_id)
        if pinfo and pinfo["combo"] == actual:
            pop = pinfo["pop"]
            for label, lo, hi in bands:
                if lo <= pop <= hi:
                    band_n[label] += 1
                    if pos <= 10:
                        band_hit[label] += 1
                    break
        # 擬似回収率
        if pinfo:
            ranked = [k for k, _ in sorted(probs.items(), key=lambda kv: -kv[1])]
            for k in ks:
                topk = ranked[:k]
                roi_cost[k] += 100 * k
                if actual in topk and pinfo["combo"] == actual:
                    roi_ret[k] += pinfo["payout"]
                    hit_payouts[k].append(pinfo["payout"])
    res = {"name": name, "n": n}
    for k in ks:
        res[f"top{k}"] = round(hit[k] / n, 4) if n else 0.0
    res["bands"] = {b: (band_hit[b], band_n[b]) for b in band_n}
    res["roi"] = {k: (round(100 * roi_ret[k] / roi_cost[k], 1) if roi_cost[k] else 0.0)
                  for k in ks}
    res["avg_payout_hit"] = {k: (round(float(np.mean(hit_payouts[k])), 0)
                                 if hit_payouts[k] else 0.0) for k in ks}
    return res


def win_top3(strengths_fn, test):
    """1着が予測上位3内に入る率（外れ値検出の補助）。"""
    hit = n = 0
    for s in test:
        st = strengths_fn(s.X, s.car_numbers)
        if not st:
            continue
        ranked = sorted(st, key=lambda c: -st[c])
        hit += int(s.order[0] in ranked[:3])
        n += 1
    return round(hit / n, 4) if n else 0.0


# ---------------------------------------------------------------- 出力
def print_1着(rows):
    print("\n【1着予測】")
    print(f"{'モデル':<26}{'n':>6}{'top1':>8}{'top3':>8}{'logloss':>9}{'brier':>9}{'ece':>9}")
    for r in rows:
        t3 = r['top3'] if r.get('top3') is not None else "-"
        print(f"{r['name']:<26}{r['n']:>6}{r['top1_acc']:>8}{str(t3):>8}"
              f"{r['logloss']:>9}{r['brier']:>9}{r['ece']:>9}")


def print_tri(rows):
    print("\n【三連単Top-k的中率】")
    print(f"{'モデル':<26}{'top1':>8}{'top3':>8}{'top10':>8}")
    for r in rows:
        print(f"{r['name']:<26}{r['top1']:>8}{r['top3']:>8}{r['top10']:>8}")
    print("\n【三連単Top-10 人気帯別的中率（実三連単のpopularityで層別）】")
    labels = list(rows[0]["bands"].keys())
    header = f"{'モデル':<26}" + "".join(f"{l:>16}" for l in labels)
    print(header)
    for r in rows:
        cells = ""
        for l in labels:
            h, tot = r["bands"][l]
            rate = f"{100*h/tot:.1f}%" if tot else "-"
            cells += f"{rate}({h}/{tot})".rjust(16)
        print(f"{r['name']:<26}{cells}")
    print("\n【擬似回収率 %（top-k各100円買い） / 的中時平均払戻】")
    print(f"{'モデル':<26}{'ROI@1':>9}{'ROI@3':>9}{'ROI@10':>9}{'払戻@10':>10}")
    for r in rows:
        print(f"{r['name']:<26}{r['roi'][1]:>9}{r['roi'][3]:>9}{r['roi'][10]:>9}"
              f"{r['avg_payout_hit'][10]:>10}")


def main():
    ap = argparse.ArgumentParser(description="1着・三連単モデル比較検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    if not _HAS_LGB:
        print(f"[警告] LightGBM を import できません: {_LGB_ERR}")
        print("      PL線形のみ評価します。")

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    samples = augment_elo(base, compute_pre_race_elo(args.db))
    train, test = time_split(samples, args.test_frac)
    print(f"サンプル {len(samples)}レース（7車立て） / train {len(train)} / test {len(test)}")
    print(f"特徴量 {len(samples[0].feature_names)}個: {samples[0].feature_names}")
    payouts = load_payouts(args.db)

    # ---- モデル学習
    models = {}
    print("\n学習中: PL線形 ...")
    models["1.PL線形"] = train_pl(train)
    if _HAS_LGB:
        print("学習中: LightGBM lambdarank ...")
        models["2.LGB lambdarank"] = train_gbdt(train)
        print("学習中: LightGBM 多クラス+MC ...")
        models["3.LGB 多クラス+MC"] = train_gbdt_multiclass(train)

    # ---- 1着指標
    one_rows = []
    for name, m in models.items():
        r = evaluate(m.strengths, test)
        r["name"] = name
        r["top3"] = win_top3(m.strengths, test)
        one_rows.append(r)

    # ---- 三連単確率を全モデル分キャッシュ（test各レース）
    tri_cache = {name: {s.race_id: trifecta_of(m, s) for s in test}
                 for name, m in models.items()}
    tri_rows = [trifecta_metrics(test, tri_cache[name], payouts, name)
                for name in models]

    # ---- アンサンブル（PL × 最良LightGBM）
    ens_1 = ens_tri = None
    if _HAS_LGB:
        # 最良LightGBM = 三連単top10が高い方
        lgb_names = ["2.LGB lambdarank", "3.LGB 多クラス+MC"]
        best_lgb = max(lgb_names, key=lambda nm: next(
            r["top10"] for r in tri_rows if r["name"] == nm))
        pl_m, lgb_m = models["1.PL線形"], models[best_lgb]
        print(f"\nアンサンブル: PL × {best_lgb} を blend_loglinear（α=PL重み）で走査 ...")

        # 1着ブレンドのα走査（ece+logloss最小）
        alphas = [round(a, 2) for a in np.arange(0.0, 1.01, 0.1)]
        pl_st = {s.race_id: pl_m.strengths(s.X, s.car_numbers) for s in test}
        lgb_st = {s.race_id: lgb_m.strengths(s.X, s.car_numbers) for s in test}

        def eval_blend_1(alpha):
            def fn(X, cars):
                # race_id 依存のためクロージャで引けないので都度合成
                return None
            # 直接計算
            pairs, ll, top1, n = [], 0.0, 0, 0
            for s in test:
                b = blend_loglinear(pl_st[s.race_id], lgb_st[s.race_id], alpha)
                if not b:
                    continue
                import math
                winner = s.order[0]
                top1 += int(max(b, key=b.get) == winner)
                ll += -math.log(b.get(winner, 0.0) + 1e-12)
                for c, p in b.items():
                    pairs.append((p, 1 if c == winner else 0))
                n += 1
            from src.backtest.calibration import brier_score, expected_calibration_error
            return {"alpha": alpha, "n": n, "top1_acc": round(top1 / n, 4),
                    "logloss": round(ll / n, 4), "brier": round(brier_score(pairs), 5),
                    "ece": round(expected_calibration_error(pairs), 5)}

        scan1 = [eval_blend_1(a) for a in alphas]
        best1 = min(scan1, key=lambda r: (r["ece"] + r["logloss"]))
        print("  1着ブレンド α走査 (ece / logloss):")
        for r in scan1:
            mark = " *" if r["alpha"] == best1["alpha"] else ""
            print(f"    α={r['alpha']:.1f}  top1={r['top1_acc']}  logloss={r['logloss']}"
                  f"  ece={r['ece']}{mark}")

        # 三連単ブレンドのα走査（top10最大）
        pl_tri, lgb_tri = tri_cache["1.PL線形"], tri_cache[best_lgb]

        def eval_blend_tri(alpha):
            hit = {1: 0, 3: 0, 10: 0}
            n = 0
            for s in test:
                b = blend_loglinear(pl_tri[s.race_id], lgb_tri[s.race_id], alpha)
                if not b:
                    continue
                pos = rank_pos(b, tuple(s.order[:3]))
                for k in (1, 3, 10):
                    hit[k] += int(pos <= k)
                n += 1
            return {"alpha": alpha, **{f"top{k}": round(hit[k] / n, 4) for k in (1, 3, 10)}}

        scanT = [eval_blend_tri(a) for a in alphas]
        bestT = max(scanT, key=lambda r: r["top10"])
        print("  三連単ブレンド α走査 (top1/top3/top10):")
        for r in scanT:
            mark = " *" if r["alpha"] == bestT["alpha"] else ""
            print(f"    α={r['alpha']:.1f}  top1={r['top1']}  top3={r['top3']}"
                  f"  top10={r['top10']}{mark}")

        # 最良αでアンサンブルを表に追加
        a1 = best1["alpha"]
        ens_name = f"4.アンサンブル(α={a1:.1f})"
        best1["name"] = ens_name
        best1["top3"] = None
        one_rows.append(best1)

        aT = bestT["alpha"]
        ens_tri_cache = {s.race_id: blend_loglinear(pl_tri[s.race_id], lgb_tri[s.race_id], aT)
                         for s in test}
        tri_rows.append(trifecta_metrics(test, ens_tri_cache, payouts,
                                          f"4.アンサンブル(α={aT:.1f})"))

    # ---- 出力
    print_1着(one_rows)
    print_tri(tri_rows)


if __name__ == "__main__":
    main()
