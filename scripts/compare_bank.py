"""選手×バンク長 適性特徴の有用性検証（読み取り専用・本番成果物は不変更）。

基準線 = 本番と同じ 20特徴 + rel_elo = 21特徴。そこへ compute_pre_race_bank の
バンク適性を「レース内相対化」して列追加し（22〜23特徴）、現行 vs +バンク特徴 を
lambdarank(train_gbdt) と PL(train_pl) の両方で比較する。

指標: 1着(top1_acc/logloss/brier/ece) + 三連単Top-k(k=1,3,10)。さらに test を
333/400/500 に層別して、薄い場でも改善するかを見る。判定は ece と logloss がともに
改善なら「有用」。

  PYTHONIOENCODING=utf-8 python scripts/compare_bank.py --db data/keirin.sqlite
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
from src.features.rider_bank import compute_pre_race_bank
from src.features.venue_meta import bank_length

try:
    from src.model.train_gbdt import train_gbdt
    _HAS_LGB, _LGB_ERR = True, ""
except Exception as e:  # LightGBM 未導入等
    _HAS_LGB, _LGB_ERR = False, str(e)


# ---------------------------------------------------------------- データ準備
def augment_elo(samples, pre_elo):
    """各サンプルXにレース内相対Elo(rel_elo)を1列追加（compare_models と同型・as-of安全）。"""
    out = []
    for s in samples:
        s2 = copy.copy(s)
        elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
        s2.X = np.hstack([s.X, (elos - elos.mean()).reshape(-1, 1)])
        s2.feature_names = list(s.feature_names) + ["rel_elo"]
        out.append(s2)
    return out


def augment_bank(samples, pre_bank):
    """バンク適性を列追加。勝率/上位率はレース内平均を引いて相対化、サポート数はlog圧縮。

    追加列: bank_win_rel（当該バンク勝率 - レース内平均）, bank_top3_rel（同・上位率）,
            bank_sup（log1p(bank_starts) をレース内平均で中心化＝サポートの相対厚み）。
    レース内相対化により「レース一定成分」を落とし、PL/ランカー双方で意味を持たせる。
    """
    out = []
    for s in samples:
        s2 = copy.copy(s)
        wins, top3s, sups = [], [], []
        for c in s.car_numbers:
            b = pre_bank.get((s.race_id, c))
            if b is None:
                wins.append(1.0 / 7.0); top3s.append(3.0 / 7.0); sups.append(0.0)
            else:
                wins.append(b["bank_win_shrunk"])
                top3s.append(b["bank_top3_shrunk"])
                sups.append(np.log1p(b["bank_starts"]))
        wins = np.array(wins); top3s = np.array(top3s); sups = np.array(sups)
        cols = np.column_stack([
            wins - wins.mean(),
            top3s - top3s.mean(),
            sups - sups.mean(),
        ])
        s2.X = np.hstack([s.X, cols])
        s2.feature_names = list(s.feature_names) + ["bank_win_rel", "bank_top3_rel", "bank_sup"]
        out.append(s2)
    return out


def bank_of_sample(db_path):
    """{race_id: bank(333/400/500 or None)} を返す（層別評価用）。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rows = conn.execute("SELECT race_id, venue_code FROM races").fetchall()
    finally:
        conn.close()
    return {rid: bank_length(vc) for rid, vc in rows}


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
    return {f"top{k}": round(hit[k] / n, 4) if n else 0.0 for k in ks}


# ---------------------------------------------------------------- 実行
def eval_all(model, test, label):
    r = evaluate(model.strengths, test)
    r.update(trifecta_topk(model, test))
    r["name"] = label
    return r


def print_block(title, rows):
    print(f"\n【{title}】")
    print(f"{'モデル':<22}{'n':>6}{'top1':>8}{'logloss':>9}{'brier':>9}{'ece':>9}"
          f"{'tri1':>8}{'tri3':>8}{'tri10':>8}")
    for r in rows:
        print(f"{r['name']:<22}{r['n']:>6}{r['top1_acc']:>8}{r['logloss']:>9}"
              f"{r['brier']:>9}{r['ece']:>9}{r['top1']:>8}{r['top3']:>8}{r['top10']:>8}")


def main():
    ap = argparse.ArgumentParser(description="選手×バンク長 適性特徴の有用性検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--k-shrink", type=int, default=20)
    args = ap.parse_args()

    if not _HAS_LGB:
        print(f"[警告] LightGBM を import できません: {_LGB_ERR}  → PLのみ評価")

    print("読み込み: samples(20特徴) + rel_elo ...")
    base = load_samples(args.db, features=PL_FEATURES_FULL)
    base21 = augment_elo(base, compute_pre_race_elo(args.db))
    print("バンク適性 as-of 集計 ...")
    pre_bank = compute_pre_race_bank(args.db, k_shrink=args.k_shrink)
    base_bank = augment_bank(base21, pre_bank)

    tr21, te21 = time_split(base21, args.test_frac)
    trbk, tebk = time_split(base_bank, args.test_frac)
    banks = bank_of_sample(args.db)
    print(f"サンプル {len(base21)}R（7車立て） / train {len(tr21)} / test {len(te21)}")
    print(f"基準線特徴 {len(base21[0].feature_names)}個 / +バンク {len(base_bank[0].feature_names)}個")
    print(f"追加列: {base_bank[0].feature_names[len(base21[0].feature_names):]}")

    # ---- 学習
    models = {}
    print("\n学習: PL(21) / PL(+bank) ...")
    models["PL 21特徴(現行)"] = (train_pl(tr21), te21, "PL")
    models["PL +bank"] = (train_pl(trbk), tebk, "PL")
    if _HAS_LGB:
        print("学習: LGB lambdarank(21) / (+bank) ...")
        models["LGB 21特徴(現行)"] = (train_gbdt(tr21), te21, "LGB")
        models["LGB +bank"] = (train_gbdt(trbk), tebk, "LGB")

    # ---- 全体評価
    rows = [eval_all(m, te, name) for name, (m, te, _) in models.items()]
    print_block("全体 test 評価（現行 vs +バンク特徴）", rows)

    # ---- バンク別 層別評価
    for bk in (333, 400, 500):
        sub_rows = []
        for name, (m, te, _) in models.items():
            te_sub = [s for s in te if banks.get(s.race_id) == bk]
            if not te_sub:
                continue
            sub_rows.append(eval_all(m, te_sub, name))
        if sub_rows:
            print_block(f"層別 test={bk}m（n={sub_rows[0]['n']}R）", sub_rows)

    # ---- LGB のバンク特徴 重要度
    if _HAS_LGB:
        m = models["LGB +bank"][0]
        try:
            gain = m.booster.feature_importance(importance_type="gain")
            fn = m.feature_names
            print("\n【LGB(+bank) 特徴重要度(gain) 上位 & バンク特徴】")
            order = np.argsort(gain)[::-1]
            bank_feats = {"bank_win_rel", "bank_top3_rel", "bank_sup"}
            for rank, i in enumerate(order, 1):
                mark = " <bank" if fn[i] in bank_feats else ""
                if rank <= 8 or fn[i] in bank_feats:
                    print(f"  {rank:>2}. {fn[i]:<18}{gain[i]:>12.1f}{mark}")
        except Exception as e:
            print(f"  重要度取得失敗: {e}")


if __name__ == "__main__":
    main()
