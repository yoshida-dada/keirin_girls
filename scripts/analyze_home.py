"""地元開催効果の検証: (A) 実測「地元3割増し」の定量化, (B) 特徴のモデル寄与。

A) 実測: 選手の登録県/地区 == 会場所在県/地区 の選手の 1着率・top3率・平均着順 を非地元と比較。
   競走得点で条件付け（実力を差し引いた地元補正）。会場別/地区別のばらつきも一瞥。
   さらに get_records（rebuild禁止・キャッシュ読込）のモデル確率に対する「超過的中」も見る。
B) 特徴評価: is_home_pref / is_home_district（per (race_id,car)）を本番31特徴へ足し、
   time_split で top1/logloss/brier/ece/三連単top10 を比較（analyze_narabi.py と同型）。レースタイプ層別。

  PYTHONIOENCODING=utf-8 python scripts/analyze_home.py --db data/keirin.sqlite

読み取り専用（PRAGMA query_only）。DB/本番は変更しない。get_records は rebuild しない。
"""
from __future__ import annotations

import argparse
import copy
import sqlite3
import sys
from collections import defaultdict
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
from src.features.venue_region import (
    is_home_pref, is_home_district, describe_venue, venue_district, pref_district)

SCORE_EDGES = [46, 48, 50, 52, 54]     # 競走得点の条件付けバケット境界


# ----------------------------------------------------------------------------
# 共通: (race_id, car) -> 地元フラグ / 得点 / 着順 を DB から読む（発走前確定＋着順は診断用）
# ----------------------------------------------------------------------------
def load_rider_rows(db_path):
    """field_size=7・着順ありの rider-row を返す。

    各要素: dict(race_id, car, venue, pref, score, pos, hp, hd)。
    pref/venue は発走前確定（特徴に使える）。pos(着順) は実測診断のみ（リーク厳禁で特徴には使わない）。
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rows = conn.execute(
            "SELECT e.race_id, e.car_number, r.venue_code, e.prefecture, e.racing_score, res.position"
            " FROM entries e JOIN races r ON e.race_id=r.race_id"
            " JOIN results res ON res.race_id=e.race_id AND res.car_number=e.car_number"
            " WHERE r.field_size=7 AND res.position IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    out = []
    for rid, car, venue, pref, score, pos in rows:
        out.append(dict(
            race_id=rid, car=car, venue=venue, pref=pref, score=score, pos=pos,
            hp=is_home_pref(pref, venue), hd=is_home_district(pref, venue)))
    return out


def _stat(rows):
    """rider-row 群の 1着率/top3率/平均着順/母数 を返す。"""
    n = len(rows)
    if not n:
        return dict(n=0, win=None, top3=None, avg=None)
    win = sum(1 for r in rows if r["pos"] == 1) / n
    top3 = sum(1 for r in rows if r["pos"] <= 3) / n
    avg = sum(r["pos"] for r in rows) / n
    return dict(n=n, win=win, top3=top3, avg=avg)


def _line(label, s):
    if s["n"] == 0:
        return f"  {label:<16} n=0"
    return (f"  {label:<16} n={s['n']:>6}  1着率 {s['win']*100:5.2f}%  "
            f"top3率 {s['top3']*100:5.2f}%  平均着順 {s['avg']:.3f}")


def compare(rows, flag):
    """flag(hp/hd) True/False で分けて実測を出力し、比率(地元/非地元)を返す。"""
    home = [r for r in rows if r[flag]]
    away = [r for r in rows if not r[flag]]
    sh, sa = _stat(home), _stat(away)
    print(_line("地元", sh)); print(_line("非地元", sa))
    if sh["n"] and sa["n"]:
        wr = sh["win"] / sa["win"] if sa["win"] else float("nan")
        tr = sh["top3"] / sa["top3"] if sa["top3"] else float("nan")
        print(f"  → 1着率比 {wr:.3f}倍 (地元/非地元)  top3率比 {tr:.3f}倍  "
              f"平均着順差 {sh['avg']-sa['avg']:+.3f}")
    return sh, sa


# ----------------------------------------------------------------------------
# A) 実測
# ----------------------------------------------------------------------------
def diagnostic_measured(rows):
    print("\n" + "=" * 78)
    print("【A) 地元効果の実測（field_size=7・着順あり）】")
    print(f"  総 rider-row: {len(rows)}")

    print("\n-- is_home_pref（登録県 == 会場県） --")
    compare(rows, "hp")
    print("\n-- is_home_district（登録地区 == 会場地区） --")
    compare(rows, "hd")

    # 競走得点で条件付け（実力を差し引いた地元補正）
    print("\n-- 競走得点で条件付け（同得点帯で 地元 vs 非地元） is_home_pref --")
    print(f"  {'得点帯':<12}{'地元n':>7}{'地元1着%':>9}{'非地元n':>8}{'非地元1着%':>11}"
          f"{'1着率差pt':>10}{'着順差':>9}")
    edges = SCORE_EDGES
    bands = [(-1e9, edges[0])] + [(edges[i], edges[i+1]) for i in range(len(edges)-1)] + [(edges[-1], 1e9)]
    labels = [f"<{edges[0]}"] + [f"{edges[i]}-{edges[i+1]}" for i in range(len(edges)-1)] + [f">={edges[-1]}"]
    for (lo, hi), lab in zip(bands, labels):
        sub = [r for r in rows if r["score"] is not None and lo <= r["score"] < hi]
        h = [r for r in sub if r["hp"]]; a = [r for r in sub if not r["hp"]]
        sh, sa = _stat(h), _stat(a)
        if sh["n"] and sa["n"]:
            print(f"  {lab:<12}{sh['n']:>7}{sh['win']*100:>8.2f}%{sa['n']:>8}{sa['win']*100:>10.2f}%"
                  f"{(sh['win']-sa['win'])*100:>+9.2f}{sh['avg']-sa['avg']:>+9.3f}")
        else:
            print(f"  {lab:<12}{sh['n']:>7}{'--':>9}{sa['n']:>8}{'--':>11}")

    # 会場別 is_home_pref のばらつき（地元選手が居た会場のみ・母数順）
    print("\n-- 会場別 is_home_pref 地元1着率（地元n>=30の会場） --")
    byv = defaultdict(lambda: {"h": [], "a": []})
    for r in rows:
        byv[r["venue"]]["h" if r["hp"] else "a"].append(r)
    recs = []
    for v, d in byv.items():
        sh, sa = _stat(d["h"]), _stat(d["a"])
        if sh["n"] >= 30 and sa["n"]:
            recs.append((sh["win"]-sa["win"], v, sh, sa))
    recs.sort(reverse=True)
    for diff, v, sh, sa in recs:
        print(f"  {describe_venue(v):<26} 地元 n={sh['n']:>4} 1着{sh['win']*100:5.2f}% / "
              f"非地元 1着{sa['win']*100:5.2f}%  差{diff*100:+5.2f}pt")

    # 地区別 is_home_district
    print("\n-- 地区別 is_home_district 地元1着率 --")
    byd = defaultdict(lambda: {"h": [], "a": []})
    for r in rows:
        vd = venue_district(r["venue"])
        if vd:
            byd[vd]["h" if r["hd"] else "a"].append(r)
    for d in sorted(byd):
        sh, sa = _stat(byd[d]["h"]), _stat(byd[d]["a"])
        if sh["n"] and sa["n"]:
            print(f"  {d:<8} 地元 n={sh['n']:>5} 1着{sh['win']*100:5.2f}% / "
                  f"非地元 1着{sa['win']*100:5.2f}%  差{(sh['win']-sa['win'])*100:+5.2f}pt")


# ----------------------------------------------------------------------------
# A2) モデル確率(get_records)に対する超過的中
# ----------------------------------------------------------------------------
def diagnostic_vs_model(db_path):
    print("\n" + "=" * 78)
    print("【A2) モデル確率に対する超過的中（get_records キャッシュ・rebuild禁止）】")
    try:
        from src.backtest.records_cache import get_records
        records = get_records(db_path)   # rebuild しない（キャッシュ読込）
    except Exception as e:  # noqa: BLE001
        print(f"  get_records 読込失敗: {e}"); return
    if not records:
        print("  records 空。スキップ。"); return

    # レースごとに 各車の モデルP(1着)/P(top3) を 210通りから再構成
    by_race = defaultdict(list)
    for r in records:
        by_race[r.race_id].append(r)
    model_win, model_top3 = {}, {}   # (race,car) -> prob
    for rid, recs in by_race.items():
        cars = {c for rec in recs for c in rec.combo}
        w = {c: 0.0 for c in cars}; t = {c: 0.0 for c in cars}
        for rec in recs:
            w[rec.combo[0]] += rec.model_prob
            for c in rec.combo:
                t[c] += rec.model_prob
        for c in cars:
            model_win[(rid, c)] = w[c]; model_top3[(rid, c)] = t[c]

    # 実測(entries/results) と突合。test期間のレースのみ。
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    rids = tuple(by_race)
    try:
        rows = conn.execute(
            "SELECT e.race_id, e.car_number, r.venue_code, e.prefecture, res.position"
            " FROM entries e JOIN races r ON e.race_id=r.race_id"
            " JOIN results res ON res.race_id=e.race_id AND res.car_number=e.car_number"
            " WHERE res.position IS NOT NULL").fetchall()
    finally:
        conn.close()
    rset = set(rids)
    agg = {}  # flagval -> [act_win, act_top3, mdl_win, mdl_top3, n]
    for grp in ("hp_home", "hp_away", "hd_home", "hd_away"):
        agg[grp] = [0.0, 0.0, 0.0, 0.0, 0]
    for rid, car, venue, pref, pos in rows:
        if rid not in rset or (rid, car) not in model_win:
            continue
        mw, mt = model_win[(rid, car)], model_top3[(rid, car)]
        aw, at = (1.0 if pos == 1 else 0.0), (1.0 if pos <= 3 else 0.0)
        for flag, gh, ga in (("hp", "hp_home", "hp_away"), ("hd", "hd_home", "hd_away")):
            home = is_home_pref(pref, venue) if flag == "hp" else is_home_district(pref, venue)
            g = agg[gh] if home else agg[ga]
            g[0] += aw; g[1] += at; g[2] += mw; g[3] += mt; g[4] += 1

    print(f"  test期間 {len(rset)}レース / 突合 rider-row {sum(agg[k][4] for k in ('hp_home','hp_away'))}")
    print(f"  {'群':<12}{'n':>7}{'実1着%':>8}{'モデル1着%':>11}{'超過pt':>8}"
          f"{'実top3%':>9}{'モデルtop3%':>12}{'超過pt':>8}")
    for lab, key in (("地元(県)", "hp_home"), ("非地元(県)", "hp_away"),
                     ("地元(地区)", "hd_home"), ("非地元(地区)", "hd_away")):
        aw, at, mw, mt, n = agg[key]
        if not n:
            continue
        print(f"  {lab:<12}{n:>7}{aw/n*100:>7.2f}%{mw/n*100:>10.2f}%{(aw-mw)/n*100:>+8.2f}"
              f"{at/n*100:>8.2f}%{mt/n*100:>11.2f}%{(at-mt)/n*100:>+8.2f}")
    print("  ※超過pt>0 = 実測がモデル予測を上回る（モデルが地元効果を織り込めていない分）。")


# ----------------------------------------------------------------------------
# B) 特徴評価
# ----------------------------------------------------------------------------
def _tri10(model, test):
    hit = 0
    for s in test:
        st = model.strengths(s.X, s.car_numbers)
        ranked = [k for k, _ in sorted(all_trifecta_probs(st).items(), key=lambda kv: -kv[1])]
        hit += int(tuple(s.order[:3]) in ranked[:10])
    return round(hit / len(test), 4) if test else 0.0


def _home_flags(db_path):
    """(race_id, car) -> (is_home_pref, is_home_district) を発走前確定情報から作る。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rows = conn.execute(
            "SELECT e.race_id, e.car_number, r.venue_code, e.prefecture"
            " FROM entries e JOIN races r ON e.race_id=r.race_id").fetchall()
    finally:
        conn.close()
    out = {}
    for rid, car, venue, pref in rows:
        out[(rid, car)] = (1.0 if is_home_pref(pref, venue) else 0.0,
                           1.0 if is_home_district(pref, venue) else 0.0)
    return out


def feature_eval(db_path):
    print("\n" + "=" * 78)
    print("【B) 地元特徴の寄与（31 vs 31+地元）】")
    base = load_samples(db_path, features=PL_FEATURES_FULL)
    print(f"  学習可能サンプル {len(base)}レース")
    if not base:
        print("  サンプル無し。スキップ。"); return
    feats31 = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
    s31 = augment_samples(base, db_path, feats31)
    flags = _home_flags(db_path)

    HOME_KEYS = ["is_home_pref", "is_home_district"]

    def add_home(samples):
        out = []
        for s in samples:
            s2 = copy.copy(s)
            hp = np.array([flags.get((s.race_id, c), (0.0, 0.0))[0] for c in s.car_numbers])
            hd = np.array([flags.get((s.race_id, c), (0.0, 0.0))[1] for c in s.car_numbers])
            s2.X = np.hstack([s.X, hp.reshape(-1, 1), hd.reshape(-1, 1)])
            s2.feature_names = list(s.feature_names) + HOME_KEYS
            out.append(s2)
        return out

    s_home = add_home(s31)
    tr0, te0 = time_split(s31, 0.30)
    tr1, te1 = time_split(s_home, 0.30)
    # 地元選手を含むtestレース割合（信号の届く範囲）
    with_home = sum(1 for s in te1 if any(
        flags.get((s.race_id, c), (0, 0))[0] for c in s.car_numbers))
    print(f"  検証 test {len(te0)}レース（うち地元県選手を含む {with_home}レース "
          f"{with_home/len(te0)*100:.1f}%）")
    m0, m1 = train_gbdt(tr0), train_gbdt(tr1)
    r0, r1 = evaluate(m0.strengths, te0), evaluate(m1.strengths, te1)
    print(f"  {'指標':<10}{'31特徴':>12}{'+地元':>12}{'差':>12}")
    for k in ("top1_acc", "logloss", "brier", "ece"):
        print(f"  {k:<10}{r0[k]:>12}{r1[k]:>12}{r1[k]-r0[k]:>+12.5f}")
    t0, t1 = _tri10(m0, te0), _tri10(m1, te1)
    print(f"  {'三連単top10':<10}{t0:>12}{t1:>12}{t1-t0:>+12.5f}")

    # 地元県選手を含むレースだけの層別（信号がある部分での効き）
    hsub0 = [s for s in te0 if any(flags.get((s.race_id, c), (0, 0))[0] for c in s.car_numbers)]
    hsub1 = [s for s in te1 if any(flags.get((s.race_id, c), (0, 0))[0] for c in s.car_numbers)]
    if hsub0:
        h0, h1 = evaluate(m0.strengths, hsub0), evaluate(m1.strengths, hsub1)
        print(f"\n  [地元県選手あり {len(hsub0)}レース] top1 {h0['top1_acc']}→{h1['top1_acc']} / "
              f"logloss {h0['logloss']}→{h1['logloss']} / ece {h0['ece']}→{h1['ece']} / "
              f"tri10 {_tri10(m0, hsub0)}→{_tri10(m1, hsub1)}")

    # レースタイプ層別
    labels = {s.race_id: classify_race(m0.strengths(s.X, s.car_numbers)).label for s in te0}
    for typ in ("軸堅", "標準", "混戦"):
        c0 = [s for s in te0 if labels.get(s.race_id) == typ]
        c1 = [s for s in te1 if labels.get(s.race_id) == typ]
        if len(c0) >= 20:
            e0, e1 = evaluate(m0.strengths, c0), evaluate(m1.strengths, c1)
            print(f"  [{typ} {len(c0)}レース] top1 {e0['top1_acc']}→{e1['top1_acc']} / "
                  f"logloss {e0['logloss']}→{e1['logloss']} / tri10 {_tri10(m0, c0)}→{_tri10(m1, c1)}")


def main():
    ap = argparse.ArgumentParser(description="地元開催効果の検証（実測＋特徴寄与）")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    args = ap.parse_args()
    rows = load_rider_rows(args.db)
    diagnostic_measured(rows)
    diagnostic_vs_model(args.db)
    feature_eval(args.db)


if __name__ == "__main__":
    main()
