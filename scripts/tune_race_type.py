"""レースタイプ分類（軸堅/標準/混戦）のしきい値チューニング分析（読み取り専用）。

現在の entropy_norm しきい値 (0.80, 0.93) が「予測しやすさ（＝軸堅ほど的中しやすい）」で
レースを分離できているかを検証し、より良いしきい値を提案する。

  python scripts/tune_race_type.py --db data/keirin.sqlite

処理:
  1) load_samples(db, features=PL_FEATURES_FULL) で3年分を読む。
  2) load_model の学習済みモデルで各レースの1着確率(win_probs, Σ=1)を得る。
     モデルが rel_elo を使う場合は compute_pre_race_elo で X に rel_elo 列を足す
     （retrain_3yr._augment と同じ拡張）。
  3) classify_race(win_probs).entropy_norm と、実データの的中を紐付ける:
       - 1着的中 : モデル本命(argmax) == 実1着
       - 上位3内 : 実1着がモデル1着確率 上位3車に入る
       - 三連単10: 実三連単(実1-2-3着) がモデル三連単 上位10 に入る
  4) entropy_norm を 0.05 刻みでビン化し、ビン別の的中率を出す。
  5) しきい値候補 (下限×上限) を総当たりし、タイプ別レース数・的中率を集計。
  6) タイプ間で的中率がよく分離し、各タイプに十分なサンプルがあるしきい値を推奨。

方針: 本番モデル(pl_model.pkl)は全3年で学習済みのため、全期間評価は in-sample。
      よって「全期間」に加えて「検証期間(後ろ25%, time_split)」でも同じ分析を行い、
      推奨は両者で整合するものを選ぶ。数値は実データ実行結果に基づく。
"""
from __future__ import annotations

import argparse
import copy
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# 日本語ラベルを含むため stdout を UTF-8 に固定（Windows既定cp932の文字化け回避）
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.evaluate import time_split
from src.model.persist import load_model, DEFAULT_MODEL_PATH
from src.model.elo import compute_pre_race_elo, DEFAULT_ELO
from src.model.race_type import classify_race, DEFAULT_ENTROPY_EDGES, JIKU, STANDARD, CHAOS
from src.model.plackett_luce import all_trifecta_probs

# しきい値候補（下限 × 上限）
# 現行付近の「指定グリッド」（下限0.75/0.80/0.85 × 上限0.90/0.93/0.95）
LO_MANDATED = (0.75, 0.80, 0.85)
HI_MANDATED = (0.90, 0.93, 0.95)
# 実データの entropy 分布が低め寄りに偏るため、バランス良く3分割できる低めの拡張グリッド
LO_EXTENDED = (0.50, 0.55, 0.60, 0.65)
HI_EXTENDED = (0.68, 0.72, 0.78, 0.85)
BIN_W = 0.05           # entropy_norm ビン幅
TRIFECTA_TOPN = 10     # 三連単「上位N」的中の N
MIN_TYPE_FRAC = 0.15   # 各タイプに最低これだけのレース割合が必要（推奨の必須条件）


def augment_with_elo(samples, pre_elo):
    """retrain_3yr._augment と同じ: X に rel_elo 列（レース内平均差）を追加。"""
    out = []
    for s in samples:
        s2 = copy.copy(s)
        elos = np.array([pre_elo.get((s.race_id, c), DEFAULT_ELO) for c in s.car_numbers])
        s2.X = np.hstack([s.X, (elos - elos.mean()).reshape(-1, 1)])
        s2.feature_names = list(s.feature_names) + ["rel_elo"]
        out.append(s2)
    return out


def build_rows(samples, model):
    """各レース → (entropy_norm, hit_top1, hit_top3, hit_tri10)。"""
    rows = []
    for s in samples:
        win_probs = model.strengths(s.X, s.car_numbers)   # {車番: P(1着)} Σ=1
        if not win_probs:
            continue
        ent = classify_race(win_probs).entropy_norm
        winner = s.order[0]
        # 1着的中
        pred = max(win_probs, key=win_probs.get)
        hit_top1 = int(pred == winner)
        # 実1着がモデル上位3車か
        top3_cars = [c for c, _ in sorted(win_probs.items(), key=lambda kv: -kv[1])[:3]]
        hit_top3 = int(winner in top3_cars)
        # 実三連単がモデル三連単 上位N に入るか（上位3着が確定しているレースのみ）
        hit_tri = 0
        if len(s.order) >= 3:
            tri = all_trifecta_probs(win_probs)
            topn = {k for k, _ in sorted(tri.items(), key=lambda kv: -kv[1])[:TRIFECTA_TOPN]}
            hit_tri = int(tuple(s.order[:3]) in topn)
        rows.append((ent, hit_top1, hit_top3, hit_tri))
    return rows


def _rate(rows, idx):
    return (sum(r[idx] for r in rows) / len(rows)) if rows else None


def _fmt(x):
    return f"{x*100:5.1f}%" if x is not None else "   -  "


def bin_table(rows):
    """entropy_norm を BIN_W 刻みにビン化して的中率表を出す。"""
    buckets = defaultdict(list)
    for r in rows:
        b = min(int(r[0] / BIN_W), int(1.0 / BIN_W) - 1)   # 0.05刻み、1.0は最終ビンへ
        buckets[b].append(r)
    print(f"{'entropyビン':>14}{'レース数':>8}{'1着的中':>9}{'上位3内':>9}{'三連単Top10':>12}")
    for b in sorted(buckets):
        lo, hi = b * BIN_W, (b + 1) * BIN_W
        rs = buckets[b]
        print(f"{f'{lo:.2f}-{hi:.2f}':>14}{len(rs):>8}"
              f"{_fmt(_rate(rs, 1)):>9}{_fmt(_rate(rs, 2)):>9}{_fmt(_rate(rs, 3)):>12}")


def classify_rows(rows, lo, hi):
    """しきい値(lo,hi)で rows をタイプ別に振り分ける。"""
    groups = {JIKU: [], STANDARD: [], CHAOS: []}
    for r in rows:
        ent = r[0]
        if ent < lo:
            groups[JIKU].append(r)
        elif ent > hi:
            groups[CHAOS].append(r)
        else:
            groups[STANDARD].append(r)
    return groups


def _eval_edges(rows, lo, hi):
    """(lo,hi) でのタイプ別集計と評価指標を返す。"""
    n_total = len(rows)
    g = classify_rows(rows, lo, hi)
    cells = {lbl: (len(g[lbl]), _rate(g[lbl], 1), _rate(g[lbl], 2), _rate(g[lbl], 3))
             for lbl in (JIKU, STANDARD, CHAOS)}
    j1, s1, c1 = cells[JIKU][1], cells[STANDARD][1], cells[CHAOS][1]
    # 端の分離（軸堅−混戦, 1着的中）
    sep = (j1 - c1) if (j1 is not None and c1 is not None) else None
    # 隣接ギャップの最小値（3タイプが均等に分離しているか＝推奨の主指標）
    if None not in (j1, s1, c1):
        min_gap = min(j1 - s1, s1 - c1)
    else:
        min_gap = None
    min_frac = min(cells[lbl][0] for lbl in (JIKU, STANDARD, CHAOS)) / n_total if n_total else 0
    enough = min_frac >= MIN_TYPE_FRAC
    return {"lo": lo, "hi": hi, "sep": sep, "min_gap": min_gap,
            "enough": enough, "min_frac": min_frac, "cells": cells}


def threshold_table(rows, lo_cands, hi_cands, title):
    """しきい値候補を総当たりし、タイプ別レース数・的中率を集計して評価。"""
    print(f"  [{title}]")
    print(f"{'しきい値(lo,hi)':>16}"
          f"{'軸堅 n/1着/3内/Tri':>26}{'標準 n/1着/3内/Tri':>26}{'混戦 n/1着/3内/Tri':>26}"
          f"{'軸-混':>8}{'最小隣接':>9}")
    results = []
    for lo in lo_cands:
        for hi in hi_cands:
            if lo >= hi:
                continue
            r = _eval_edges(rows, lo, hi)
            cells = r["cells"]

            def cell(lbl):
                n, a, b, c = cells[lbl]
                return f"{n:>4} {_fmt(a)}{_fmt(b)}{_fmt(c)}"
            mark = "" if r["enough"] else " *"
            print(f"{f'({lo:.2f},{hi:.2f})':>16}"
                  f"{cell(JIKU):>26}{cell(STANDARD):>26}{cell(CHAOS):>26}"
                  f"{_fmt(r['sep']):>8}{(_fmt(r['min_gap']) + mark):>9}")
            results.append(r)
    print("  * = いずれかのタイプがレース割合 < "
          f"{MIN_TYPE_FRAC*100:.0f}%（サンプル不足）で推奨から除外")
    return results


def recommend(results):
    """サンプル十分な候補のうち、3タイプの隣接的中率ギャップの最小値が最大のものを推奨。

    端(軸堅−混戦)の分離だけを最大化すると混戦バケットを潰す退化解に陥るため、
    「軸堅>標準>混戦 が均等に分離し、各タイプに十分サンプル」を主目的にする。
    """
    cands = [r for r in results if r["enough"] and r["min_gap"] is not None]
    if not cands:
        cands = [r for r in results if r["min_gap"] is not None]
    return max(cands, key=lambda r: r["min_gap"])


def analyze(rows, title):
    print(f"\n{'='*78}\n■ {title}（{len(rows)}レース）\n{'='*78}")
    print("\n--- (1) entropy_norm ビン別の的中率 ---")
    bin_table(rows)
    print("\n--- (2) しきい値候補ごとのタイプ別レース数・的中率 ---")
    res_mand = threshold_table(rows, LO_MANDATED, HI_MANDATED, "指定グリッド 0.75/0.80/0.85 × 0.90/0.93/0.95")
    print()
    res_ext = threshold_table(rows, LO_EXTENDED, HI_EXTENDED, "拡張グリッド（低め: バランス3分割を探索）")
    best = recommend(res_mand + res_ext)
    print(f"\n--- 推奨しきい値: ({best['lo']:.2f}, {best['hi']:.2f}) ---")
    b = best["cells"]
    for lbl in (JIKU, STANDARD, CHAOS):
        n, a, bb, c = b[lbl]
        print(f"   {lbl}: {n}レース  1着的中 {_fmt(a)}  上位3内 {_fmt(bb)}  三連単Top10 {_fmt(c)}")
    print(f"   軸堅−混戦 1着的中 分離: {_fmt(best['sep'])}"
          f"（最小タイプ割合 {best['min_frac']*100:.1f}%）")
    # 現行 (0.80, 0.93) との比較
    cur = classify_rows(rows, *DEFAULT_ENTROPY_EDGES)
    print(f"\n   [現行 {DEFAULT_ENTROPY_EDGES} との比較]")
    for lbl in (JIKU, STANDARD, CHAOS):
        rs = cur[lbl]
        print(f"   {lbl}: {len(rs)}レース  1着的中 {_fmt(_rate(rs, 1))}"
              f"  上位3内 {_fmt(_rate(rs, 2))}  三連単Top10 {_fmt(_rate(rs, 3))}")
    js, cs = _rate(cur[JIKU], 1), _rate(cur[CHAOS], 1)
    cur_sep = (js - cs) if (js is not None and cs is not None) else None
    print(f"   軸堅−混戦 1着的中 分離: {_fmt(cur_sep)}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="レースタイプ分類しきい値チューニング")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    samples = load_samples(args.db, features=PL_FEATURES_FULL)
    model = load_model(args.model)
    print(f"サンプル {len(samples)}レース / モデル特徴 {len(model.feature_names)}: "
          f"{model.feature_names}")

    # モデルが rel_elo を使うなら X を拡張（retrain_3yr と同じ経路）
    if "rel_elo" in model.feature_names:
        base = model.feature_names[:model.feature_names.index("rel_elo")]
        if list(base) != list(PL_FEATURES_FULL):
            print(f"警告: モデル特徴の並びが PL_FEATURES_FULL と不一致: {base}")
        pre_elo = compute_pre_race_elo(args.db)
        samples = augment_with_elo(samples, pre_elo)
        print("rel_elo 列を X に付与（compute_pre_race_elo）")

    rows_full = build_rows(samples, model)
    _, test = time_split(samples, args.test_frac)
    rows_test = build_rows(test, model)

    print(f"\n注意: pl_model.pkl は全3年で学習済み。全期間評価は in-sample のため、"
          f"検証期間(後ろ{args.test_frac:.0%})の結果も併記して整合を確認する。")

    best_full = analyze(rows_full, "全期間（in-sample）")
    best_test = analyze(rows_test, f"検証期間（後ろ{args.test_frac:.0%}, time_split）")

    print(f"\n{'='*78}\n■ 総括\n{'='*78}")
    print(f"  全期間 推奨: ({best_full['lo']:.2f}, {best_full['hi']:.2f})  "
          f"分離 {_fmt(best_full['sep'])}")
    print(f"  検証期間 推奨: ({best_test['lo']:.2f}, {best_test['hi']:.2f})  "
          f"分離 {_fmt(best_test['sep'])}")


if __name__ == "__main__":
    main()
