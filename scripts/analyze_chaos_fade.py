"""混戦(CHAOS)レースの「本命(◎)嫌い」戦略を層別し、購入点数≤30でエッジ(ROI>100%)が
出る部分集合を探す探索スクリプト（S5・読み取り専用・DB/本番不変）。

背景（既知の実測）:
  混戦で「1着≠◎（本命が1着でない）」を全買い(≈179点/R)すると ROI均等97.3%（的中135/214R）。
  対照「1着=◎」は86.3%。＝混戦では◎が過剰人気。全179点必要なのが難点で、点数を≤30に
  絞りつつROIを上げる部分集合を探すのが目的。

印の定義: 各レースの車の1着確率(model)を降順ソートし
  ◎=1位, ○=2位, ▲=3位, △=4位, ×=5位（head_rank 0..4）。ベースは常に「1着≠◎」。

層別軸:
  1. ◎の強さ  : ◎の1着確率(model)を分位で層別
  2. ◎過剰人気度: (市場◎1着 / モデル◎1着) の閾値で嫌うレースを選別
  3. オッズ帯   : 1着≠◎ の中で大穴(高オッズ)を上限で足切り
  4. 紐の形     : 1着を ○ / ○▲ に固定した総流しで点数圧縮
点数キャップ(全/≤30/≤15)は、選定後に model_prob 降順で上位K点だけ残す（最も当たりやすい側を残す）。

実行（Windowsは文字化け回避）:
  set PYTHONIOENCODING=utf-8 && python scripts/analyze_chaos_fade.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.backtest.records_cache import get_records
from src.backtest.selection import group_by_race, _first_place_probs, settle_full
from src.ev.market import implied_trifecta_probs
from src.model.race_type import CHAOS

MARKS = ["◎", "○", "▲", "△", "×"]


# --------------------------------------------------------------------------
# 前処理: 混戦レースごとに ◎/印/過剰人気度 を計算
# --------------------------------------------------------------------------
def build_infos(records):
    """混戦レースだけを対象に、各レースの分析用メタ情報を作る。"""
    infos = []
    for rid, recs in group_by_race(records).items():
        if not recs or recs[0].race_type != CHAOS:
            continue
        fp = _first_place_probs(recs)
        if not fp:
            continue
        ranked = sorted(fp.items(), key=lambda kv: -kv[1])
        fav = ranked[0][0]
        head_rank = {car: i for i, (car, _) in enumerate(ranked)}
        odds = {r.combo: r.odds for r in recs}
        q = implied_trifecta_probs(odds)
        market_fav = sum(v for c, v in q.items() if c[0] == fav)
        model_fav = fp[fav]
        overpop = (market_fav / model_fav) if model_fav > 0 else float("inf")
        infos.append({
            "race_id": rid, "recs": recs, "fav": fav, "head_rank": head_rank,
            "model_fav": model_fav, "market_fav": market_fav, "overpop": overpop,
        })
    return infos


# --------------------------------------------------------------------------
# 選定ロジック
# --------------------------------------------------------------------------
def cap_combos(combos, cap, rank_key="prob"):
    """点数キャップ。cap を超える場合 rank_key 降順で上位 cap 点を残す。"""
    if cap is None or len(combos) <= cap:
        return combos
    if rank_key == "prob":
        key = lambda r: -r.model_prob
    elif rank_key == "ev":
        key = lambda r: -r.ev
    elif rank_key == "odds":
        key = lambda r: -r.odds
    elif rank_key == "odds_asc":
        key = lambda r: r.odds
    else:
        raise ValueError(rank_key)
    return sorted(combos, key=key)[:cap]


def race_passes(info, cfg):
    """レース単位のゲート（過剰人気度・◎強さ範囲）。"""
    if cfg.get("overpop_min") is not None and info["overpop"] < cfg["overpop_min"]:
        return False
    lo, hi = cfg.get("modelfav_range", (None, None))
    if lo is not None and info["model_fav"] < lo:
        return False
    if hi is not None and info["model_fav"] >= hi:
        return False
    return True


def race_select(info, cfg):
    """1レース分の買い目集合を返す（ベース=1着≠◎）。"""
    fav = info["fav"]
    head_rank = info["head_rank"]
    combos = [r for r in info["recs"] if r.combo[0] != fav]      # 1着≠◎
    heads = cfg.get("heads")                                     # 許可する頭 head_rank 集合
    if heads is not None:
        combos = [r for r in combos if head_rank.get(r.combo[0]) in heads]
    if cfg.get("max_odds") is not None:
        combos = [r for r in combos if r.odds <= cfg["max_odds"]]
    if cfg.get("min_odds") is not None:
        combos = [r for r in combos if r.odds >= cfg["min_odds"]]
    return cap_combos(combos, cfg.get("cap"), cfg.get("rank_key", "prob"))


def run_cfg(infos, cfg):
    """設定 cfg を全混戦レースへ適用し settle_full で決済（均等＋ドッチ）。"""
    chosen = {}
    for info in infos:
        if not race_passes(info, cfg):
            continue
        sel = race_select(info, cfg)
        if sel:
            chosen[info["race_id"]] = sel
    return settle_full(chosen)


# --------------------------------------------------------------------------
# 表示
# --------------------------------------------------------------------------
def _fmt(v):
    return f"{v*100:5.1f}%" if v is not None else "   -  "


def header(title):
    print(f"\n{title}")
    print(f"{'条件':<34}{'R数':>5}{'点/R':>7}{'的中':>5}{'的中率':>8}"
          f"{'ROI均等':>9}{'ROIドッチ':>10}")


def row(label, res):
    if res is None:
        print(f"{label:<34}{'(該当レースなし)':>10}")
        return
    note = "  ※少的中" if res["n_hits"] < 10 else ""
    print(f"{label:<34}{res['n_races']:>5}{res['pts']:>7.1f}{res['n_hits']:>5}"
          f"{_fmt(res['hit_rate']):>8}{_fmt(res['roi_eq']):>9}"
          f"{_fmt(res['roi_du']):>10}{note}")


def caps_row(infos, label, base_cfg, caps=(None, 30, 15)):
    """同一基底 cfg で点数キャップだけ変えた行を並べる。"""
    for cap in caps:
        cfg = dict(base_cfg, cap=cap)
        tag = "全" if cap is None else f"≤{cap}"
        row(f"{label} [{tag}]", run_cfg(infos, cfg))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    records = get_records()
    infos = build_infos(records)
    n = len(infos)
    print(f"混戦(CHAOS)レース数: {n}")
    mf = sorted(i["model_fav"] for i in infos)
    op = sorted(i["overpop"] for i in infos if i["overpop"] != float("inf"))
    if mf:
        print(f"  モデル◎1着確率  中央値={mf[len(mf)//2]:.3f}  範囲[{mf[0]:.3f},{mf[-1]:.3f}]")
    if op:
        print(f"  ◎過剰人気度(市場/モデル) 中央値={op[len(op)//2]:.2f} "
              f"範囲[{op[0]:.2f},{op[-1]:.2f}]  >1.0のR数={sum(1 for x in op if x>1.0)}/{len(op)}")
    print("\n凡例: 点/R=1レース平均点数, 的中率=的中R/買ったR, "
          "ROI均等=1点100円, ROIドッチ=1R予算1000円を1/oddsで按分。控除率≈25%で<100%が基本。")

    # ---- 0. ベースライン再現（1着≠◎ 全買い vs 1着=◎） ----
    header("【0】ベースライン再現（混戦・点数キャップなし）")
    row("1着≠◎ 全買い（◎嫌い）", run_cfg(infos, {}))
    # 対照: 1着=◎ を全買い
    chosen_fav = {}
    for info in infos:
        s = [r for r in info["recs"] if r.combo[0] == info["fav"]]
        if s:
            chosen_fav[info["race_id"]] = s
    row("1着=◎ 全買い（対照）", settle_full(chosen_fav))

    # ---- 1. ◎の強さ層別（model_fav 3分位） ----
    header("【1】◎の強さ層別（modelの◎1着確率で3分位・各層 1着≠◎ 全買い）")
    q1, q2 = mf[n // 3], mf[2 * n // 3]
    for lo, hi, name in ((None, q1, "弱◎"), (q1, q2, "中◎"), (q2, None, "強◎")):
        row(f"◎強さ={name}[{lo if lo else 0:.3f},{hi if hi else 1:.3f})",
            run_cfg(infos, {"modelfav_range": (lo, hi)}))
    print("  （◎が弱いほど過剰人気で嫌う価値が大きいか＝弱◎層のROIが高いかを見る）")

    # ---- 2. ◎過剰人気度の閾値掃引 ----
    header("【2】◎過剰人気度ゲート（市場◎1着/モデル◎1着 ≥ 閾値 のレースだけ・1着≠◎ 全買い）")
    for thr in (1.0, 1.1, 1.3, 1.5, 1.7):
        row(f"過剰人気度 ≥ {thr}", run_cfg(infos, {"overpop_min": thr}))

    # ---- 3. オッズ帯足切り（大穴除外で点数圧縮） ----
    header("【3】オッズ上限足切り（1着≠◎ で odds≤上限のみ・大穴の当たらない紐を捨てる）")
    for mo in (None, 200, 100, 50, 30):
        tag = "上限なし" if mo is None else f"odds≤{mo}"
        row(f"{tag}", run_cfg(infos, {"max_odds": mo}))

    # ---- 4. 紐の形（頭固定総流しで点数圧縮） ----
    header("【4】紐の形（頭を印で固定した総流し・1着≠◎）")
    row("頭=○ 総流し", run_cfg(infos, {"heads": {1}}))
    row("頭=○▲ 総流し", run_cfg(infos, {"heads": {1, 2}}))
    row("頭=○▲△ 総流し", run_cfg(infos, {"heads": {1, 2, 3}}))
    row("頭=○▲△× 総流し", run_cfg(infos, {"heads": {1, 2, 3, 4}}))

    # ---- 5. 組み合わせ × 点数キャップ（全/≤30/≤15） ----
    header("【5】組み合わせ × 点数キャップ（キャップは model_prob 上位を残す）")
    combos = [
        ("素の1着≠◎",                {}),
        ("odds≤100",                 {"max_odds": 100}),
        ("odds≤50",                  {"max_odds": 50}),
        ("過剰1.3",                   {"overpop_min": 1.3}),
        ("過剰1.3+odds≤100",         {"overpop_min": 1.3, "max_odds": 100}),
        ("過剰1.5+odds≤100",         {"overpop_min": 1.5, "max_odds": 100}),
        ("過剰1.3+odds≤50",          {"overpop_min": 1.3, "max_odds": 50}),
        ("頭○▲+odds≤100",           {"heads": {1, 2}, "max_odds": 100}),
        ("過剰1.3+頭○▲+odds≤100",   {"overpop_min": 1.3, "heads": {1, 2}, "max_odds": 100}),
        ("過剰1.3+頭○▲+odds≤50",    {"overpop_min": 1.3, "heads": {1, 2}, "max_odds": 50}),
    ]
    for label, cfg in combos:
        caps_row(infos, label, cfg, caps=(None, 30, 15))
        print()

    # ---- 6. キャップ内の残し方比較（≤30点・prob/ev/odds） ----
    header("【6】≤30点キャップ内の残し方比較（基底=過剰1.3+odds≤100）")
    base = {"overpop_min": 1.3, "max_odds": 100, "cap": 30}
    for rk, name in (("prob", "確率上位を残す"), ("ev", "EV上位を残す"),
                     ("odds", "高オッズを残す"), ("odds_asc", "低オッズを残す")):
        row(f"残し方={name}", run_cfg(infos, dict(base, rank_key=rk)))

    # ---- 7. ROI vs 点数フロンティア ----
    header("【7】ROI vs 点数フロンティア（基底=過剰1.3+odds≤100・確率上位を残す）")
    base = {"overpop_min": 1.3, "max_odds": 100}
    for cap in (None, 60, 40, 30, 25, 20, 15, 10, 6):
        tag = "全" if cap is None else f"≤{cap}"
        row(f"cap={tag}", run_cfg(infos, dict(base, cap=cap)))

    # ---- 8. 最良候補の探索（≤30点でROI>100%かつ的中≥20を満たすか総当り） ----
    header("【8】≤30点 最良候補の総当り探索（ROI均等 降順・的中≥15のみ）")
    grid = []
    for op_thr in (None, 1.1, 1.3, 1.5, 1.7):
        for mo in (None, 150, 100, 70, 50, 30):
            for heads in (None, frozenset({1, 2, 3}), frozenset({1, 2}), frozenset({1})):
                for cap in (30, 20, 15):
                    cfg = {"cap": cap, "rank_key": "prob"}
                    if op_thr is not None:
                        cfg["overpop_min"] = op_thr
                    if mo is not None:
                        cfg["max_odds"] = mo
                    if heads is not None:
                        cfg["heads"] = set(heads)
                    res = run_cfg(infos, cfg)
                    if res is None or res["n_races"] < 20:
                        continue
                    grid.append((res["roi_eq"], op_thr, mo, heads, cap, res))
    grid.sort(key=lambda x: -x[0])
    print(f"{'ROI均等':>8}{'ドッチ':>8}{'R数':>5}{'点/R':>6}{'的中':>5}{'的中率':>8}"
          f"  過剰 / odds上限 / 頭 / cap")
    shown = 0
    for roi, op_thr, mo, heads, cap, res in grid:
        if res["n_hits"] < 15:
            continue
        hd = ("全" if heads is None else "".join(MARKS[i] for i in sorted(heads)))
        print(f"{_fmt(res['roi_eq']):>8}{_fmt(res['roi_du']):>8}{res['n_races']:>5}"
              f"{res['pts']:>6.1f}{res['n_hits']:>5}{_fmt(res['hit_rate']):>8}"
              f"   {str(op_thr):>4} / {str(mo):>5} / {hd:<5} / ≤{cap}")
        shown += 1
        if shown >= 15:
            break

    # ---- 判定 ----
    winners = [g for g in grid if g[0] > 1.0 and g[5]["n_hits"] >= 20]
    print("\n=== 判定 ===")
    if winners:
        roi, op_thr, mo, heads, cap, res = winners[0]
        hd = ("全" if heads is None else "".join(MARKS[i] for i in sorted(heads)))
        print(f"≤{cap}点で ROI>100% かつ 的中≥20 の条件あり: "
              f"過剰≥{op_thr} / odds≤{mo} / 頭={hd} / cap≤{cap}")
        print(f"  → ROI均等 {res['roi_eq']*100:.1f}% / ドッチ {res['roi_du']*100:.1f}% / "
              f"{res['n_races']}R / {res['pts']:.1f}点/R / 的中{res['n_hits']}({res['hit_rate']*100:.1f}%)")
    else:
        best = None
        for g in grid:
            if g[5]["n_hits"] >= 20 and (best is None or g[0] > best[0]):
                best = g
        if best:
            roi, op_thr, mo, heads, cap, res = best
            hd = ("全" if heads is None else "".join(MARKS[i] for i in sorted(heads)))
            print("≤30点でROI>100%(的中≥20)の条件は無し。最も100%に近い≤30点条件:")
            print(f"  過剰≥{op_thr} / odds≤{mo} / 頭={hd} / cap≤{cap}")
            print(f"  → ROI均等 {res['roi_eq']*100:.1f}% / ドッチ {res['roi_du']*100:.1f}% / "
                  f"{res['n_races']}R / {res['pts']:.1f}点/R / 的中{res['n_hits']}({res['hit_rate']*100:.1f}%)")
        else:
            print("的中≥20を満たす≤30点条件が無い（点数を絞ると的中が不足）。")


if __name__ == "__main__":
    main()
