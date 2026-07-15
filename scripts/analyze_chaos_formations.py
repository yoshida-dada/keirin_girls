"""混戦(CHAOS)レース向け ≤30点フォーメーションの実現ROI/点数バランス比較（読み取り専用）。

背景: 混戦では本命◎が過剰人気になり、「1着≠◎」の総流し(≈179点)でも ROI 均等 ≈97.3%。
      本スクリプトは「◎を1着から外す/2・3着へ置く」思想の **現実的な ≤30点フォーメーション**
      を複数設計し、点/R と 実現ROI(均等・ドッチング) のトレードオフを表化する。

印の付け方: 各レースの車を model の 1着確率(_first_place_probs)降順に並べ、上位から
      ◎ ○ ▲ △ × を割り当てる(6位以降は印なし=買い目に使わない)。

データ: from src.backtest.records_cache import get_records → キャッシュ(_bt_records.pkl)を即読込。
        DB・本番には一切触れない。混戦のみ対象(r.race_type=="混戦")。

  # Windows 文字化け回避
  set PYTHONIOENCODING=utf-8 && python scripts/analyze_chaos_formations.py

★控除率≈25%のため ROI は 1.0(100%) 未満が基本。的中<10 は分散ノイズ(注記)。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # Windows コンソール文字化け対策
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.backtest.records_cache import get_records
from src.backtest.selection import group_by_race, _first_place_probs, settle_full
from src.model.race_type import CHAOS

MARK_LABELS = ["◎", "○", "▲", "△", "×"]  # index 0..4 = 1着確率 上位から
MIN_HITS = 10  # これ未満は分散ノイズとして注記


# --------------------------------------------------------------------------
# レース単位の準備: 印(marks) と combo→record マップ
# --------------------------------------------------------------------------
def race_context(recs):
    """1レース分 recs から (marks, cmap) を返す。marks=[◎○▲△×の車番], cmap={combo:record}。

    marks は 1着確率降順の上位5車。5車に満たないレースは None(除外)。
    """
    fp = _first_place_probs(recs)
    if len(fp) < 5:
        return None
    marks = [c for c, _ in sorted(fp.items(), key=lambda kv: -kv[1])][:5]
    cmap = {r.combo: r for r in recs}
    return marks, cmap


def _distinct_perms(a_set, b_set, c_set):
    """1着∈a_set, 2着∈b_set, 3着∈c_set で車番が相異なる順列(a,b,c)を列挙。"""
    out = []
    for a in a_set:
        for b in b_set:
            if b == a:
                continue
            for c in c_set:
                if c == a or c == b:
                    continue
                out.append((a, b, c))
    return out


# --------------------------------------------------------------------------
# フォーメーション定義: marks -> combo タプルのリスト
# 各関数は「◎を1着から外す/紐に置く」思想。点数は marks から決定的に定まる。
# --------------------------------------------------------------------------
def f1_maru_sankaku_head(marks, n2=4, n3=5):
    """F1 ○▲頭: 1着∈{○,▲} → 2着=上位n2印 → 3着=上位n3印。"""
    heads = [marks[1], marks[2]]
    return _distinct_perms(heads, marks[:n2], marks[:n3])


def f2_maru_head_fixed(marks, n2=3, n3=4):
    """F2 ○頭固定: 1着=○ → 2着=(○除く上位)n2 → 3着=(○除く上位)n3。"""
    head = marks[1]
    others = [m for m in marks if m != head]
    return _distinct_perms([head], others[:n2], others[:n3])


def f3_honmei_second(marks, n1=2, n3=4):
    """F3 ◎を2着固定: 1着∈上位n1(◎除く) → 2着=◎ → 3着=(◎除く上位)n3。"""
    honmei = marks[0]
    heads = [m for m in marks if m != honmei][:n1]  # ○,▲,...
    third = [m for m in marks if m != honmei][:n3]
    return _distinct_perms(heads, [honmei], third)


def f4_honmei_third(marks, n1=2, n2=3):
    """F4 ◎を3着固定: 1着∈上位n1(◎除く) → 2着=(◎除く上位)n2 → 3着=◎。"""
    honmei = marks[0]
    heads = [m for m in marks if m != honmei][:n1]
    second = [m for m in marks if m != honmei][:n2]
    return _distinct_perms(heads, second, [honmei])


def f5_honmei_himo_nagashi(marks, n_head=3, n_partner=5):
    """F5 ◎を必ず紐に含む相手流し: 1着∈上位n_head(◎除く),
    2着/3着 のどちらかは必ず ◎、相方は上位n_partner印。"""
    honmei = marks[0]
    heads = [m for m in marks if m != honmei][:n_head]
    partners = marks[:n_partner]
    out = []
    for a in heads:
        for p in partners:
            if p == a or p == honmei:
                continue
            out.append((a, honmei, p))  # ◎が2着
            out.append((a, p, honmei))  # ◎が3着
    return out


# --------------------------------------------------------------------------
# EV/オッズで軽く絞る版(F6): ベース買い目に条件フィルタを掛けて点数を圧縮
# --------------------------------------------------------------------------
def apply_filter(combos, cmap, ev_thr=None, odds_min=None, odds_max=None):
    """combo リストを cmap の record を見て EV/オッズ条件で絞る。"""
    out = []
    for cb in combos:
        r = cmap.get(cb)
        if r is None:
            continue
        if ev_thr is not None and r.ev < ev_thr:
            continue
        if odds_min is not None and r.odds < odds_min:
            continue
        if odds_max is not None and r.odds > odds_max:
            continue
        out.append(cb)
    return out


# --------------------------------------------------------------------------
# 評価: フォーメーション関数 → settle_full(均等/ドッチング両方)
# --------------------------------------------------------------------------
def evaluate(chaos_races, gen, **kwargs):
    """gen(marks,**kwargs)->combos を全混戦レースに適用し settle_full の結果を返す。

    filt(cmap でフィルタ)を渡す版は evaluate_filtered を使う。
    """
    chosen = {}
    for rid, recs in chaos_races.items():
        ctx = race_context(recs)
        if ctx is None:
            continue
        marks, cmap = ctx
        combos = gen(marks, **kwargs)
        picked = [cmap[cb] for cb in combos if cb in cmap]
        if picked:
            chosen[rid] = picked
    return settle_full(chosen)


def evaluate_filtered(chaos_races, gen, gen_kwargs, filt_kwargs):
    """gen で候補 combo を作り apply_filter で絞ってから settle_full。"""
    chosen = {}
    for rid, recs in chaos_races.items():
        ctx = race_context(recs)
        if ctx is None:
            continue
        marks, cmap = ctx
        combos = gen(marks, **gen_kwargs)
        combos = apply_filter(combos, cmap, **filt_kwargs)
        picked = [cmap[cb] for cb in combos if cb in cmap]
        if picked:
            chosen[rid] = picked
    return settle_full(chosen)


# --------------------------------------------------------------------------
# 点数-ROI トレードオフ: 「◎≠1着」宇宙から order 降順で top-k/R を取る
# --------------------------------------------------------------------------
def evaluate_topk(chaos_races, k, order="ev"):
    """各混戦レースで「1着≠◎」の買い目を order(ev|prob) 降順に並べ top-k を買う。"""
    chosen = {}
    for rid, recs in chaos_races.items():
        ctx = race_context(recs)
        if ctx is None:
            continue
        marks, cmap = ctx
        honmei = marks[0]
        cand = [r for r in recs if r.combo[0] != honmei]
        key = (lambda r: r.ev) if order == "ev" else (lambda r: r.model_prob)
        cand.sort(key=key, reverse=True)
        picked = cand[:k]
        if picked:
            chosen[rid] = picked
    return settle_full(chosen)


# --------------------------------------------------------------------------
# 表示ヘルパ
# --------------------------------------------------------------------------
def _fmt(res):
    if res is None:
        return None
    note = "  ※的中少" if res["n_hits"] < MIN_HITS else ""
    return (f"{res['pts']:>6.1f}{res['n_races']:>6}{res['n_hits']:>6}"
            f"{res['hit_rate']*100:>8.1f}%{res['roi_eq']*100:>9.1f}%"
            f"{res['roi_du']*100:>9.1f}%{note}")


def _print_header(title):
    print(f"\n{title}")
    print(f"{'フォーメーション':<30}{'点/R':>6}{'R数':>6}{'的中':>6}"
          f"{'的中率':>8}{'ROI均等':>9}{'ROIドッチ':>10}")


def _row(label, res):
    body = _fmt(res)
    if body is None:
        print(f"{label:<30}  (該当レース無し)")
    else:
        print(f"{label:<30}{body}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    records = get_records()
    chaos = [r for r in records if r.race_type == CHAOS]
    chaos_races = group_by_race(chaos)
    n_races = len(chaos_races)
    print("=" * 78)
    print("混戦(CHAOS)レース向け ≤30点フォーメーション 実現ROI比較")
    print("=" * 78)
    print(f"混戦レース数: {n_races} / 買い目レコード: {len(chaos)}")
    print("印: 1着確率降順に ◎○▲△×(6位以降−) / 決済: 均等(1点100円) と ドッチング(1R予算∝1/odds)")
    print("凡例: 点/R=1レース点数, ROI均等/ドッチ=回収率. 控除率≈25%のため<100%が基本.")

    # ---- 参考: 本命嫌い 総流し(1着≠◎ 全通り) のベースライン ----
    _print_header("【0】参考ベースライン: 1着≠◎ の総流し(全点)")
    _row("1着≠◎ 総流し(≈179点)", evaluate_topk(chaos_races, k=10**9, order="ev"))

    # ---- F1 ○▲頭 ----
    _print_header("【F1】○▲頭: 1着∈{○,▲}, 2着=上位n2印, 3着=上位n3印")
    for n2, n3 in [(4, 4), (4, 5), (5, 5)]:
        _row(f"F1 n2={n2},n3={n3}", evaluate(chaos_races, f1_maru_sankaku_head, n2=n2, n3=n3))

    # ---- F2 ○頭固定 ----
    _print_header("【F2】○頭固定: 1着=○, 2着=(○除く)n2, 3着=(○除く)n3")
    for n2, n3 in [(3, 4), (4, 4), (4, 5)]:
        _row(f"F2 n2={n2},n3={n3}", evaluate(chaos_races, f2_maru_head_fixed, n2=n2, n3=n3))

    # ---- F3 ◎を2着固定 ----
    _print_header("【F3】◎を2着固定: 1着∈上位n1(◎除く), 2着=◎, 3着=(◎除く)n3")
    for n1, n3 in [(2, 4), (3, 4), (3, 5)]:
        _row(f"F3 n1={n1},n3={n3}", evaluate(chaos_races, f3_honmei_second, n1=n1, n3=n3))

    # ---- F4 ◎を3着固定 ----
    _print_header("【F4】◎を3着固定: 1着∈上位n1(◎除く), 2着=(◎除く)n2, 3着=◎")
    for n1, n2 in [(2, 3), (3, 3), (3, 4)]:
        _row(f"F4 n1={n1},n2={n2}", evaluate(chaos_races, f4_honmei_third, n1=n1, n2=n2))

    # ---- F5 ◎を必ず紐に含む相手流し ----
    _print_header("【F5】◎必ず紐 相手流し: 1着∈上位n_head(◎除く), 2/3着の一方=◎")
    for nh, npr in [(2, 4), (3, 4), (3, 5)]:
        _row(f"F5 head={nh},partner={npr}",
             evaluate(chaos_races, f5_honmei_himo_nagashi, n_head=nh, n_partner=npr))

    # ---- F6 EV/オッズ条件で軽く絞る版(点数圧縮) ----
    _print_header("【F6】絞り込み版: F1(5,5)ベースに EV/オッズ条件を付与し点数圧縮")
    _row("F6 F1(5,5) base(無絞り)",
         evaluate(chaos_races, f1_maru_sankaku_head, n2=5, n3=5))
    _row("F6 +EV≥1.0", evaluate_filtered(chaos_races, f1_maru_sankaku_head,
                                          {"n2": 5, "n3": 5}, {"ev_thr": 1.0}))
    _row("F6 +EV≥1.2", evaluate_filtered(chaos_races, f1_maru_sankaku_head,
                                          {"n2": 5, "n3": 5}, {"ev_thr": 1.2}))
    _row("F6 +odds≥20", evaluate_filtered(chaos_races, f1_maru_sankaku_head,
                                          {"n2": 5, "n3": 5}, {"odds_min": 20.0}))
    _row("F6 F5(3,5)+EV≥1.0", evaluate_filtered(chaos_races, f5_honmei_himo_nagashi,
                                                 {"n_head": 3, "n_partner": 5}, {"ev_thr": 1.0}))

    # ---- 点数-ROI トレードオフ(1着≠◎ 宇宙 top-k) ----
    print("\n" + "=" * 78)
    print("点数-ROIトレードオフ: 「1着≠◎」の買い目を order 降順に top-k/R 購入")
    print("=" * 78)
    for order in ("ev", "prob"):
        _print_header(f"order={order}(降順) 上位k点を購入")
        for k in (30, 20, 15, 10):
            _row(f"top-{k}/R", evaluate_topk(chaos_races, k=k, order=order))

    # ---- 結論の材料: ≤30点で ROI 最良のフォーメーションを自動抽出 ----
    print("\n" + "=" * 78)
    print("結論: ≤30点フォーメーションの ROI均等 ベスト(的中≥10 を優先)")
    print("=" * 78)
    candidates = []

    def _collect(label, res):
        if res is not None and res["pts"] <= 30:
            candidates.append((label, res))

    for n2, n3 in [(4, 4), (4, 5), (5, 5)]:
        _collect(f"F1 n2={n2},n3={n3}", evaluate(chaos_races, f1_maru_sankaku_head, n2=n2, n3=n3))
    for n2, n3 in [(3, 4), (4, 4), (4, 5)]:
        _collect(f"F2 n2={n2},n3={n3}", evaluate(chaos_races, f2_maru_head_fixed, n2=n2, n3=n3))
    for n1, n3 in [(2, 4), (3, 4), (3, 5)]:
        _collect(f"F3 n1={n1},n3={n3}", evaluate(chaos_races, f3_honmei_second, n1=n1, n3=n3))
    for n1, n2 in [(2, 3), (3, 3), (3, 4)]:
        _collect(f"F4 n1={n1},n2={n2}", evaluate(chaos_races, f4_honmei_third, n1=n1, n2=n2))
    for nh, npr in [(2, 4), (3, 4), (3, 5)]:
        _collect(f"F5 head={nh},partner={npr}",
                 evaluate(chaos_races, f5_honmei_himo_nagashi, n_head=nh, n_partner=npr))
    for k in (30, 20, 15, 10):
        _collect(f"top-{k}/R(ev)", evaluate_topk(chaos_races, k=k, order="ev"))
        _collect(f"top-{k}/R(prob)", evaluate_topk(chaos_races, k=k, order="prob"))

    # 的中≥10 を優先しつつ ROI均等 で降順
    reliable = [c for c in candidates if c[1]["n_hits"] >= MIN_HITS]
    pool = reliable if reliable else candidates
    pool.sort(key=lambda c: -c[1]["roi_eq"])
    _print_header("≤30点 ROI均等 上位5(的中≥10 を優先)")
    for label, res in pool[:5]:
        _row(label, res)

    best = pool[0]
    print(f"\n→ ≤30点の最良: 「{best[0]}」 "
          f"点/R={best[1]['pts']:.1f} ROI均等={best[1]['roi_eq']*100:.1f}% "
          f"ROIドッチ={best[1]['roi_du']*100:.1f}% 的中={best[1]['n_hits']}/{best[1]['n_races']}")
    verdict = "黒字(>100%)達成" if best[1]["roi_eq"] >= 1.0 else "100%未満(控除率の壁)"
    print(f"  判定: {verdict}")


if __name__ == "__main__":
    main()
