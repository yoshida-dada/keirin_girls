"""標準(STANDARD)・軸堅(JIKU)に「本命◎の過剰人気を突くエッジ」が ≤30点で無いか調査する。

背景（既知の実測）:
  ・混戦(CHAOS)では本命◎が市場で過剰人気→「◎嫌い(1着≠◎)」が有効(ROI≈97.3%)。
  ・軸堅の◎頭10点フォーメーションはエッジ無し(≈82%)。
  ・標準は未検証。また軸堅でも「◎が過剰人気なレースだけ」を突けばエッジが出る可能性。
本スクリプトは records_cache の out-of-sample records（再学習不要・DB非依存）だけを消費し、
以下4パターンを ≤30点フォーメーションで検証、均等/ドッチング両決済のROIを表化する。

  1. 標準・本命嫌い : 1着≠◎ の ≤30点フォーメーション（○▲頭 / ◎を2・3着固定 等）
  2. 標準・本命軸   : 1着=◎ 頭固定フォーメーション（◎→○▲→○▲△× 等）
  3. ◎過剰人気レース抽出（全タイプ横断）: 過剰人気度=(市場◎1着q)/(モデル◎1着p) が閾値超の
       レースに限定して ◎嫌い を買う。閾値 1.2/1.4/1.6 を掃引しタイプ別ROIを出す。
  4. ◎過小評価レース（逆張り・対照）: 過剰人気度<1 のとき ◎頭が有利か。

印の定義: 1レースの各車1着確率 P(1着=car)=Σ_{combo[0]=car} model_prob を降順に並べ
          ◎(0) ○(1) ▲(2) △(3) ×(4) …。過剰人気度は odds→implied_trifecta_probs の
          市場フェア確率 q を使い、Σ_{combo[0]=◎} q / モデル◎1着確率 で定義する。

実行(Windows は文字化け回避):
  set PYTHONIOENCODING=utf-8 && python scripts/analyze_std_jiku_overbet.py

★読み取り専用。DB/本番は一切変更しない。records はキャッシュ(_bt_records.pkl)を読むだけ。
★控除率≈25%のためROIは1.0未満が基本。少的中(<10)は分散ノイズとして注記する。
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # Windows コンソール文字化け対策
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.backtest.records_cache import get_records
from src.backtest.selection import group_by_race, _first_place_probs, settle_full
from src.ev.market import implied_trifecta_probs
from src.model.race_type import JIKU, STANDARD, CHAOS

RACE_TYPES = [STANDARD, JIKU, CHAOS]
CAP = 30            # 点数キャップ（1レースあたり）
MIN_HITS = 10       # これ未満の的中は分散ノイズとして注記

# 印ラベル（index→記号）。表示用
_MARK = {0: "◎", 1: "○", 2: "▲", 3: "△", 4: "×", 5: "△5", 6: "×6"}


# --------------------------------------------------------------------------
# レース単位ユーティリティ
# --------------------------------------------------------------------------
def mark_index(recs) -> dict[int, int]:
    """各車 → 印インデックス（0=◎ が最有力）。1着確率 P(1着=car) の降順。"""
    fp = _first_place_probs(recs)
    order = [c for c, _ in sorted(fp.items(), key=lambda kv: -kv[1])]
    return {car: i for i, car in enumerate(order)}


def overbet_ratio(recs) -> float | None:
    """◎過剰人気度 = (Σ_{combo[0]=◎} 市場q) / (モデル◎1着確率)。

    >1 なら市場が◎1着をモデル以上に高く評価＝過剰人気（◎嫌いが効くはず）。
    <1 なら◎が過小人気。q が定義できなければ None。
    """
    fp = _first_place_probs(recs)
    if not fp:
        return None
    honmei = max(fp, key=fp.get)
    model_p1 = fp[honmei]
    if model_p1 <= 0:
        return None
    q = implied_trifecta_probs({r.combo: r.odds for r in recs})
    if not q:
        return None
    mkt_p1 = sum(v for k, v in q.items() if k[0] == honmei)
    if mkt_p1 <= 0:
        return None
    return mkt_p1 / model_p1


# --------------------------------------------------------------------------
# 印ベースのフォーメーション生成（純粋関数・1レース分 → 買う ComboRecord）
# --------------------------------------------------------------------------
def make_formation(head_idx: set[int], second_idx: set[int], third_idx: set[int],
                   honmei_himo: bool = False):
    """印インデックス集合で頭/2着/3着を指定するフォーメーション生成関数を返す。

    honmei_himo=True: さらに ◎(index0) が2着or3着に入る買い目だけに絞る（◎を紐で拾う）。
    生成関数は 1レース分 recs → 買う ComboRecord のリスト（自然に少点数になる）。
    """
    def build(recs):
        mi = mark_index(recs)
        out = []
        for r in recs:
            a, b, c = r.combo
            ia, ib, ic = mi.get(a, 99), mi.get(b, 99), mi.get(c, 99)
            if ia in head_idx and ib in second_idx and ic in third_idx:
                if honmei_himo and (0 not in (ib, ic)):
                    continue
                out.append(r)
        return out
    return build


def _lbl(idx_set: set[int]) -> str:
    return "".join(_MARK.get(i, f"?{i}") for i in sorted(idx_set))


# --------------------------------------------------------------------------
# 評価・表示
# --------------------------------------------------------------------------
def eval_formation(records, build_fn, want_type=None, ratio_pred=None,
                   ratio_by_race=None):
    """フォーメーションを条件付きレース集合に適用し settle_full（均等+ドッチ）を返す。

    want_type   : この race_type のレースだけ（None=全タイプ）
    ratio_pred  : 過剰人気度 ratio に対する述語（True のレースだけ）。ratio_by_race 必須。
    各レースの買い目は EV 降順で CAP 点に丸める（≤30点保証）。
    """
    chosen = {}
    for rid, recs in group_by_race(records).items():
        if want_type is not None and recs and recs[0].race_type != want_type:
            continue
        if ratio_pred is not None:
            rr = ratio_by_race.get(rid)
            if rr is None or not ratio_pred(rr):
                continue
        bought = build_fn(recs)
        if len(bought) > CAP:
            bought = sorted(bought, key=lambda r: -r.ev)[:CAP]
        if bought:
            chosen[rid] = bought
    return settle_full(chosen)


def _fmt(v):
    return f"{v*100:6.1f}%" if v is not None else "   -   "


def print_row(label, res):
    if res is None:
        print(f"{label:<30}{'(該当レース無し)':>10}")
        return
    note = "  ※的中少" if res["n_hits"] < MIN_HITS else ""
    print(f"{label:<30}{res['n_races']:>6}{res['pts']:>7.1f}{res['n_bets']:>7}"
          f"{res['n_hits']:>6}{_fmt(res['hit_rate']):>9}"
          f"{_fmt(res['roi_eq']):>9}{_fmt(res['roi_du']):>9}{note}")


def print_header(title):
    print(f"\n{title}")
    print(f"{'条件':<30}{'R数':>6}{'点/R':>7}{'点数':>7}{'的中':>6}"
          f"{'的中率':>9}{'ROI均等':>9}{'ROIドッチ':>9}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    records = get_records()
    races = group_by_race(records)
    n_races = len(races)
    n_by_type = defaultdict(int)
    for recs in races.values():
        if recs:
            n_by_type[recs[0].race_type] += 1
    print("=" * 78)
    print("標準・軸堅の『◎過剰人気を突くエッジ』調査（≤30点・out-of-sample・読み取り専用）")
    print("=" * 78)
    print(f"検証レース: {n_races}  "
          + " / ".join(f"{t}:{n_by_type.get(t, 0)}" for t in RACE_TYPES))
    print("印: モデル1着確率 降順で ◎○▲△×  ／  凡例 ROI均等=1点100円, "
          "ROIドッチ=1R1000円を1/oddsで配分")
    print("控除率≈25%のため<100%が基本。的中<10は分散ノイズ。")

    # 過剰人気度をレースごとに1回だけ計算
    ratio_by_race = {rid: overbet_ratio(recs) for rid, recs in races.items()}
    valid = [v for v in ratio_by_race.values() if v is not None]
    valid.sort()
    if valid:
        n = len(valid)
        med = valid[n // 2]
        share_over = sum(1 for v in valid for _ in [0] if v > 1.0)
        print(f"\n◎過剰人気度分布(全{n}R): "
              f"中央値={med:.2f}  >1.0割合={share_over/n*100:.0f}%  "
              f"25%点={valid[n//4]:.2f}  75%点={valid[3*n//4]:.2f}")

    # ==================================================================
    # パターン1: 標準・本命嫌い（1着≠◎）
    # ==================================================================
    formations_hate = [
        ("F1 ○▲頭→◎○▲△→◎○▲△×",
         make_formation({1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4})),
        ("F2 ○▲頭→◎○▲→◎○▲△(絞)",
         make_formation({1, 2}, {0, 1, 2}, {0, 1, 2, 3})),
        ("F3 ○▲頭×◎を2-3着固定",
         make_formation({1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4}, honmei_himo=True)),
        ("F4 ○▲△頭→◎○▲△→◎○▲△×",
         make_formation({1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3, 4})),
    ]
    print_header("【1】標準・本命嫌い（1着≠◎）フォーメーション ≤30点  ※タイプ別")
    for name, fn in formations_hate:
        for t in RACE_TYPES:
            print_row(f"{name} [{t}]", eval_formation(records, fn, want_type=t))
        print()

    # ==================================================================
    # パターン2: 標準・本命軸（1着=◎ 頭固定）
    # ==================================================================
    formations_jiku = [
        ("G1 ◎→○▲→○▲△×",
         make_formation({0}, {1, 2}, {1, 2, 3, 4})),
        ("G2 ◎→○▲▲→○▲△×5",
         make_formation({0}, {1, 2, 3}, {1, 2, 3, 4, 5})),
        ("G3 ◎→○▲→総流し(○▲△×5×6)",
         make_formation({0}, {1, 2}, {1, 2, 3, 4, 5, 6})),
        ("G4 ◎→○▲△×→○▲△×",
         make_formation({0}, {1, 2, 3, 4}, {1, 2, 3, 4})),
    ]
    print_header("【2】本命軸（1着=◎ 頭固定）フォーメーション ≤30点  ※タイプ別")
    for name, fn in formations_jiku:
        for t in RACE_TYPES:
            print_row(f"{name} [{t}]", eval_formation(records, fn, want_type=t))
        print()

    # ==================================================================
    # パターン3: ◎過剰人気レース抽出（全タイプ横断）× ◎嫌い
    # ==================================================================
    hate_fn = make_formation({1, 2}, {0, 1, 2, 3}, {0, 1, 2, 3, 4})  # =F1
    print_header("【3】◎過剰人気レースだけ ◎嫌い(F1)を買う  過剰人気度閾値の掃引 × タイプ別")
    print("  （過剰人気度 = 市場◎1着q / モデル◎1着p。閾値超のレースに限定）")
    for thr in (1.0, 1.2, 1.4, 1.6):
        pred = (lambda r, th=thr: r > th)
        for t in RACE_TYPES + [None]:
            tl = t if t is not None else "全タイプ"
            print_row(f"過剰人気>{thr} [{tl}]",
                      eval_formation(records, hate_fn, want_type=t,
                                     ratio_pred=pred, ratio_by_race=ratio_by_race))
        print()

    # ==================================================================
    # パターン4: ◎過小評価レース（過剰人気度<1）× ◎頭（逆張り対照）
    # ==================================================================
    jiku_fn = make_formation({0}, {1, 2, 3}, {1, 2, 3, 4, 5})  # =G2
    print_header("【4】◎過小評価レース(過剰人気度<1)で ◎頭(G2)を買う（対照）× タイプ別")
    under = (lambda r: r < 1.0)
    for t in RACE_TYPES + [None]:
        tl = t if t is not None else "全タイプ"
        print_row(f"過剰人気<1.0 [{tl}]",
                  eval_formation(records, jiku_fn, want_type=t,
                                 ratio_pred=under, ratio_by_race=ratio_by_race))
    print("\n  参考: 同 ◎頭(G2) を過剰人気>1.2 のレースで買った場合（過小評価との対比）")
    over12 = (lambda r: r > 1.2)
    for t in RACE_TYPES + [None]:
        tl = t if t is not None else "全タイプ"
        print_row(f"過剰人気>1.2 [{tl}]",
                  eval_formation(records, jiku_fn, want_type=t,
                                 ratio_pred=over12, ratio_by_race=ratio_by_race))

    # ==================================================================
    # 結論の材料
    # ==================================================================
    print("\n" + "=" * 78)
    print("結論の材料（標準/軸堅に ≤30点 の黒字>100%候補があるか）")
    print("=" * 78)
    best = None  # (roi, label)
    scan = []
    # 標準の全フォーメーション（無条件）
    for name, fn in formations_hate + formations_jiku:
        for t in (STANDARD, JIKU):
            res = eval_formation(records, fn, want_type=t)
            if res and res["n_hits"] >= MIN_HITS:
                scan.append((res["roi_eq"], res["roi_du"], f"{name} [{t}]", res))
    # 過剰人気抽出（◎嫌い）
    for thr in (1.2, 1.4, 1.6):
        pred = (lambda r, th=thr: r > th)
        for t in (STANDARD, JIKU):
            res = eval_formation(records, hate_fn, want_type=t,
                                 ratio_pred=pred, ratio_by_race=ratio_by_race)
            if res and res["n_hits"] >= MIN_HITS:
                scan.append((res["roi_eq"], res["roi_du"],
                             f"◎嫌いF1 過剰人気>{thr} [{t}]", res))
    scan.sort(key=lambda x: -max(x[0], x[1]))
    print("有効的中(≥10本)ありの条件を ROI(均等/ドッチの高い方)降順 top10:")
    print(f"{'条件':<34}{'R数':>6}{'的中':>6}{'ROI均等':>9}{'ROIドッチ':>9}")
    for roi_eq, roi_du, label, res in scan[:10]:
        star = " ★>100%" if max(roi_eq, roi_du) > 1.0 else ""
        print(f"{label:<34}{res['n_races']:>6}{res['n_hits']:>6}"
              f"{_fmt(roi_eq):>9}{_fmt(roi_du):>9}{star}")
    if scan:
        top = scan[0]
        over = [s for s in scan if max(s[0], s[1]) > 1.0]
        print()
        if over:
            print(f"→ ≤30点で黒字(>100%)候補あり: {len(over)}件。最良 {top[2]} "
                  f"ROI均等{_fmt(top[0])}/ドッチ{_fmt(top[1])}（要・的中本数と分散を確認）")
        else:
            print(f"→ ≤30点・的中≥10本での黒字(>100%)候補は無し。"
                  f"最良は {top[2]} ROI均等{_fmt(top[0])}/ドッチ{_fmt(top[1])}。")
    print("（混戦◎嫌い≈97.3%の優位が標準/軸堅でも再現するかは上表・掃引で判断）")


if __name__ == "__main__":
    main()
