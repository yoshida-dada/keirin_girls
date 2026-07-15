"""三連単「紐選定戦略」の実装と決済（S5・課題A/Phase4の派生）。

1着勝率予測が良好になった次の課題は「2着以下の紐（himo）をどう選べば回収率が上がるか」。
ユーザーの着眼＝**勝率（モデル確率）が同等なら高オッズの紐を選ぶ方が期待値(EV=prob×odds)が高い**。
本モジュールは bucket_analysis.build_records が返す ComboRecord（レース×210通り全買い目・モデル非依存）を
消費し、レースごとに「買う買い目集合」を選ぶ戦略関数群と、共通の決済関数を提供する。

戦略関数はいずれも「1レース分の recs（ComboRecord のリスト）」を受け取り、買う ComboRecord の
部分リストを返す（純粋関数・副作用なし）。複数レースへの適用は run_strategy が担う。

  ・select_all_ev        : EV≥thr を全買い（現行ベースライン）
  ・select_head_fixed    : 1着をtop1に固定し、2着/3着候補を order(ev|prob) で選ぶフォーメーション
  ・select_head_box      : 上位heads車を頭にしたフォーメーション（頭ボックス）
  ・select_longshot_himo : prob順の安全フォーメーションに、EV条件を満たす高オッズ紐を追加取り込み

★リーク防止: records は build_records 呼び出し側で検証期間(out-of-sample)に限定済み。本モジュールは
  確率もオッズも records の値をそのまま使うだけで、学習・分割には一切関与しない。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable

from src.backtest.bucket_analysis import ComboRecord

_STAKE = 100  # 1点あたり投票額（円）。bucket_analysis と同一。


# --------------------------------------------------------------------------
# レース単位のユーティリティ
# --------------------------------------------------------------------------
def group_by_race(records: list[ComboRecord]) -> dict[str, list[ComboRecord]]:
    """records を race_id でグルーピングする。"""
    races: dict[str, list[ComboRecord]] = defaultdict(list)
    for r in records:
        races[r.race_id].append(r)
    return dict(races)


def _first_place_probs(recs: list[ComboRecord]) -> dict[int, float]:
    """1レース分の全買い目から、各車の1着確率 P(1着=car)=Σ_{combo[0]=car} model_prob を復元。"""
    d: dict[int, float] = defaultdict(float)
    for r in recs:
        d[r.combo[0]] += r.model_prob
    return d


def _head_scores(recs: list[ComboRecord], key: str) -> dict[int, float]:
    """頭候補の車を並べるためのスコア。key='prob'→1着確率, key='ev'→頭固定時のEV総和。"""
    d: dict[int, float] = defaultdict(float)
    for r in recs:
        d[r.combo[0]] += r.model_prob if key == "prob" else r.ev
    return d


def _rank_cars_for_position(recs: list[ComboRecord], head: int, pos: int,
                            order: str) -> list[int]:
    """頭を head に固定したとき、pos番目(1=2着,2=3着)の紐候補車を order で降順に並べる。

    order='prob' … その位置に来る model_prob 総和が大きい車（＝人気の堅い紐）
    order='ev'   … その位置の EV 総和が大きい車（prob が同等なら高オッズ＝高EV が上位に来る）
    order='odds' … その位置の平均オッズが大きい車（純粋な高配当狙い・比較用）
    """
    score: dict[int, float] = defaultdict(float)
    cnt: dict[int, int] = defaultdict(int)
    for r in recs:
        if r.combo[0] != head:
            continue
        car = r.combo[pos]
        cnt[car] += 1
        if order == "prob":
            score[car] += r.model_prob
        elif order == "ev":
            score[car] += r.ev
        elif order == "odds":
            score[car] += r.odds
        else:
            raise ValueError(f"unknown order: {order}")
    if order == "odds":
        score = {c: v / cnt[c] for c, v in score.items()}
    return [c for c, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def _formation(recs: list[ComboRecord], head: int, n_second: int, n_third: int,
               order: str, ev_thr: float | None) -> list[ComboRecord]:
    """頭 head 固定・2着n_second車×3着n_third車のフォーメーション買い目を返す。"""
    second = set(_rank_cars_for_position(recs, head, 1, order)[:n_second])
    third = set(_rank_cars_for_position(recs, head, 2, order)[:n_third])
    out: list[ComboRecord] = []
    for r in recs:
        a, b, c = r.combo
        if a == head and b in second and c in third and b != c:
            if ev_thr is None or r.ev >= ev_thr:
                out.append(r)
    return out


# --------------------------------------------------------------------------
# 戦略関数（いずれも 1レース分の recs → 買う ComboRecord のリスト）
# --------------------------------------------------------------------------
def select_all_ev(recs: list[ComboRecord], thr: float = 1.10) -> list[ComboRecord]:
    """EV≥thr を全買い（現行ベースライン。頭も紐も選定しない全210点フィルタ）。"""
    return [r for r in recs if r.ev >= thr]


def select_head_fixed(recs: list[ComboRecord], head_by: str = "prob",
                      n_second: int = 4, n_third: int = 4, order: str = "ev",
                      ev_thr: float | None = None) -> list[ComboRecord]:
    """1着を top1（head_by で選ぶ最有力車）に固定し、2着/3着候補を order で選ぶ。

    head_by='prob'（1着確率最大）/ 'ev'（頭固定EV総和最大）。
    order='ev' と 'prob' を切り替えて「高EV紐 vs 高確率紐」を直接比較できる（ユーザー仮説の検証軸）。
    ev_thr を与えると、選んだフォーメーション内をさらに EV でフィルタする。
    """
    scores = _head_scores(recs, head_by)
    if not scores:
        return []
    head = max(scores, key=scores.get)
    return _formation(recs, head, n_second, n_third, order, ev_thr)


def select_head_box(recs: list[ComboRecord], heads: int = 2, n_second: int = 4,
                    n_third: int = 4, order: str = "ev",
                    ev_thr: float | None = None) -> list[ComboRecord]:
    """上位 heads 車それぞれを頭にしたフォーメーションの和集合（頭ボックス）。

    頭は1着確率上位 heads 車。各頭について order で 2着/3着候補を選び、買い目を重複排除して返す。
    """
    fp = _first_place_probs(recs)
    if not fp:
        return []
    head_cars = [c for c, _ in sorted(fp.items(), key=lambda kv: -kv[1])[:heads]]
    seen: set[tuple] = set()
    out: list[ComboRecord] = []
    for h in head_cars:
        for r in _formation(recs, h, n_second, n_third, order, ev_thr):
            if r.combo not in seen:
                seen.add(r.combo)
                out.append(r)
    return out


def select_longshot_himo(recs: list[ComboRecord], head_by: str = "prob",
                         n_second: int = 4, n_third: int = 4,
                         base_order: str = "prob", ev_thr: float = 1.20,
                         min_odds: float = 50.0) -> list[ComboRecord]:
    """prob順の安全フォーメーションに、EV条件を満たす高オッズ紐(head固定)を追加取り込みする。

    ベース = head固定・base_order（既定 prob＝堅い紐）フォーメーション。
    追加   = 同じ頭で odds≥min_odds かつ EV≥ev_thr の買い目（人気薄でもEVが正当化する高配当紐）。
    「高オッズ紐をEV条件付きで取り込む」効果を、ベースとの差分で測るための戦略。
    """
    scores = _head_scores(recs, head_by)
    if not scores:
        return []
    head = max(scores, key=scores.get)
    base = _formation(recs, head, n_second, n_third, base_order, None)
    seen = {r.combo for r in base}
    out = list(base)
    for r in recs:
        if r.combo[0] == head and r.combo not in seen \
                and r.odds >= min_odds and r.ev >= ev_thr:
            seen.add(r.combo)
            out.append(r)
    return out


# --------------------------------------------------------------------------
# 決済・評価
# --------------------------------------------------------------------------
def settle(bought: list[ComboRecord]) -> dict:
    """買い目集合を決済して集計指標を返す。

    戻り値:
      n_bets          : 買い目総数（＝点数の総和）
      n_hits          : 的中買い目数（三連単は1レース1点しか当たらないので＝的中レース数）
      stake / ret     : 投票額 / 回収額（円。payout は100円あたり払戻）
      roi             : ret / stake（控除率≈25%のため1.0未満が基本）
      n_races         : 実際に買ったレース数（≥1点買ったレースのみ）
      hit_rate        : n_hits / n_races（レース的中率）
      points_per_race : n_bets / n_races（平均点数）
    """
    n_bets = len(bought)
    stake = n_bets * _STAKE
    n_hits = sum(1 for r in bought if r.is_win)
    ret = sum(r.payout for r in bought if r.is_win)
    n_races = len({r.race_id for r in bought})
    return {
        "n_bets": n_bets,
        "n_hits": n_hits,
        "stake": stake,
        "ret": ret,
        "roi": round(ret / stake, 4) if stake else None,
        "n_races": n_races,
        "hit_rate": round(n_hits / n_races, 4) if n_races else None,
        "points_per_race": round(n_bets / n_races, 3) if n_races else None,
    }


def run_strategy(records: list[ComboRecord],
                 strategy: Callable[..., list[ComboRecord]],
                 race_type: str | None = None, **kwargs) -> dict:
    """全レースに strategy を適用して決済する。race_type 指定時はそのタイプのレースだけ集計。

    strategy は「1レース分の recs → 買う ComboRecord のリスト」。kwargs は strategy に渡す。
    """
    bought: list[ComboRecord] = []
    for recs in group_by_race(records).values():
        if race_type is not None and recs and recs[0].race_type != race_type:
            continue
        bought.extend(strategy(recs, **kwargs))
    return settle(bought)
