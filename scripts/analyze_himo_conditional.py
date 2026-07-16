"""紐(2着力)の条件付き構造分析。◎の勝ち負けで条件付けた実分布 vs PL(独立)予測のズレを測る。

現行はPlackett-Luce＝強さから独立に着順を引く前提。だが競輪の展開では
  ・◎(自力)が勝つ時、その直後マーク/同県の選手が2着に来やすい(番手・連携)
  ・◎が負ける時、◎を差した番手や中団の捲りが1着に来やすい
といった「◎との関係性」由来の偏りがPLでは表せない可能性がある。

各out-of-sampleレースで strengths→win_probs を出し、
  (A) ◎が1着のとき: 実際の2着が [モデル○ / ◎の直後マーク(並び) / ◎と同県] である率を、
      PL条件付き期待率 p[j]/(1-p[◎]) の総和と比較。
  (B) ◎が2着以下のとき: 実際の1着が [モデル○ / ◎の直後マーク / 中団(並び2-4) / 捲自在脚質] である率を、
      同じPL期待率と比較。
実 − PL期待 が＋かつ頑健なら、その関係を紐選定で加点する価値がある。

  PYTHONIOENCODING=utf-8 python scripts/analyze_himo_conditional.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS, compute_narabi_features
from src.backtest.walkforward import fold_boundaries


def _aux(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    leg, pref = {}, {}
    for rid, car, lt, pf in c.execute("SELECT race_id,car_number,leg_type,prefecture FROM entries"):
        leg[(rid, car)] = lt
        pref[(rid, car)] = pf
    c.close()
    return leg, pref


def _is_aggr(lt: str | None) -> bool:
    return bool(lt) and any(k in lt for k in ("捲", "自在", "両", "逃", "先"))


class Acc:
    """実測率と PL期待率を貯めて比較する。"""
    def __init__(self, label):
        self.label = label
        self.n = 0
        self.act = 0.0     # 実際に条件を満たした回数
        self.exp = 0.0     # PL期待率の総和
        self.byfold = {}   # fold -> [act, exp, n]

    def add(self, hit: float, exp: float, fold: int):
        self.n += 1
        self.act += hit
        self.exp += exp
        a = self.byfold.setdefault(fold, [0.0, 0.0, 0])
        a[0] += hit; a[1] += exp; a[2] += 1

    def line(self):
        if self.n == 0:
            return f"  {self.label:<22} n=0"
        act = self.act / self.n * 100
        exp = self.exp / self.n * 100
        agree = 0; tot = 0
        for f, (a, e, m) in self.byfold.items():
            if m >= 10:
                tot += 1
                if (a / m - e / m) * (self.act / self.n - self.exp / self.n) > 0:
                    agree += 1
        return (f"  {self.label:<22} n={self.n:>5}  実{act:>5.1f}%  PL期待{exp:>5.1f}%  "
                f"差{act-exp:>+5.1f}pt  fold一致{agree}/{tot}")


def main():
    ap = argparse.ArgumentParser(description="紐の条件付き構造(実 vs PL)")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    base = load_samples(args.db, features=PL_FEATURES_FULL)
    feats = list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES) + list(NARABI_KEYS)
    samples = augment_samples(base, args.db, feats)
    leg, pref = _aux(args.db)
    narabi = compute_narabi_features(args.db)
    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40, window="expanding")

    # (A) ◎勝ち時の2着
    A_maru = Acc("2着=モデル○")
    A_mark = Acc("2着=◎の直後マーク")
    A_pref = Acc("2着=◎と同県")
    # (B) ◎負け時の1着
    B_maru = Acc("勝者=モデル○")
    B_mark = Acc("勝者=◎の直後マーク")
    B_mid = Acc("勝者=中団(並び2-4)")
    B_aggr = Acc("勝者=捲/自在/両/逃脚質")
    win_when_lose_isuma = Acc("(参考)◎負け率")

    for fi, (a, b, c) in enumerate(bounds):
        model = train_gbdt(samples[a:b])
        for s in samples[b:c]:
            st = model.strengths(s.X, s.car_numbers)
            fav = max(st, key=st.get)
            p_fav = st[fav]
            denom = 1.0 - p_fav
            if denom <= 1e-9:
                continue
            others = [car for car in s.car_numbers if car != fav]
            maru = max(others, key=st.get)                      # モデル○
            fav_pos = narabi.get((s.race_id, fav), {}).get("narabi_pos")
            # ◎の直後マーク = 並び位置が ◎+1 の選手
            markers = [car for car in others
                       if fav_pos is not None
                       and narabi.get((s.race_id, car), {}).get("narabi_pos") == fav_pos + 1]
            samepref = [car for car in others if pref.get((s.race_id, car)) and pref.get((s.race_id, car)) == pref.get((s.race_id, fav))]
            mids = [car for car in others
                    if (lambda p: p is not None and 2 <= p <= 4)(narabi.get((s.race_id, car), {}).get("narabi_pos"))]
            aggrs = [car for car in others if _is_aggr(leg.get((s.race_id, car)))]

            def pl(cars):   # PL条件付き期待率の総和 P(その集合の誰かが該当ポジション)
                return sum(st[x] for x in cars) / denom

            won = (s.order[0] == fav)
            if won:
                second = s.order[1] if len(s.order) > 1 else None
                if second is not None and second != fav:
                    A_maru.add(1.0 if second == maru else 0.0, st[maru] / denom, fi)
                    A_mark.add(1.0 if second in markers else 0.0, pl(markers), fi)
                    A_pref.add(1.0 if second in samepref else 0.0, pl(samepref), fi)
            else:
                winner = s.order[0]
                B_maru.add(1.0 if winner == maru else 0.0, st[maru] / denom, fi)
                B_mark.add(1.0 if winner in markers else 0.0, pl(markers), fi)
                B_mid.add(1.0 if winner in mids else 0.0, pl(mids), fi)
                B_aggr.add(1.0 if winner in aggrs else 0.0, pl(aggrs), fi)
            win_when_lose_isuma.add(0.0 if won else 1.0, denom, fi)

    print("紐の条件付き構造: 実測率 vs PL(独立)期待率。差＋かつfold一致が高いほど PLの取りこぼし\n")
    print(f"参考: {win_when_lose_isuma.line()}\n")
    print("【A】◎が1着のとき、2着は誰か（◎勝ちレースのみ）")
    for acc in (A_maru, A_mark, A_pref):
        print(acc.line())
    print("\n【B】◎が2着以下のとき、1着は誰か（◎負けレースのみ）")
    for acc in (B_maru, B_mark, B_mid, B_aggr):
        print(acc.line())


if __name__ == "__main__":
    main()
