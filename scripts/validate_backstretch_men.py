"""(b) 男子の主導権(B)予測モデルを walk-forward で検証する。

男子はガールズと違い「誰が主導権を取るか」がほぼ決着構造を決める。実測(25,269R)では
ライン先頭の主導権率が 先行55.2% / 押え先34.7% / カマシ33.0% / 自在15.0% と脚質で40pt違い、
過去B回数で層別してもなお分かれる。この情報は (a) で脚質one-hotとして本番特徴に入った。

**比較する基準線（すべて同じ hold-out で測る）**:
  1. 記者の並び予想の先頭（ライン先頭のうち隊列最前）… Phase0で48%と実測
  2. 直近B回数(b_count)の最大 … ガールズでは51.3%
  3. 学習モデル（本モデル）

**事前登録した採否基準（後から緩めない）**:
  主基準: B的中が上の基準線1・2の**両方**を過半fold(3/5以上)で上回る
  副基準: 平均B的中が基準線の最良を +2pt 以上上回る
  外れたら「展開表示に使わない」。当たらない主導権予測を出すと展開の読みを丸ごと誤らせる。

  PYTHONIOENCODING=utf-8 python scripts/validate_backstretch_men.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_gbdt import train_gbdt
from src.model.feature_augment import augment_samples
from src.model.feature_sets import men_features
from src.backtest.walkforward import fold_boundaries


def _ctx(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    b_of: dict[str, list[int]] = defaultdict(list)
    for rid, car, v in c.execute("SELECT race_id,car_number,sb FROM results"):
        if v and "B" in str(v):
            b_of[rid].append(car)
    head, bcnt = {}, defaultdict(dict)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi"
            " WHERE line_id IS NOT NULL"):
        if li == 0 and pi == 0:          # 隊列の最前＝記者予想の先頭ライン先頭
            head[rid] = car
    for rid, car, b in c.execute("SELECT race_id,car_number,b_count FROM recent_form"):
        bcnt[rid][car] = b or 0
    c.close()
    # B取得者が一意なレースだけを対象にする（複数Bは判定が曖昧）
    return {rid: cs[0] for rid, cs in b_of.items() if len(cs) == 1}, head, bcnt


def main() -> None:
    ap = argparse.ArgumentParser(description="男子の主導権予測モデルの検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    b_taker, head, bcnt = _ctx(args.db)
    feats = men_features()
    raw = load_samples(args.db, field_size=[7, 9], features=PL_FEATURES_FULL)
    samples = [s for s in augment_samples(raw, args.db, feats) if s.race_id in b_taker]
    print(f"対象 {len(samples):,}レース（B取得者が一意）  特徴{len(samples[0].feature_names)}列")

    bounds = fold_boundaries(len(samples), n_folds=args.folds, warmup_frac=0.40,
                             window="expanding")
    print(f"\n{'fold':>5}{'n_test':>8}{'記者先頭':>10}{'B回数最大':>11}{'モデル':>9}"
          f"{'vs記者':>9}{'vs B回数':>10}")
    d_rep, d_bc, m_all = [], [], []
    for fi, (a, b, c) in enumerate(bounds):
        train = samples[a:b]
        test = samples[b:c]
        # Stage1: 目的変数を「B取得車が1位」の順位に差し替えて lambdarank を流用
        tr = []
        for s in train:
            t = type(s)(**{**s.__dict__})
            bt = b_taker[s.race_id]
            t.order = [bt] + [x for x in s.car_numbers if x != bt]
            tr.append(t)
        model = train_gbdt(tr)
        n = rep = bc = mm = 0
        for s in test:
            truth = b_taker[s.race_id]
            n += 1
            rep += int(head.get(s.race_id) == truth)
            bm = max(s.car_numbers, key=lambda c2: bcnt[s.race_id].get(c2, 0))
            bc += int(bm == truth)
            st = model.strengths(s.X, s.car_numbers)
            if st:
                mm += int(max(st, key=st.get) == truth)
        r, bb, m = rep / n * 100, bc / n * 100, mm / n * 100
        d_rep.append(m - r)
        d_bc.append(m - bb)
        m_all.append(m)
        print(f"{fi:>5}{n:>8}{r:>9.1f}%{bb:>10.1f}%{m:>8.1f}%{m-r:>+9.1f}{m-bb:>+10.1f}")

    nr = sum(1 for d in d_rep if d > 0)
    nb2 = sum(1 for d in d_bc if d > 0)
    mean_m = sum(m_all) / len(m_all)
    mean_best = mean_m - min(sum(d_rep) / len(d_rep), sum(d_bc) / len(d_bc))
    print(f"\n平均: モデル {mean_m:.1f}% / 基準線の最良 {mean_best:.1f}%  "
          f"(Δ{mean_m-mean_best:+.1f}pt)")
    ok_main = nr >= 3 and nb2 >= 3
    ok_sub = (mean_m - mean_best) >= 2.0
    print("\n事前基準の判定:")
    print(f"  主基準（記者先頭とB回数最大の両方を3/5fold以上で上回る）: "
          f"記者 {nr}/5 / B回数 {nb2}/5 → {'充足' if ok_main else '不充足'}")
    print(f"  副基準（基準線の最良を+2pt以上）: {'充足' if ok_sub else '不充足'}")
    print(f"\n→ {'採用（展開表示に使う）' if ok_main and ok_sub else '不採用'}")


if __name__ == "__main__":
    main()
