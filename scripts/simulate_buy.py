"""確定済みレース(直近N日)で「現行の買い目 vs 新戦略」の回収率をシミュレーション。

現行: ◎頭固定 三連単6点（ダッシュボードの参考フォーメーション既定）。
新  : レース種別(勝ち上がり)で可変—準決勝/決勝=◎頭6(絞る) / 予選・特選=◎頭8 / 選抜・一般=全体top10(広げる)。
      ガールズは常に◎軸(◎頭)。三連複2点の保険は別建て(直近日はtrio払戻未収集のため参考注記)。

本番モデル(load_for)＋補正確率で予測し、実払戻(payouts_trifecta)で決済。両戦略は同じ予測を使い
買い方だけ変える＝相対比較はリーク影響を受けない。

  PYTHONIOENCODING=utf-8 python scripts/simulate_buy.py            # ガールズ
  PYTHONIOENCODING=utf-8 python scripts/simulate_buy.py --men --days 2
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples
from src.model.feature_sets import load_for
from src.features.rider_narabi import compute_narabi_features
from src.model.himo_adjust import corrected_trifecta_probs, MEN_PARAMS, DEFAULT_PARAMS


def _actual_date(rid):
    y, m, d = int(rid[2:6]), int(rid[6:8]), int(rid[8:10])
    return date(y, m, d) + timedelta(days=int(rid[10:12]) - 1)


def _role(name):
    if not name:
        return "標"
    for kw, lab in [("準決", "堅"), ("決勝", "堅"), ("予選", "標"),
                    ("選抜", "荒"), ("一般", "荒"), ("特選", "標")]:
        if kw in name:
            return lab
    return "標"


def _load(db):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    role = {rid: _role(nm) for rid, nm in c.execute("SELECT race_id,race_name FROM races")}
    pay = {}
    for rid, combo, p in c.execute("SELECT race_id,combo,payout FROM payouts_trifecta"):
        pay[rid] = (tuple(int(x) for x in combo.split("-")), p)
    tmp = defaultdict(list)
    for rid, car, li, pi in c.execute(
            "SELECT race_id,car_number,line_id,pos_in_line FROM narabi WHERE line_id IS NOT NULL"):
        tmp[rid].append((li, pi, car))
    c.close()
    lines = {}
    for rid, rows in tmp.items():
        mx = max(li for li, _, _ in rows)
        ls = [[] for _ in range(mx + 1)]
        for li, pi, car in sorted(rows, key=lambda x: (x[0], x[1])):
            ls[li].append(car)
        lines[rid] = [x for x in ls if x]
    return role, pay, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--men", action="store_true")
    ap.add_argument("--days", type=int, default=2, help="直近何日分の確定レースを対象にするか")
    args = ap.parse_args()
    is_girls = not args.men
    db = str(DATA_DIR / ("keirin_men.sqlite" if args.men else "keirin.sqlite"))
    params = MEN_PARAMS if args.men else DEFAULT_PARAMS
    model, elo, lbl = load_for(is_girls)
    if model is None:
        print("本番モデル未配置"); return
    role, pay, lines = _load(db)

    base = load_samples(db, field_size=([7, 9] if args.men else 7), features=PL_FEATURES_FULL)
    samples = augment_samples(base, db, model.feature_names)
    narabi = compute_narabi_features(db)

    # 払戻ありレースの実施日→直近days日
    dated = [(s, _actual_date(s.race_id)) for s in samples if s.race_id in pay]
    if not dated:
        print("対象レースなし"); return
    days = sorted({d for _, d in dated})[-args.days:]
    target = [s for s, d in dated if d in days]
    print(f"{lbl} 直近{args.days}日 {days[0]}〜{days[-1]}  対象{len(target)}レース  モデル={type(model).__name__}\n")

    def topk(dist, k, pred=None):
        it = [(o, p) for o, p in dist.items() if (pred is None or pred(o))]
        it.sort(key=lambda op: -op[1])
        return set(o for o, _ in it[:k])

    cur = {"stake": 0, "ret": 0, "hit": 0}
    new = {"stake": 0, "ret": 0, "hit": 0}
    bydate = defaultdict(lambda: {"cur_s": 0, "cur_r": 0, "new_s": 0, "new_r": 0, "n": 0})
    for s in target:
        st = model.strengths(s.X, s.car_numbers)
        fav = max(st, key=st.get)
        npos = {cc: narabi.get((s.race_id, cc), {}).get("narabi_pos") for cc in s.car_numbers}
        dist = corrected_trifecta_probs(st, npos, params, lines=lines.get(s.race_id))
        combo, p = pay[s.race_id]
        r = role.get(s.race_id, "標")
        # 現行: ◎頭6
        cbuy = topk(dist, 6, lambda o: o[0] == fav)
        # 新: 種別別
        if not is_girls and r == "荒":
            nbuy = topk(dist, 10)                          # 選抜/一般=手広く
        elif r == "堅":
            nbuy = topk(dist, 6, lambda o: o[0] == fav)    # 準決勝/決勝=◎頭6絞る
        else:
            nbuy = topk(dist, 8, lambda o: o[0] == fav)    # 予選/特選=◎頭8
        for tag, buy, acc in (("cur", cbuy, cur), ("new", nbuy, new)):
            acc["stake"] += len(buy) * 100
            hit = tuple(combo) in buy
            acc["ret"] += p if hit else 0
            acc["hit"] += int(hit)
        bd = bydate[_actual_date(s.race_id)]
        bd["n"] += 1
        bd["cur_s"] += len(cbuy) * 100; bd["cur_r"] += p if tuple(combo) in cbuy else 0
        bd["new_s"] += len(nbuy) * 100; bd["new_r"] += p if tuple(combo) in nbuy else 0

    n = len(target)
    def line(name, a):
        roi = a["ret"] / a["stake"] * 100 if a["stake"] else 0
        print(f"  {name:<8} 点数計{a['stake']//100:>4}  払戻計{a['ret']:>7}円  回収率{roi:>6.1f}%  的中{a['hit']}/{n}({a['hit']/n*100:.0f}%)")
    print("【全体】現行(◎頭6) vs 新(種別別)")
    line("現行", cur); line("新", new)
    print("\n【日別】")
    for d in sorted(bydate):
        b = bydate[d]
        cr = b["cur_r"] / b["cur_s"] * 100 if b["cur_s"] else 0
        nr = b["new_r"] / b["new_s"] * 100 if b["new_s"] else 0
        print(f"  {d} ({b['n']}R): 現行 回収{cr:.1f}%(払戻{b['cur_r']}/賭{b['cur_s']}) / 新 回収{nr:.1f}%(払戻{b['new_r']}/賭{b['new_s']})")


if __name__ == "__main__":
    main()
