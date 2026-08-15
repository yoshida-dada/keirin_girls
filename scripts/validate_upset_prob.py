"""波乱確率を「万車券率」として実測から出し直す（男子）。

**現状の問題**: ダッシュボードの「波乱 50.6%」は `1 - 本命の1着確率` でしかない。
これは「本命が1着を外す確率」であって、荒れて高配当になる確率ではない。
本命が飛んでも2番人気→3番人気で決まれば配当は安く、波乱とは呼べない。

**やりたいこと**: 三連単の払戻が10,000円以上になる確率＝万車券率を出す。
実測ベース（男子25,335レース）で 28.37%。

比較する推定量:
  cur    現行の代理指標 `1 - max(win_prob)`。そのまま万車券率として読めるか。
  pl     モデル分布だけから: 控除率25%なので 配当 ≈ 0.75/p。
         p <= 0.75/100 = 0.0075 の目を「万車券になる目」とみなし、その確率を合計する。
         オッズが無くても出せる＝発売前でも表示できる。
  mkt    モデル分布 × 実オッズ: sum( p_model(目) ) over 実オッズ>=100倍 の目。
         市場が「万車券圏」と値付けした目に、モデルがどれだけ確率を置いているか。
  plc    pl と同じ形だが、**しきい値を過去データから推定する**。
         0.75/p という控除率の仮定は男女で合っておらず、ガールズでは平均13.50%に対し
         実測16.94%と系統的に低く出た（男子は+2.5pt高い）。そこで
         「それ以前のfoldの out-of-sample 予測」で、予測平均が実測率に一致する
         しきい値を二分探索で求め、次のfoldに使う。fold0は推定材料が無いので評価から外す。
         時系列的に前のデータしか使わないのでリークしない。
  base   定数（学習期間の万車券率）。識別力ゼロの下限。これに勝てなければ意味が無い。

  pls    plc と同じだが、**車立てごとに**しきい値を推定する。
         pl を全体プールで判定したとき ECE 0.0278 で基準を満たしたが、車立てで層別すると
         **9車で平均65.8%に対し実測45.4%（+20pt）** と大きく外れていた。男子の9車は
         2,353/25,335=9.3% しかなく、全体プールのECEでは埋もれる。
         組合せ数が 210(7車) と 504(9車) で倍以上違うので、同じ p のしきいを当てれば
         9車の方が多くの目が万車券圏に入る。層を分けなければ直らない。

**plc / pls について事前登録した基準（pl とは別に宣言する。plの基準を緩めたものではない）**:
  主基準: **7車と9車それぞれで** ECE <= 0.03 かつ その車立ての定数予測より Brier が良い
          （全体プールのECEだけでは 9車の偏りが埋もれるため、層別を主基準に置く）
  副基準: それぞれの層で十分位の実測が単調増加
  男女それぞれで判定する。片方が通っても他方には流用しない。
  fold0 はしきい値の推定材料が無いので評価から外す。

**事前登録した採否基準（後から緩めない）**:
  主基準: out-of-sample の ECE <= 0.03（3pt）かつ 5foldの過半で base より Brier が良い
  副基準: 予測値の十分位ごとの実測万車券率が単調増加（順序が壊れていない）
  どちらも満たす推定量だけを採用する。満たさなければ現行表示のまま「本命が1着を
  外す確率」と正しく言い換える（万車券率だと偽らない）。

  PYTHONIOENCODING=utf-8 python scripts/validate_upset_prob.py
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
from src.model.feature_sets import men_features, girls_features
from src.model.plackett_luce import all_trifecta_probs
from src.backtest.walkforward import fold_boundaries

MAN = 10000          # 万車券のしきい（100円あたりの払戻）
TAKEOUT = 0.75       # 控除率25% → 配当 ≈ TAKEOUT / p
P_MAN = TAKEOUT / (MAN / 100)      # = 0.0075


def load_labels(db) -> dict[str, int]:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {r: int(p >= MAN) for r, p in
           c.execute("SELECT race_id, payout FROM payouts_trifecta WHERE payout IS NOT NULL")}
    c.close()
    return out


def load_big_odds(db) -> dict[str, set]:
    """race_id → 確定オッズが100倍以上の目の集合。"""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, set] = {}
    for rid, combo in c.execute(
            "SELECT race_id, combo FROM odds_final_trifecta WHERE odds >= 100"):
        out.setdefault(rid, set()).add(combo)
    c.close()
    return out


def _ece(pred: list[float], y: list[int], bins: int = 10) -> float:
    """十分位ビンでの |予測平均 - 実測率| の加重平均。"""
    if not pred:
        return 0.0
    idx = sorted(range(len(pred)), key=lambda i: pred[i])
    n, tot = len(pred), 0.0
    for b in range(bins):
        part = idx[b * n // bins:(b + 1) * n // bins]
        if not part:
            continue
        pm = sum(pred[i] for i in part) / len(part)
        ym = sum(y[i] for i in part) / len(part)
        tot += len(part) * abs(pm - ym)
    return tot / n


def _brier(pred, y) -> float:
    return sum((p - t) ** 2 for p, t in zip(pred, y)) / len(pred) if pred else 0.0


def _deciles(pred, y, bins: int = 10):
    idx = sorted(range(len(pred)), key=lambda i: pred[i])
    n = len(pred)
    rows = []
    for b in range(bins):
        part = idx[b * n // bins:(b + 1) * n // bins]
        if not part:
            continue
        rows.append((sum(pred[i] for i in part) / len(part),
                     sum(y[i] for i in part) / len(part), len(part)))
    return rows


def fit_threshold(dists: list[dict], ys: list[int]) -> float:
    """予測平均が実測率に一致する p のしきい値を二分探索する。

    sum_{p<=t} p は t について単調増加なので二分探索でよい。
    渡すのは**評価対象より前のfoldの out-of-sample 予測**だけ（リーク防止）。
    """
    if not dists:
        return P_MAN
    target = sum(ys) / len(ys)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        m = sum(sum(p for p in d.values() if p <= mid) for d in dists) / len(dists)
        if m < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    ap = argparse.ArgumentParser(description="波乱確率＝万車券率の検証")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin_men.sqlite"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--girls", action="store_true",
                    help="ガールズDBを見る（特徴セットと車立てが男子と違う）")
    args = ap.parse_args()

    lab = load_labels(args.db)
    big = load_big_odds(args.db)
    print(f"払戻あり {len(lab):,}R / 万車券 {sum(lab.values()):,} "
          f"({sum(lab.values())/len(lab)*100:.2f}%)  オッズあり {len(big):,}R")

    # 特徴セットも車立ても男女で違う。ここを取り違えると別モデルの列を食わせることになる
    feats = girls_features() if args.girls else men_features()
    raw = load_samples(args.db, field_size=[7] if args.girls else [7, 9],
                       features=PL_FEATURES_FULL)
    smp = augment_samples(raw, args.db, feats)
    print(f"サンプル {len(smp):,}（{len(smp[0].feature_names)}列）")

    names = ["cur", "pl", "mkt", "plc", "pls", "base"]
    acc = {k: {"p": [], "y": []} for k in names}
    past_d: list[dict] = []      # 過去foldの三連単分布（plc のしきい値推定用）
    past_y: list[int] = []
    past_f: list[int] = []       # 車立て（pls 用）
    thr_used: list[float] = []
    fs_of: dict[str, int] = {}   # race_id → 車立て
    strata: list[tuple] = []     # (推定量, 車立て, 予測, 実測) 層別判定用
    print(f"\n{'fold':>5}{'n':>7}{'実測':>8}"
          + "".join(f"{k+' ECE':>11}{k+' Brier':>12}" for k in names))
    wins = {k: 0 for k in names}
    for fi, (a, b, c2) in enumerate(fold_boundaries(len(smp), n_folds=args.folds,
                                                    warmup_frac=0.40, window="expanding")):
        model = train_gbdt(smp[a:b])
        rate = sum(lab[s.race_id] for s in smp[a:b] if s.race_id in lab) / \
               max(1, sum(1 for s in smp[a:b] if s.race_id in lab))    # 学習期間の万車券率
        cur = {k: {"p": [], "y": []} for k in names}
        thr = fit_threshold(past_d, past_y)          # 前のfoldまでで推定（fold0は既定値）
        thr_used.append(thr)
        # 車立てごとのしきい値。組合せ数が違うので同じしきいは当てられない
        thr_fs = {f: fit_threshold([d for d, g in zip(past_d, past_f) if g == f],
                                   [y for y, g in zip(past_y, past_f) if g == f])
                  for f in set(past_f)}
        fold_d, fold_y, fold_f = [], [], []
        for s in smp[b:c2]:
            if s.race_id not in lab:
                continue
            st = model.strengths(s.X, s.car_numbers)
            if not st:
                continue
            probs = all_trifecta_probs(st)
            y = lab[s.race_id]
            wp = max(st.values()) / sum(st.values()) if sum(st.values()) else 0.0
            vals = {
                "cur": 1.0 - wp,
                "pl": sum(p for p in probs.values() if p <= P_MAN),
                "mkt": sum(p for k, p in probs.items()
                           if f"{k[0]}-{k[1]}-{k[2]}" in big.get(s.race_id, set())),
                "plc": sum(p for p in probs.values() if p <= thr),
                "pls": sum(p for p in probs.values()
                           if p <= thr_fs.get(len(s.car_numbers), thr)),
                "base": rate,
            }
            fs = len(s.car_numbers)
            fold_d.append(probs); fold_y.append(y); fold_f.append(fs)
            for k in names:
                cur[k]["p"].append(vals[k]); cur[k]["y"].append(y)
                # plc/pls は fold0（しきい値が既定値のまま＝推定材料なし）を通算から外す
                if k in ("plc", "pls") and fi == 0:
                    continue
                acc[k]["p"].append(vals[k]); acc[k]["y"].append(y)
                strata.append((k, fs, vals[k], y))
        past_d.extend(fold_d); past_y.extend(fold_y); past_f.extend(fold_f)
        n = len(cur["base"]["y"])
        obs = sum(cur["base"]["y"]) / n if n else 0.0
        bb = _brier(cur["base"]["p"], cur["base"]["y"])
        line = f"{fi:>5}{n:>7}{obs*100:>7.2f}%"
        for k in names:
            e, br = _ece(cur[k]["p"], cur[k]["y"]), _brier(cur[k]["p"], cur[k]["y"])
            if k != "base" and br < bb and not (k == "plc" and fi == 0):
                wins[k] += 1
            line += f"{e:>11.4f}{br:>12.4f}"
        print(line)
    print("\nplc のしきい値: " + " ".join(f"{t:.5f}" for t in thr_used)
          + f"（既定 {P_MAN:.5f}。fold0は既定値なので評価から外す）")

    print("\n=== 通算（全fold結合） ===")
    for k in names:
        p, y = acc[k]["p"], acc[k]["y"]
        print(f"\n[{k}] 平均予測 {sum(p)/len(p)*100:.2f}% / 実測 {sum(y)/len(y)*100:.2f}% "
              f"/ ECE {_ece(p,y):.4f} / Brier {_brier(p,y):.4f}"
              + (f" / baseに勝ったfold {wins[k]}/{args.folds}" if k != "base" else ""))
        if k == "base":
            continue
        rows = _deciles(p, y)
        mono = all(rows[i][1] <= rows[i+1][1] + 1e-9 for i in range(len(rows)-1))
        print("   十分位: " + " ".join(f"{r[1]*100:.0f}%" for r in rows)
              + f"   単調増加: {'はい' if mono else 'いいえ'}")
        nf = args.folds - 1 if k in ("plc", "pls") else args.folds
        ok = _ece(p, y) <= 0.03 and wins[k] >= (nf + 1) // 2
        print(f"   全体プール 主(ECE<=0.03 かつ 過半foldでbaseにBrier勝ち): "
              f"{'充足' if ok else '不充足'} / 副(単調): {'充足' if mono else '不充足'}")

    # --- 車立て別（主基準）。全体プールでは 9車(男子9.3%)の偏りが埋もれる ---
    print("\n=== 車立て別（ここが主基準） ===")
    sizes = sorted({f for _, f, _, _ in strata})
    for k in names:
        if k == "base":
            continue
        print(f"\n[{k}]")
        allok = True
        for f in sizes:
            sub = [(pp, yy) for kk, ff, pp, yy in strata if kk == k and ff == f]
            if len(sub) < 100:
                print(f"   {f}車: n={len(sub)} 少なすぎるため判定しない")
                continue
            p = [x for x, _ in sub]; y = [t for _, t in sub]
            obs = sum(y) / len(y)
            e, br = _ece(p, y), _brier(p, y)
            bb = _brier([obs] * len(y), y)          # その車立ての定数予測
            rows = _deciles(p, y)
            mono = all(rows[i][1] <= rows[i+1][1] + 1e-9 for i in range(len(rows)-1))
            ok = e <= 0.03 and br < bb and mono
            allok = allok and ok
            print(f"   {f}車 n={len(y):,} 予測{sum(p)/len(p)*100:>5.1f}% 実測{obs*100:>5.1f}% "
                  f"ECE {e:.4f} Brier {br:.4f}(定数{bb:.4f}) 単調{'○' if mono else '×'}"
                  f"  → {'充足' if ok else '不充足'}")
        print(f"   → {'採用可' if allok else '不採用'}")


if __name__ == "__main__":
    main()
