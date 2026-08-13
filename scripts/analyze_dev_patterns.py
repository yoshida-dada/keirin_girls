"""展開6パターンの発生率と、パターン別の「紐（2・3着）」構造を実測する。

パターン定義（◎=モデル1着本命、B=バック先頭＝主導権）:
  ◎勝 ①◎逃げ切り      : ◎が1着 & 決まり手=逃 & ◎がB取得
      ②◎捲り          : ◎が1着 & 決まり手=捲
      ③◎差し(前崩れ)   : ◎が1着 & 決まり手=差
  ◎負 ④別選手の逃げ残り : 勝者の決まり手=逃 & ◎≠B
      ⑤捲り台頭        : 勝者の決まり手=捲
      ⑥差し/マーク決着  : 勝者の決まり手=差 or ク
（①で◎がB非取得の逃げ勝ちは稀だが「①'」として別掲。決まり手は1・2着のみ記録される疎データ）

条件付けはレース前に分かる ペース区分（先行型人数の相対定義＝production の development.pace と同型）。
紐構造は「2着/3着がモデル何番手か」「B取得者が2-3着に来るか」「◎が沈む時どこまで落ちるか」で見る。

注意: ◎の判定にデプロイ済みモデルを使う（学習データを含むため◎精度は楽観側）。
      パターン発生率の**分岐比**を見る診断用であり、ROI主張には使えない。

  PYTHONIOENCODING=utf-8 python scripts/analyze_dev_patterns.py --db data/keirin.sqlite
  PYTHONIOENCODING=utf-8 python scripts/analyze_dev_patterns.py --apply dashboard/data.json --venue 伊東
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.model.persist import load_model
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.feature_augment import augment_samples

PATTERNS = ["①◎逃げ切り", "①'◎逃げ(B無)", "②◎捲り", "③◎差し",
            "④別の逃げ残り", "⑤捲り台頭", "⑥差し/マーク", "その他"]
WIN_PATS = PATTERNS[:4]


def _pace_level(b_counts: dict[int, float]) -> str:
    """先行型人数の相対定義（production の development.pace と同型）→ スロー/ミドル/ハイ。"""
    vals = [v for v in b_counts.values() if v is not None]
    if not vals or max(vals) <= 0:
        return "不明"
    thr = max(vals) * 0.4
    n_front = sum(1 for v in vals if v >= thr)
    return "スロー" if n_front <= 2 else ("ハイ" if n_front >= 4 else "ミドル")


def _load_ctx(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    res = defaultdict(dict)      # rid -> car -> (pos, sb, kim)
    for rid, car, pos, sb, kim in c.execute(
            "SELECT race_id,car_number,position,sb,kimarite FROM results"):
        res[rid][car] = (pos, sb, kim)
    bcnt = defaultdict(dict)     # rid -> car -> b_count
    for rid, car, b in c.execute("SELECT race_id,car_number,b_count FROM recent_form"):
        bcnt[rid][car] = b
    c.close()
    return res, bcnt


def _classify(fav: int, order_rank: dict[int, int], res_cars: dict) -> tuple[str, int | None]:
    """(パターン名, B取得車) を返す。"""
    b_car = next((c for c, (_, sb, _) in res_cars.items() if sb and "B" in sb), None)
    win = next((c for c, (p, _, _) in res_cars.items() if p == 1), None)
    if win is None:
        return "その他", b_car
    kim = (res_cars[win][2] or "")
    if win == fav:
        if "逃" in kim:
            return ("①◎逃げ切り" if b_car == fav else "①'◎逃げ(B無)"), b_car
        if "捲" in kim:
            return "②◎捲り", b_car
        if "差" in kim:
            return "③◎差し", b_car
        return "その他", b_car
    if "逃" in kim:
        return ("④別の逃げ残り" if b_car != fav else "その他"), b_car
    if "捲" in kim:
        return "⑤捲り台頭", b_car
    if "差" in kim or "ク" in kim:
        return "⑥差し/マーク", b_car
    return "その他", b_car


def _mark(rank: int | None) -> str:
    return {1: "◎", 2: "○", 3: "▲", 4: "△"}.get(rank, "×") if rank else "—"


def analyze(db: str, is_girls: bool = True) -> dict:
    """展開パターンの分岐比を実測する。

    is_girls=False で男子モデル(39特徴)＋男子DBを使う。男子は決まり手構造がガールズと
    真逆（男子 差50%/捲30%/逃19% vs ガールズ 逃21%/捲47%/差32%）なので、
    ガールズの分岐比を男子に流用してはいけない。
    """
    from src.model.feature_sets import load_for
    model, _elo, _lbl = load_for(is_girls)
    if model is None:
        raise SystemExit("モデルが見つかりません（男子は data/models/pl_model_men.pkl）")
    feats = list(model.feature_names)
    # 男子は7車・9車が混在する。車立てを絞らずに全部見る（分岐比は車立てに依らない構造の話）
    base = load_samples(db, features=PL_FEATURES_FULL,
                        field_size=7 if is_girls else [7, 9])
    samples = augment_samples(base, db, feats)
    res, bcnt = _load_ctx(db)

    recs = []
    for s in samples:
        rc = res.get(s.race_id)
        if not rc:
            continue
        st = model.strengths(s.X, s.car_numbers)
        if not st:
            continue
        ranked = sorted(st, key=lambda c: -st[c])
        rank = {c: i + 1 for i, c in enumerate(ranked)}
        fav = ranked[0]
        pat, b_car = _classify(fav, rank, rc)
        pos = {c: p for c, (p, _, _) in rc.items() if p}
        second = next((c for c, p in pos.items() if p == 2), None)
        third = next((c for c, p in pos.items() if p == 3), None)
        winner = next((c for c, p in pos.items() if p == 1), None)
        recs.append({"pat": pat, "pace": _pace_level(bcnt.get(s.race_id, {})),
                     "fav": fav, "b_car": b_car,
                     "fav_pos": pos.get(fav), "n": len(s.car_numbers),
                     "rw": rank.get(winner),
                     "r2": rank.get(second), "r3": rank.get(third),
                     "win_is_b": (b_car is not None and winner == b_car),
                     "b2": (b_car is not None and second == b_car),
                     "b3": (b_car is not None and third == b_car)})
    return {"recs": recs}


def report(recs: list[dict]) -> None:
    n = len(recs)
    print(f"対象 {n}レース（デプロイ済みモデルで◎判定＝楽観側。分岐比の診断用）\n")

    print("【パターン発生率】全体 と ペース区分別")
    paces = ["スロー", "ミドル", "ハイ"]
    print(f"  {'パターン':<14}{'全体':>8}" + "".join(f"{p:>9}" for p in paces))
    for p in PATTERNS:
        row = f"  {p:<14}{sum(1 for r in recs if r['pat']==p)/n*100:>7.1f}%"
        for pc in paces:
            sub = [r for r in recs if r["pace"] == pc]
            row += f"{(sum(1 for r in sub if r['pat']==p)/len(sub)*100 if sub else 0):>8.1f}%"
        print(row)
    for pc in paces:
        sub = [r for r in recs if r["pace"] == pc]
        w = sum(1 for r in sub if r["pat"] in WIN_PATS)
        print(f"  → {pc:<6} n={len(sub):>5}  ◎勝ち率 {w/len(sub)*100:>5.1f}%" if sub else "")

    print("\n【パターン別の紐構造】2着・3着がモデル何番手か（◎○▲△×）／B取得者が絡む率")
    print(f"  {'パターン':<14}{'n':>6}  {'2着の印':<26}{'3着の印':<26}{'B2着':>6}{'B3着':>6}")
    for p in PATTERNS[:-1]:
        sub = [r for r in recs if r["pat"] == p]
        if len(sub) < 40:
            print(f"  {p:<14}{len(sub):>6}  (少数, 省略)")
            continue
        def dist(key):
            d = defaultdict(int)
            for r in sub:
                d[_mark(r[key])] += 1
            return " ".join(f"{m}{d[m]/len(sub)*100:.0f}%" for m in ["◎", "○", "▲", "△", "×"])
        b2 = sum(1 for r in sub if r["b2"]) / len(sub) * 100
        b3 = sum(1 for r in sub if r["b3"]) / len(sub) * 100
        print(f"  {p:<14}{len(sub):>6}  {dist('r2'):<26}{dist('r3'):<26}{b2:>5.0f}%{b3:>5.0f}%")

    print("\n【◎が負けた時、◎はどこまで落ちるか】")
    for p in ["④別の逃げ残り", "⑤捲り台頭", "⑥差し/マーク"]:
        sub = [r for r in recs if r["pat"] == p and r["fav_pos"]]
        if not sub:
            continue
        s2 = sum(1 for r in sub if r["fav_pos"] == 2) / len(sub) * 100
        s3 = sum(1 for r in sub if r["fav_pos"] == 3) / len(sub) * 100
        out = sum(1 for r in sub if r["fav_pos"] >= 4) / len(sub) * 100
        print(f"  {p:<14}n={len(sub):>5}  ◎2着{s2:>5.1f}%  ◎3着{s3:>5.1f}%  ◎着外{out:>5.1f}%"
              f"  （◎を3着以内に残す妙味={100-out:>4.1f}%）")


MARKS = ["◎", "○", "▲", "△", "×"]


def _rank_dist(sub: list[dict], key: str) -> dict[str, float]:
    """sub 内の key(印ランク) の構成比を {印: 割合} で返す。"""
    d = defaultdict(int)
    for r in sub:
        d[_mark(r[key])] += 1
    tot = len(sub) or 1
    return {m: d[m] / tot for m in MARKS}


def _fmt_dist(dist: dict[str, float], names: dict[str, str]) -> str:
    """{印:割合} を「○ 2鈴木 48%」形式の並びへ。割合降順・0%は省略。"""
    items = [(m, p) for m, p in dist.items() if p >= 0.005]
    items.sort(key=lambda x: -x[1])
    return " / ".join(f"{m}{names.get(m,'')} {p*100:.0f}%" for m, p in items)


def _detail(recs: list[dict], pat: str, names: dict[str, str], pooled_note: str) -> None:
    """1パターンの内訳（勝者の印・2着の印・◎の着順）を印字。"""
    sub = [r for r in recs if r["pat"] == pat]
    if len(sub) < 40:
        print(f"      （履歴 n={len(sub)} と少なく内訳は省略）")
        return
    if pat.startswith(("④", "⑤", "⑥")):
        print(f"      勝つのは : {_fmt_dist(_rank_dist(sub,'rw'), names)}")
        wb = sum(1 for r in sub if r["win_is_b"]) / len(sub) * 100
        print(f"                 （うち主導権を取った車が勝つ: {wb:.0f}%）")
        # 勝者の印ごとに2着の内訳（薄いときはパターン全体へフォールバック）
        for m in MARKS:
            ws = [r for r in sub if _mark(r["rw"]) == m]
            if len(ws) >= 40:
                print(f"      {m}が勝つ時の2着: {_fmt_dist(_rank_dist(ws,'r2'), names)}")
        fp = [r for r in sub if r["fav_pos"]]
        if fp:
            s2 = sum(1 for r in fp if r["fav_pos"] == 2) / len(fp) * 100
            s3 = sum(1 for r in fp if r["fav_pos"] == 3) / len(fp) * 100
            out = 100 - s2 - s3
            print(f"      ◎の着順  : 2着 {s2:.0f}% / 3着 {s3:.0f}% / 着外 {out:.0f}%"
                  f"  → 3着以内に残る {s2+s3:.0f}%")
    else:
        print(f"      2着       : {_fmt_dist(_rank_dist(sub,'r2'), names)}")
        print(f"      3着       : {_fmt_dist(_rank_dist(sub,'r3'), names)}")
        b2 = sum(1 for r in sub if r["b2"]) / len(sub) * 100
        print(f"                 （主導権を取った車が2着: {b2:.0f}%）")
    print(f"      {pooled_note}")


def apply_to(recs: list[dict], data_json: str, venue: str) -> None:
    doc = json.loads(Path(data_json).read_text(encoding="utf-8"))
    races = [r for r in doc.get("predictions", {}).get("races", [])
             if venue in str(r.get("venue", ""))]
    if not races:
        print(f"\n{venue} のレースが {data_json} にありません")
        return
    # ペース区分別・◎勝ち/負け内でのパターン構成比（分岐比）
    def shares(pace: str):
        sub = [r for r in recs if r["pace"] == pace] or recs
        w = [r for r in sub if r["pat"] in WIN_PATS]
        l = [r for r in sub if r["pat"] not in WIN_PATS and r["pat"] != "その他"]
        sw = {p: (sum(1 for r in w if r["pat"] == p) / len(w) if w else 0) for p in WIN_PATS}
        sl = {p: (sum(1 for r in l if r["pat"] == p) / len(l) if l else 0)
              for p in ["④別の逃げ残り", "⑤捲り台頭", "⑥差し/マーク"]}
        return sw, sl

    for r in races:
        top1 = r.get("top1_prob") or 0.0
        pace = ((r.get("development") or {}).get("pace") or {})
        lvl = str(pace.get("level", ""))
        key = "ハイ" if "ハイ" in lvl else ("スロー" if "スロー" in lvl else "ミドル")
        sw, sl = shares(key)
        bs = ((r.get("development") or {}).get("backstretch") or {})
        fav_car = (r.get("riders") or [{}])[0].get("car")
        p_fav_b = (bs.get("probs") or {}).get(str(fav_car))
        print(f"\n=== {r.get('venue')} R{r.get('race_no')}  締切{r.get('deadline')} "
              f"[{r.get('race_type')}] ===")
        print(f"  ◎{fav_car} {(r.get('riders') or [{}])[0].get('name','')}  1着確率{top1*100:.1f}%"
              f" / 波乱確率{(1-top1)*100:.1f}%")
        print(f"  ペース: {lvl}（先行型{pace.get('n_front')}人）→ 履歴区分「{key}」で条件付け")
        print(f"  推定主導権: {bs.get('lead_car')}番 (P={bs.get('lead_p')})"
              f" / ◎のP(B)={p_fav_b}")
        # 印 → 「車番+氏名」の対応（riders は1着確率降順）
        rs = r.get("riders") or []
        names = {MARKS[i]: f"{rs[i].get('car')}{rs[i].get('name','')}"
                 for i in range(min(len(MARKS), len(rs)))}
        if len(rs) > len(MARKS):
            names["×"] = "以下(" + ",".join(str(x.get("car")) for x in rs[len(MARKS)-1:]) + ")"
        print("  印: " + " / ".join(f"{m}{names[m]}" for m in MARKS if m in names))
        note = "※内訳は全5794Rプール（ペース別だと薄いため）。確率はこのレースのtop1_prob×分岐比。"
        for p in WIN_PATS:
            if sw[p] < 0.005:
                continue
            print(f"\n  {p}  {top1*sw[p]*100:.1f}%")
            _detail(recs, p, names, note)
        for p in ["④別の逃げ残り", "⑤捲り台頭", "⑥差し/マーク"]:
            print(f"\n  {p}  {(1-top1)*sl[p]*100:.1f}%")
            _detail(recs, p, names, note)


def emit_stats(recs: list[dict], out: Path) -> None:
    """本番（DB非依存の refresh_predictions）から参照する統計をJSONへ書き出す。"""
    paces = ["スロー", "ミドル", "ハイ"]
    pace_rates = {}
    for pc in paces:
        sub = [r for r in recs if r["pace"] == pc]
        if not sub:
            continue
        pace_rates[pc] = {p: sum(1 for r in sub if r["pat"] == p) / len(sub) for p in PATTERNS}
    pats = {}
    for p in PATTERNS[:-1]:
        sub = [r for r in recs if r["pat"] == p]
        if len(sub) < 40:
            continue
        d = {"n": len(sub), "second": _rank_dist(sub, "r2"), "third": _rank_dist(sub, "r3"),
             "b_second": sum(1 for r in sub if r["b2"]) / len(sub)}
        if p.startswith(("④", "⑤", "⑥")):
            d["winner"] = _rank_dist(sub, "rw")
            d["b_win"] = sum(1 for r in sub if r["win_is_b"]) / len(sub)
            fp = [r for r in sub if r["fav_pos"]]
            if fp:
                d["fav_pos"] = {
                    "2": sum(1 for r in fp if r["fav_pos"] == 2) / len(fp),
                    "3": sum(1 for r in fp if r["fav_pos"] == 3) / len(fp),
                    "out": sum(1 for r in fp if r["fav_pos"] >= 4) / len(fp)}
        pats[p] = d
    doc = {"generated": str(date.today()), "n_races": len(recs),
           "win_patterns": WIN_PATS, "pace_rates": pace_rates, "patterns": pats,
           "note": "◎判定はデプロイ済みモデル（学習データ含む＝楽観側）。分岐比の診断用。"}
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n統計を書き出し: {out}  ({len(pats)}パターン / {len(recs)}レース)")


def main() -> None:
    ap = argparse.ArgumentParser(description="展開6パターンの発生率と紐構造")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    ap.add_argument("--men", action="store_true",
                    help="男子モデル・男子DBとして解析する（既定=ガールズ）")
    ap.add_argument("--apply", help="このdata.jsonの当日レースへ適用")
    ap.add_argument("--venue", default="", help="--apply時の会場フィルタ")
    ap.add_argument("--emit", help="本番参照用の統計JSONを書き出す先")
    args = ap.parse_args()
    recs = analyze(args.db, is_girls=not args.men)["recs"]
    report(recs)
    if args.emit:
        emit_stats(recs, Path(args.emit))
    if args.apply:
        apply_to(recs, args.apply, args.venue)


if __name__ == "__main__":
    main()
