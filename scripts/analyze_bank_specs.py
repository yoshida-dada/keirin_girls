"""バンク諸元（カント/みなし直線/幅員）と当日コンディションが決まり手を説明するかの検証。

**結論: いずれも不採用。** 同一周長内での説明力が無い。再検証を繰り返さないための記録。

  (1) バンク諸元 … 全場で見ると相関が出るが**周長との交絡**（カント vs 周長 r=-0.760）。
      同一周長(400m,31場)に絞ると全変数が事前基準|r|>0.5に届かない:
        カント -0.18 / みなし直線 +0.01 / 直線部傾斜 +0.10 / センター幅 -0.03
      唯一超えたホーム幅(+0.54)も四日市(13.3m、他は9.7〜11.4m)の1点由来で、
      除外すると+0.35へ低下＝外れ値アーティファクト。
      決定的な反例: 岐阜/四日市/和歌山はカント32.25°・直線59〜62mとほぼ同一なのに
      逃げ率が14.8%/29.5%/24.2%と2倍開く＝幾何形状では原理的に説明できない。

  (2) 当日コンディション … 同一レース内の上がりタイムで見ると逃げ率7.0%→33.8%と
      劇的だが**循環**（遅い展開だから逃げが残る＝結果同士の関係）。
      同じ日の**前のレース**から推定する as-of 検証（カバー率63%）に直すと
      逃げ率は21.7%→20.1%でほぼ横ばい＝予測には使えない。
      ただし捲り↔差しには単調な効果が残る（差し 27.0%→32.8%、捲り 51.3%→47.0%）。
      注意: ガールズは1会場1日あたり中央値2レースのため多くが「前1走」推定＝検出力は低い。
      「効果が無い」ではなく「この代理変数では使える強さを検出できない」が正確。

  → 会場差自体は実在する（400m内で逃げ率 9.4%〜29.5%、各n=67〜292で有意）。
     個別要因の特定は行き止まりなので、会場ごとの実測レートを直接使う方針へ
     （scripts/validate_venue_interaction.py）。

諸元の出典: keirin-brother.com/race-track/ （2026-07-29取得）。
みなし直線は Wikipedia「競輪場」と全場一致を確認（2ソース照合）。広島(57.9m)は
Wikipediaに記載が無く本ソースのみ。千葉(PIST6)はガールズ対象外。

  PYTHONIOENCODING=utf-8 python scripts/analyze_bank_specs.py --db data/keirin.sqlite
"""
from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR
from src.features import venue_meta as vm


def _dms(s: str) -> float:
    """32°15'07" → 32.2519（度）"""
    m = re.match(r"(\d+)°(\d+)['′´](\d+)", s)
    d, mi, se = (int(x) for x in m.groups())
    return d + mi / 60 + se / 3600


# 会場名 -> (周長, みなし直線m, センター部カント, 直線部傾斜, ホーム幅, バック幅, センター幅)
SPEC_RAW: dict[str, tuple] = {
    "函館": (400, 51.3, "30°36'51\"", "3°26'1\"", 10.8, 9.8, 7.8),
    "青森": (400, 58.9, "32°15'07\"", "2°51'45\"", 10.8, 9.8, 7.8),
    "いわき平": (400, 62.7, "32°54'45\"", "3°26'1\"", 10.0, 10.0, 7.3),
    "弥彦": (400, 63.1, "32°24'17\"", "2°51'45\"", 10.1, 9.0, 7.3),
    "前橋": (333, 46.7, "36°0'0\"", "4°0'0\"", 9.9, 9.9, 9.9),
    "取手": (400, 54.8, "31°30'25\"", "2°51'44\"", 10.0, 10.0, 7.5),
    "宇都宮": (500, 63.3, "25°47'44\"", "2°51'44\"", 10.3, 11.3, 8.3),
    "大宮": (500, 66.7, "26°16'40\"", "3°26'1\"", 10.3, 9.3, 7.5),
    "西武園": (400, 47.6, "29°26'54\"", "2°51'45\"", 11.0, 10.0, 7.5),
    "京王閣": (400, 51.5, "32°10'34\"", "2°51'44\"", 10.3, 9.0, 7.5),
    "立川": (400, 58.0, "31°13'6\"", "2°17'27\"", 9.7, 8.7, 7.7),
    "松戸": (333, 38.2, "29°44'42\"", "3°1'2\"", 11.1, 9.6, 8.1),
    "川崎": (400, 58.0, "32°10'14\"", "3°26'1\"", 10.3, 9.3, 8.3),
    "平塚": (400, 54.2, "31°28'37\"", "3°26'1\"", 11.0, 9.3, 7.5),
    "小田原": (333, 36.1, "35°34'12\"", "3°26'1\"", 11.3, 9.0, 7.5),
    "伊東温泉": (333, 46.6, "34°41'9\"", "3°26'1\"", 11.0, 9.3, 7.8),
    "静岡": (400, 56.4, "30°43'22\"", "2°51'45\"", 10.3, 9.3, 7.5),
    "名古屋": (400, 58.8, "34°1'47\"", "2°51'45\"", 10.3, 9.3, 7.3),
    "岐阜": (400, 59.3, "32°15'7\"", "2°51'45\"", 10.2, 9.0, 7.4),
    "大垣": (400, 56.0, "30°37'8\"", "2°51'45\"", 10.2, 9.0, 7.4),
    "豊橋": (400, 60.3, "33°50'22\"", "2°17'26\"", 10.3, 9.3, 7.8),
    "富山": (333, 43.0, "33°41'24\"", "3°26'1\"", 10.2, 9.2, 6.4),
    "松阪": (400, 61.5, "34°25'29\"", "2°51'45\"", 10.9, 9.0, 7.7),
    "四日市": (400, 62.4, "32°15'7\"", "2°51'45\"", 13.3, 11.5, 8.5),
    "福井": (400, 52.8, "31°28'37\"", "2°51'45\"", 10.5, 9.0, 7.5),
    "奈良": (333, 38.0, "33°25'47\"", "4°51'48\"", 10.8, 7.8, 7.8),
    "京都向日町": (400, 47.3, "30°29'7\"", "3°26'1\"", 10.3, 9.3, 7.6),
    "和歌山": (400, 59.9, "32°15'7\"", "2°51'45\"", 11.4, 9.3, 7.7),
    "岸和田": (400, 56.7, "30°56'0\"", "2°51'45\"", 10.2, 10.1, 7.3),
    "玉野": (400, 47.9, "30°37'33\"", "3°26'1\"", 10.3, 9.3, 7.5),
    "広島": (400, 57.9, "32°31'40\"", "3°26'1\"", 10.5, 8.5, 7.3),
    "防府": (333, 42.5, "34°41'9\"", "4°34'26\"", 10.2, 9.1, 7.4),
    "高松": (400, 54.8, "33°15'50\"", "2°51'45\"", 11.0, 9.0, 8.0),
    "小松島": (400, 55.5, "29°46'27\"", "2°51'45\"", 10.3, 9.3, 8.3),
    "高知": (500, 52.0, "24°29'51\"", "3°26'1\"", 11.3, 10.8, 7.8),
    "松山": (400, 58.6, "34°1'48\"", "2°51'45\"", 10.3, 9.3, 7.3),
    "小倉": (400, 56.9, "34°1'48\"", "3°26'1\"", 11.0, 10.0, 8.0),
    "久留米": (400, 50.7, "31°28'37\"", "3°26'1\"", 11.0, 10.0, 9.0),
    "武雄": (400, 64.4, "32°0'19\"", "2°17'26\"", 9.7, 8.7, 7.4),
    "佐世保": (400, 40.2, "31°28'37\"", "3°26'1\"", 10.0, 9.0, 7.5),
    "別府": (400, 59.9, "33°41'24\"", "2°51'45\"", 10.0, 9.0, 8.0),
    "熊本": (400, 60.3, "34°15'29\"", "2°51'45\"", 10.0, 9.0, 8.0),
}
SPEC = {k: {"bank": v[0], "straight": v[1], "cant": _dms(v[2]), "cant_str": _dms(v[3]),
            "w_home": v[4], "w_back": v[5], "w_center": v[6]} for k, v in SPEC_RAW.items()}


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return float("nan") if sx == 0 or sy == 0 else \
        sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _spearman(xs, ys):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for p, i in enumerate(s):
            r[i] = p
        return r
    return _pearson(rk(xs), rk(ys))


def _load(db: str):
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    c.execute("PRAGMA query_only=1")
    meta = {rid: (d, v, rno) for rid, d, v, rno
            in c.execute("SELECT race_id,race_date,venue_code,race_no FROM races")}
    win = {rid: (kim, lap) for rid, kim, lap
           in c.execute("SELECT race_id,kimarite,last_lap FROM results WHERE position=1")}
    c.close()
    return meta, win


def specs_report(meta, win) -> None:
    agg = defaultdict(lambda: {"n": 0, "esc": 0, "sas": 0, "mak": 0})
    for rid, (d, v, rno) in meta.items():
        w = win.get(rid)
        if not w:
            continue
        a = agg[v]; a["n"] += 1
        if w[0] == "逃": a["esc"] += 1
        elif w[0] == "差": a["sas"] += 1
        elif w[0] == "捲": a["mak"] += 1
    rows = []
    for v, a in agg.items():
        if a["n"] < 60:
            continue
        sp = SPEC.get(vm.venue_name(v) or "")
        if not sp:
            continue
        rows.append({**sp, "name": vm.venue_name(v), "n": a["n"],
                     "esc": a["esc"] / a["n"] * 100, "sas": a["sas"] / a["n"] * 100,
                     "mak": a["mak"] / a["n"] * 100})
    print(f"【(1) バンク諸元】対象 {len(rows)}場（n>=60）")
    VARS = [("cant", "カント"), ("straight", "みなし直線"), ("cant_str", "直線部傾斜"),
            ("w_home", "ホーム幅"), ("w_center", "センター幅")]
    for label, sub in [("全場", rows), ("400mのみ", [r for r in rows if r["bank"] == 400])]:
        if len(sub) < 6:
            continue
        print(f"\n  [{label}] {len(sub)}場  (Pearson/Spearman)")
        print(f"    {'変数':<12}{'逃げ率':>18}{'差し率':>18}{'捲り率':>18}")
        for k, jp in VARS:
            xs = [r[k] for r in sub]
            cells = [f"{_pearson(xs,[r[m] for r in sub]):+.2f}/{_spearman(xs,[r[m] for r in sub]):+.2f}"
                     for m in ("esc", "sas", "mak")]
            print(f"    {jp:<12}" + "".join(f"{c:>18}" for c in cells))
    xs = [r["cant"] for r in rows]
    print(f"\n  交絡の確認: カント vs 周長 Pearson {_pearson(xs,[r['bank'] for r in rows]):+.3f}"
          f" / 周長 vs 逃げ率 {_pearson([r['bank'] for r in rows],[r['esc'] for r in rows]):+.3f}")
    s4 = [r for r in rows if r["bank"] == 400]
    if s4:
        no4 = [r for r in s4 if r["name"] != "四日市"]
        print(f"  ホーム幅 vs 逃げ率: 全{len(s4)}場 {_pearson([r['w_home'] for r in s4],[r['esc'] for r in s4]):+.3f}"
              f" → 四日市除外 {_pearson([r['w_home'] for r in no4],[r['esc'] for r in no4]):+.3f}（外れ値由来）")


def conditions_report(meta, win) -> None:
    """当日コンディション（前走の上がりタイム）→ その後のレースの決まり手。as-of。"""
    byv = defaultdict(list)
    for rid, (d, v, rno) in meta.items():
        w = win.get(rid)
        if w and w[1]:
            byv[v].append(w[1])
    vmean = {v: sum(x) / len(x) for v, x in byv.items() if len(x) >= 30}
    day = defaultdict(list)
    for rid, (d, v, rno) in meta.items():
        w = win.get(rid)
        if w and w[1] and v in vmean:
            day[(d, v)].append((rno or 0, w[0], w[1]))

    buck = defaultdict(lambda: {"n": 0, "esc": 0, "sas": 0, "mak": 0})
    used = first = 0
    for (d, v), lst in day.items():
        lst.sort()
        prior = []
        for rno, kim, lap in lst:
            if prior:
                z = sum(prior) / len(prior) - vmean[v]
                b = ("速い(-0.3s超)" if z < -0.3 else "やや速い" if z < -0.1 else
                     "標準" if z < 0.1 else "やや遅い" if z < 0.3 else "遅い(+0.3s超)")
                a = buck[b]; a["n"] += 1
                if kim == "逃": a["esc"] += 1
                elif kim == "差": a["sas"] += 1
                elif kim == "捲": a["mak"] += 1
                used += 1
            else:
                first += 1
            prior.append(lap)
    print(f"\n\n【(2) 当日コンディション（as-of）】判定可 {used}R / 当日1走目 {first}R"
          f"（カバー率 {used/(used+first):.0%}）")
    print(f"  {'当日これまでの上がり':<18}{'n':>7}{'逃':>8}{'捲':>8}{'差':>8}")
    for b in ("速い(-0.3s超)", "やや速い", "標準", "やや遅い", "遅い(+0.3s超)"):
        a = buck[b]
        if a["n"] < 50:
            continue
        print(f"  {b:<18}{a['n']:>7}{a['esc']/a['n']*100:>7.1f}%"
              f"{a['mak']/a['n']*100:>7.1f}%{a['sas']/a['n']*100:>7.1f}%")
    print("  → 逃げはほぼ横ばい＝予測に使えない。捲り↔差しには弱い単調効果（約5pt）。")
    print("  → 参考: 同一レース内の上がりで見ると逃げ率7.0%→33.8%と出るが、これは循環。")


def main() -> None:
    ap = argparse.ArgumentParser(description="バンク諸元・当日条件の説明力検証（結論: 不採用）")
    ap.add_argument("--db", default=str(DATA_DIR / "keirin.sqlite"))
    args = ap.parse_args()
    meta, win = _load(args.db)
    specs_report(meta, win)
    conditions_report(meta, win)


if __name__ == "__main__":
    main()
