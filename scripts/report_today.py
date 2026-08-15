"""確定済みレースの実績を集計する（◎の勝率/複勝率、展開分岐と買い目の的中率/回収率）。

ダッシュボードに出している予測をそのまま採点する。**予測は発走前に出したものが
据え置かれている**（build_predictions は確定済みレースを辞書ごと保存する）ので、
後から良く見えるように書き換わってはいない。

**買い目は推奨ではない。** 回収率100%超のゾーンは存在しないと検証済み
（men_keirin_plan.md 4.15 / 4.21.1）。ここで出すのは「表示している買い目が
その日どう転んだか」の事後集計であって、期待値の主張ではない。

  PYTHONIOENCODING=utf-8 python scripts/report_today.py
  python scripts/report_today.py --date 2026-08-14
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "dashboard" / "data_men.json"
STAKE = 100


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二項比率の95%信頼区間（Wilson）。n が小さいので正規近似は使わない。"""
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - s) / d, (c + s) / d)


def pct(k: int, n: int) -> str:
    if not n:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k/n*100:>5.1f}% ({k}/{n}) [{lo*100:.0f}–{hi*100:.0f}%]"


def boot_roi(rows: list[tuple], n_boot: int = 4000, seed: int = 0):
    """rows=[(賭け金, 払戻)]。レース単位でリサンプルした95%区間。"""
    if not rows:
        return None, None, None
    rnd = random.Random(seed)
    s = sum(a for a, _ in rows)
    r = sum(b for _, b in rows)
    point = r / s if s else 0.0
    n = len(rows)
    vals = []
    for _ in range(n_boot):
        ss = rr = 0.0
        for _ in range(n):
            a, b = rows[rnd.randrange(n)]
            ss += a; rr += b
        if ss:
            vals.append(rr / ss)
    vals.sort()
    return point, vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def hits(form: dict, combo: tuple) -> bool:
    """買い目（1着候補×2着候補×3着候補）が的中したか。"""
    if not form:
        return False
    a, b, c = combo
    return (a in (form.get("first") or []) and b in (form.get("second") or [])
            and c in (form.get("third") or []))


def main() -> None:
    ap = argparse.ArgumentParser(description="確定済みレースの実績集計")
    ap.add_argument("--path", default=str(DEFAULT))
    ap.add_argument("--date", help="対象日（既定=データの当日）")
    args = ap.parse_args()

    doc = json.loads(Path(args.path).read_text(encoding="utf-8"))
    P = doc.get("predictions") or {}
    day = args.date or P.get("date")
    races = [r for r in (P.get("races") or [])
             if r.get("date") == day and (r.get("result") or {}).get("order")]
    print(f"対象: {day} の確定済み {len(races)}レース\n")
    if not races:
        return

    # ---------- A. ◎（1着確率トップ）----------
    w = q2 = q3 = 0
    for r in races:
        rd = (r.get("riders") or [{}])[0]
        order = [o["car"] for o in r["result"]["order"]]
        if rd.get("car") is None or not order:
            continue
        c = rd["car"]
        w += int(order[0] == c)
        q2 += int(c in order[:2])
        q3 += int(c in order[:3])
    n = len(races)
    # モデルが事前に言っていた期待値。実績だけ見ても「良かった/悪かった」は判断できない
    import sys
    sys.path.insert(0, str(ROOT))
    from src.model.plackett_luce import all_trifecta_probs
    e1 = e2 = e3 = 0.0
    for r in races:
        st = {x["car"]: x.get("win_prob") or 0.0
              for x in (r.get("riders") or []) if x.get("car") is not None}
        rd = (r.get("riders") or [{}])[0]
        c = rd.get("car")
        if not st or c is None:
            continue
        e1 += st.get(c, 0.0)
        pr = all_trifecta_probs(st)
        e2 += sum(p for k, p in pr.items() if c in k[:2])
        e3 += sum(p for k, p in pr.items() if c in k)
    print("=== ◎（モデルの1着確率トップ）===")
    print(f"{'':8}{'実績':<28}{'モデルの事前期待':>18}")
    print(f"  勝率   {pct(w, n):<28}{e1/n*100:>16.1f}%")
    print(f"  連対率 {pct(q2, n):<28}{e2/n*100:>16.1f}%")
    print(f"  複勝率 {pct(q3, n):<28}{e3/n*100:>16.1f}%")

    # ---------- B. 展開分岐 ----------
    # 主導権(B)は結果の sb 欄に 'B' が付く。並びの「主導権を取るライン」の予測を採点する。
    nb = bhit = 0
    wb = whit = 0
    for r in races:
        br = ((r.get("dev_branches") or {}).get("branches") or [])
        if not br:
            continue
        top = br[0]
        bcar = next((o["car"] for o in r["result"]["order"] if "B" in (o.get("sb") or "")), None)
        if bcar is not None:
            nb += 1
            # 分岐は「そのラインが主導権」。ライン内の誰かがBを取れば的中とする
            bhit += int(bcar in (top.get("line") or [top.get("b_car")]))
        # 分岐内の勝者予測（win 分布の最頻）
        win = top.get("win") or {}
        if win:
            wb += 1
            pick = max(win, key=lambda k: win[k])
            whit += int(int(pick) == r["result"]["order"][0]["car"])
    print("\n=== 展開分岐 ===")
    print(f"  主導権ラインの的中（最有力分岐） {pct(bhit, nb)}")
    print(f"  最有力分岐の1着予想の的中       {pct(whit, wb)}")

    # ---------- C. 買い目 ----------
    print("\n=== 買い目（表示している形をそのまま採点。推奨ではない）===")
    strat: dict[str, list] = {}
    for r in races:
        pay = (r.get("result") or {}).get("payout") or {}
        if not pay.get("combo") or pay.get("yen") is None:
            continue
        combo = tuple(int(x) for x in str(pay["combo"]).split("-"))
        yen = pay["yen"]
        br = ((r.get("dev_branches") or {}).get("branches") or [])
        if not br:
            continue

        def add(name, form):
            if not form or not form.get("points"):
                return
            strat.setdefault(name, []).append(
                (STAKE * form["points"], yen if hits(form, combo) else 0))

        add("最有力分岐の買い目", br[0].get("formation"))
        for ft in (br[0].get("form_types") or []):
            add(f"{ft['kind']}（最有力分岐）", ft.get("formation"))

        # 全分岐の買い目を足し合わせた場合（点数も合算）
        pts, hit = 0, 0
        for b in br:
            f = b.get("formation") or {}
            if f.get("points"):
                pts += f["points"]
                hit = max(hit, yen if hits(f, combo) else 0)
        if pts:
            strat.setdefault("全分岐の買い目を合算", []).append((STAKE * pts, hit))

        # 3つの型を全部買う
        pts, hit = 0, 0
        for ft in (br[0].get("form_types") or []):
            f = ft.get("formation") or {}
            if f.get("points"):
                pts += f["points"]
                hit = max(hit, yen if hits(f, combo) else 0)
        if pts:
            strat.setdefault("◎頭/◎2着/◎抜きを全部", []).append((STAKE * pts, hit))

    print(f"{'買い目':<24}{'R数':>5}{'的中':>5}{'平均点数':>9}{'回収率':>9}{'95%区間':>18}")
    for name, rows in strat.items():
        k = sum(1 for _, r in rows if r > 0)
        p, lo, hi = boot_roi(rows)
        avg = sum(a for a, _ in rows) / len(rows) / STAKE
        print(f"{name:<24}{len(rows):>5}{k:>5}{avg:>8.1f}点{p*100:>8.1f}%"
              f"{f'[{lo*100:.0f}–{hi*100:.0f}%]':>18}")

    # 100%を超えた戦略は、的中の配当がどれだけ偏っているかを必ず見る。
    # 少数の高配当で持ち上がっているだけなら再現しない（過去に同じ罠にはまっている）
    print("\n--- 回収率100%超の戦略の中身（配当の偏りを見る）---")
    for name, rows in strat.items():
        p, _, _ = boot_roi(rows)
        if p is None or p <= 1.0:
            continue
        pays = sorted((r for _, r in rows if r > 0), reverse=True)
        tot = sum(pays)
        stake = sum(a for a, _ in rows)
        top1 = pays[0] / stake if stake else 0
        print(f"  {name}: 的中配当 {[f'{x:,}' for x in pays]}")
        print(f"    最高配当1本だけで回収率 {top1*100:.1f}pt 分。"
              f"これを除くと {((tot-pays[0])/stake)*100:.1f}%")

    print("\n※ n が小さいので区間は広い。控除率25%のため回収率の上限は75%。"
          "\n※ 買い目は展開の読みの材料であって推奨ではない（黒字ゾーンは存在しないと検証済み）。")


if __name__ == "__main__":
    main()
