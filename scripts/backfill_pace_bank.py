"""既存 dashboard/data.json の development.pace を「ペース×バンク」の実測値へ更新する。

再予測もネットワークもDB接続も不要（会場名→venue_code→バンク長 と、保存済みの n_front から
引き直すだけ）ので、確定済みレースにも安全に適用できる。

  PYTHONIOENCODING=utf-8 python scripts/backfill_pace_bank.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import venue_meta as vm
from src.model.kimarite_hint import hint as khint, pace_note as knote


def _code_by_name() -> dict[str, str]:
    out = {}
    for code in vm.VENUE:
        nm = vm.venue_name(code)
        if nm:
            out[nm] = code
    return out


def _resolve_code(vname: str, name2code: dict[str, str]) -> str | None:
    """会場名 → venue_code。venue_code を持たない旧レコード用のフォールバック。

    表記が食い違う（data.json「伊東競輪」 vs テーブル「伊東温泉」）ため、
    「競輪(場)」を落としたうえで双方向の前方一致を許し、複数該当は最長一致を採る。
    """
    v = str(vname or "").replace("競輪場", "").replace("競輪", "").strip()
    if len(v) < 2:
        return None
    if v in name2code:
        return name2code[v]
    cands = [(n, c) for n, c in name2code.items() if n.startswith(v) or v.startswith(n)]
    if not cands:
        return None
    cands.sort(key=lambda nc: -len(nc[0]))
    if len(cands) > 1:
        print(f"  ※ 会場名 {vname!r} が複数該当 {[n for n, _ in cands]} → 最長一致 {cands[0][0]!r} を採用")
    return cands[0][1]


def main() -> None:
    ap = argparse.ArgumentParser(description="data.json のペース読みをバンク対応へ更新")
    ap.add_argument("--path", default=str(ROOT / "dashboard" / "data.json"))
    args = ap.parse_args()

    p = Path(args.path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    races = (doc.get("predictions") or {}).get("races") or []
    name2code = _code_by_name()
    done = skipped = 0
    for r in races:
        dev = r.get("development") or {}
        pace = dev.get("pace") or {}
        nf = pace.get("n_front")
        if nf is None:
            skipped += 1
            continue
        # 新しいレコードは venue_code を持つ。無ければ会場名から解決する。
        code = r.get("venue_code") or _resolve_code(r.get("venue") or "", name2code)
        h = khint(nf, code or "")
        if not h:
            skipped += 1
            continue
        pace["kimarite_hint"] = h["kimarite_hint"]
        if h.get("b_reliability"):
            pace["b_reliability"] = h["b_reliability"]
        pace["basis"] = h.get("basis")
        pace["bank"] = h.get("bank")
        pace["note"] = knote(nf, h["kimarite_hint"])
        dev["pace"] = pace
        r["development"] = dev
        done += 1
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"更新 {done}レース / 未更新 {skipped}レース → {p}")
    for r in races[:4]:
        pc = ((r.get("development") or {}).get("pace") or {})
        k = pc.get("kimarite_hint") or {}
        print(f"  {r.get('venue')} R{r.get('race_no')} 先行型{pc.get('n_front')} "
              f"{pc.get('bank')}m 逃{k.get('逃')}%/捲{k.get('捲')}%/差{k.get('差')}%  {pc.get('basis')}")


if __name__ == "__main__":
    main()
