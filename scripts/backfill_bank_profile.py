"""既存 dashboard/data.json の各レースに bank_profile（バンク諸元＋有利脚質）を後付けする。

venue_code（無ければ会場名から解決）だけで組み立てるため、再予測もネットワークもDBも不要。
確定済みレースにも安全に適用できる（result・予測はそのまま保持）。

  PYTHONIOENCODING=utf-8 python scripts/backfill_bank_profile.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.bank_profile import profile
from src.features import venue_meta as vm


def _resolve_code(vname: str) -> str | None:
    """会場名 → venue_code。venue_code を持たない旧レコード用。
    表記差（data.json「伊東競輪」 vs テーブル「伊東温泉」）に備え双方向の前方一致。"""
    v = str(vname or "").replace("競輪場", "").replace("競輪", "").strip()
    if len(v) < 2:
        return None
    name2code = {vm.venue_name(c): c for c in vm.VENUE if vm.venue_name(c)}
    if v in name2code:
        return name2code[v]
    cands = [(n, c) for n, c in name2code.items() if n.startswith(v) or v.startswith(n)]
    if not cands:
        return None
    cands.sort(key=lambda nc: -len(nc[0]))
    return cands[0][1]


def main() -> None:
    ap = argparse.ArgumentParser(description="data.json へバンク特性を後付け")
    ap.add_argument("--path", default=str(ROOT / "dashboard" / "data.json"))
    args = ap.parse_args()

    p = Path(args.path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    races = (doc.get("predictions") or {}).get("races") or []
    done = skipped = 0
    for r in races:
        code = r.get("venue_code") or _resolve_code(r.get("venue") or "")
        prof = profile(code)
        if prof:
            r["bank_profile"] = prof
            if code and not r.get("venue_code"):
                r["venue_code"] = code          # 以後の照合を名前解決に頼らない
            done += 1
        else:
            skipped += 1
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"付与 {done}レース / 未付与 {skipped}レース → {p}")
    seen = set()
    for r in races:
        bp = r.get("bank_profile")
        if not bp or bp["venue"] in seen:
            continue
        seen.add(bp["venue"])
        a = (bp.get("advantage") or {}).get("text", "")
        print(f"  {bp['venue']:<8}{bp['bank']}m 直線{bp['straight']}m カント{bp['cant']}°  {a}")


if __name__ == "__main__":
    main()
