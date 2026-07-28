"""既存 dashboard/data.json の各レースに dev_patterns（展開6パターン上位3）を後付けする。

再予測もネットワークもDBも不要（top1_prob / development.pace / riders から組み立てるだけ）ので、
**すでに確定した過去レースにも安全に適用できる**（result はそのまま保持）。

  PYTHONIOENCODING=utf-8 python scripts/backfill_dev_patterns.py
  PYTHONIOENCODING=utf-8 python scripts/backfill_dev_patterns.py --path dashboard/data.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.model.dev_patterns import build_dev_patterns


def main() -> None:
    ap = argparse.ArgumentParser(description="data.json へ展開パターンを後付け")
    ap.add_argument("--path", default=str(ROOT / "dashboard" / "data.json"))
    args = ap.parse_args()

    p = Path(args.path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    races = (doc.get("predictions") or {}).get("races") or []
    done = skipped = 0
    for r in races:
        pace = ((r.get("development") or {}).get("pace") or {}).get("level", "")
        dp = build_dev_patterns(r.get("top1_prob"), pace, r.get("riders") or [])
        if dp:
            r["dev_patterns"] = dp
            done += 1
        else:
            skipped += 1
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"付与 {done}レース / 未付与 {skipped}レース → {p}")
    for r in races[:3]:
        dp = r.get("dev_patterns")
        if not dp:
            continue
        top = " / ".join(f"{t['key']} {t['prob']*100:.1f}%" for t in dp["top"])
        print(f"  {r.get('venue')} R{r.get('race_no')}: {top}")


if __name__ == "__main__":
    main()
