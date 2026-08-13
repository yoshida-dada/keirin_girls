"""決まり手の目安と主導権(B)の信頼度を「ペース×バンク」の実測から引く（表示用）。

背景: 従来はペース区分だけの固定値だったが、実測では**バンク長の方が3〜4倍強い**:
  逃げ率  333m 35.3% / 400m 19.3% / 500m 12.8%（差 22.5pt）
  ペース  スロー23.5% → ハイ16.7%（差 6.8pt）
主導権の信頼度も同様（B連対 333m 72% / 500m 51%）。バンクを無視すると短走路で
逃げを過小評価し、長走路で過大評価する。

参照順は (ペース×バンク) → バンク → ペース → 全体。標本が min_n 未満のセルは
上位へフォールバックする（例: ハイ×500m は n=47 なのでバンク500mの値を使う）。
統計は `kimarite_stats.json`（scripts/analyze_gear_bank.py --emit が生成）。
**DBに触らない**ので refresh_predictions.py（DB非依存）からも安全に呼べる。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.features import venue_meta as vm

# 統計の所在は stats_profile.py が持つ（男女で別ファイル）。


@lru_cache(maxsize=1)
def _stats(is_girls: bool = True) -> dict:
    from src.model.stats_profile import profile as _sp
    path = _sp(is_girls).kimarite_stats
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def pace_key(n_front: int) -> str:
    """先行型人数（相対定義）→ 履歴のペース区分。analyze_gear_bank と同一。"""
    return "スロー" if n_front <= 2 else ("ハイ" if n_front >= 4 else "ミドル")


def _usable(cell: dict | None, min_n: int) -> bool:
    return bool(cell and cell.get("kim") and cell.get("n", 0) >= min_n)


def hint(n_front: int, venue_code: str, is_girls: bool = True) -> dict | None:
    """{kimarite_hint, b_reliability, basis, bank} を返す。統計が無ければ None。"""
    st = _stats(is_girls)
    if not st:
        return None
    min_n = st.get("min_n", 60)
    pace = pace_key(n_front)
    bank = vm.bank_length(venue_code or "")

    # 細かい順に試し、標本が足りるところで採用する
    cands = []
    if bank:
        cands.append((f"ペース({pace})×バンク{bank}m", (st.get("cells") or {}).get(f"{pace}|{bank}")))
        cands.append((f"バンク{bank}m", (st.get("bank") or {}).get(str(bank))))
    cands.append((f"ペース({pace})", (st.get("pace") or {}).get(pace)))
    cands.append(("全体", st.get("global")))
    for label, cell in cands:
        if _usable(cell, min_n if label != "全体" else 0):
            k = cell["kim"]
            out = {"kimarite_hint": {"逃": round(k["逃"]), "捲": round(k["捲"]), "差": round(k["差"])},
                   "basis": f"{label} 実測n={cell['n']}", "bank": bank}
            if cell.get("b_rentai") is not None:
                out["b_reliability"] = {"rentai": cell["b_rentai"], "gaiji": cell["b_gaiji"],
                                        "note": _b_note(cell["b_rentai"], cell["b_gaiji"])}
            return out
    return None


def _b_note(rentai: int, gaiji: int) -> str:
    if gaiji >= 35:
        return "主導権を取っても垂れやすい（差し・捲りを評価）"
    if gaiji >= 25:
        return "主導権はやや不安定"
    return "主導権はそのまま残りやすい"


def pace_note(n_front: int, kim: dict) -> str:
    """決まり手の実測構成から一言。バンクで逃げ残りやすさが大きく変わるため数値基準で書く。"""
    esc = kim.get("逃", 0)
    if esc >= 30:
        return "短走路で逃げが残りやすい（先行有利）"
    if esc <= 15:
        return "逃げは残りにくく差し・捲りが台頭"
    return "逃げが飛びやすく捲り・差しが優勢" if n_front >= 4 else "捲り・差しがやや優勢"
