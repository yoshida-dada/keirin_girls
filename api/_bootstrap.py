"""プロジェクトルートを sys.path へ追加する共通ブートストラップ（他スクリプトと同方式）。

`api/` 配下の各モジュールが `src.*` / `config.*` / `scripts.*` を import できるようにする。
絶対パス埋め込みは禁止のため pathlib で相対解決する。
"""
from __future__ import annotations

import sys
from pathlib import Path

# api/ の1つ上 = プロジェクトルート（KEIRIN/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
