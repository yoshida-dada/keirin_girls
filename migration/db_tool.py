"""banei.sqlite3 の移行補助: 整合バックアップと件数照合。

  python migration/db_tool.py backup <src.sqlite3> <dst.sqlite3> <manifest.json>
      稼働中でも一貫したスナップショットを取る（sqlite3 online backup API）。
      コピー後に PRAGMA integrity_check と全テーブル件数を取り manifest.json に保存する。
  python migration/db_tool.py check <db.sqlite3> <manifest.json>
      新PC側で DB を検証する（integrity_check + 件数がmanifest以上/一致か）。

標準ライブラリのみ。終了コード 0=OK / 1=NG。
"""
import json
import sqlite3
import sys
import time
from pathlib import Path


def _counts(con: sqlite3.Connection) -> dict:
    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    return {t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}


def backup(src: str, dst: str, manifest: str) -> int:
    dst_p = Path(dst)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    if dst_p.exists():
        dst_p.unlink()
    t0 = time.time()
    s = sqlite3.connect(src, timeout=60)
    d = sqlite3.connect(dst)
    with d:
        s.backup(d, pages=2000, sleep=0.01)
    s.close()
    ok = d.execute("PRAGMA integrity_check").fetchone()[0]
    counts = _counts(d)
    d.close()
    Path(manifest).write_text(json.dumps(
        {"integrity_check": ok, "counts": counts, "bytes": dst_p.stat().st_size,
         "created": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"backup done in {time.time() - t0:.0f}s: integrity={ok} tables={len(counts)} "
          f"rows={sum(counts.values()):,} size={dst_p.stat().st_size / 1e6:.0f}MB")
    return 0 if ok == "ok" else 1


def check(db: str, manifest: str) -> int:
    m = json.loads(Path(manifest).read_text(encoding="utf-8-sig"))
    con = sqlite3.connect(f"file:{Path(db).as_posix()}?mode=ro", uri=True)
    ok = con.execute("PRAGMA integrity_check").fetchone()[0]
    counts = _counts(con)
    con.close()
    bad = []
    if ok != "ok":
        bad.append(f"integrity_check={ok}")
    for t, n in m["counts"].items():
        if counts.get(t) is None:
            bad.append(f"missing table {t}")
        elif counts[t] < n:  # 新PCで収集が進めば増えるのは正常。減っていたらNG
            bad.append(f"{t}: {counts[t]} < manifest {n}")
    print(f"integrity={ok} tables={len(counts)} rows={sum(counts.values()):,}")
    for b in bad:
        print("NG:", b)
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "backup":
        sys.exit(backup(*sys.argv[2:5]))
    if len(sys.argv) == 4 and sys.argv[1] == "check":
        sys.exit(check(*sys.argv[2:4]))
    print(__doc__)
    sys.exit(2)
