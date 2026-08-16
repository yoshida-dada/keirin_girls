"""選手のライン位置別成績・戦法系実績の集計テスト。"""
import sqlite3

import pytest

from src.features.rider_line_stats import compute_line_stats, MIN_N


def _db(tmp_path, rows):
    """rows = [(race_id, date, car, line_id, pos_in_line, leg, seri, 着順, sb, 名前)]"""
    p = tmp_path / "t.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
      CREATE TABLE races(race_id TEXT PRIMARY KEY, race_date TEXT);
      CREATE TABLE narabi(race_id TEXT, car_number INT, line_id INT, pos_in_line INT,
                          leg TEXT, seri_group INT);
      CREATE TABLE results(race_id TEXT, position INT, car_number INT, sb TEXT);
      CREATE TABLE entries(race_id TEXT, car_number INT, rider_name TEXT);
    """)
    for rid, d, car, li, pi, leg, sg, pos, sb, nm in rows:
        c.execute("INSERT OR IGNORE INTO races VALUES(?,?)", (rid, d))
        c.execute("INSERT INTO narabi VALUES(?,?,?,?,?,?)", (rid, car, li, pi, leg, sg))
        c.execute("INSERT INTO results VALUES(?,?,?,?)", (rid, pos, car, sb))
        c.execute("INSERT INTO entries VALUES(?,?,?)", (rid, car, nm))
    c.commit(); c.close()
    compute_line_stats.cache_clear()
    return str(p)


def test_position_labels_and_counts(tmp_path):
    """先頭/番手/3番手以降/単騎 を人数で判定し、着順を集計する。"""
    rows = []
    for i in range(6):                       # 同じ構成を6回
        rid = f"r{i}"
        rows += [(rid, "2026-01-01", 1, 0, 0, "先行", None, 1, "B", "A"),
                 (rid, "2026-01-01", 2, 0, 1, "追込", None, 2, "", "B"),
                 (rid, "2026-01-01", 3, 0, 2, "追込", None, 5, "", "C"),
                 (rid, "2026-01-01", 4, 1, 0, "自在", None, 3, "", "D")]  # 1人ライン=単騎
    s = compute_line_stats(_db(tmp_path, rows), None)
    assert s["A"]["pos"]["先頭"]["n"] == 6
    assert s["A"]["pos"]["先頭"]["win"] == 1.0
    assert s["B"]["pos"]["番手"]["top3"] == 1.0
    assert s["C"]["pos"]["3番手以降"]["win"] == 0.0
    assert "単騎" in s["D"]["pos"]          # 1人ラインは単騎に入る


def test_thin_samples_return_none_rate(tmp_path):
    """走数が MIN_N 未満なら率を出さない（分母は返す）。数字を出すと信じられるため。"""
    rows = [("r0", "2026-01-01", 1, 0, 0, "先行", None, 1, "B", "A"),
            ("r0", "2026-01-01", 2, 0, 1, "追込", None, 2, "", "B")]
    s = compute_line_stats(_db(tmp_path, rows), None)
    assert s["A"]["pos"]["先頭"]["n"] == 1
    assert s["A"]["pos"]["先頭"]["win"] is None      # n=1 < MIN_N
    assert MIN_N >= 2


def test_as_of_excludes_later_races(tmp_path):
    """as_of より後のレースは集計しない（表示でも実績は as-of で出す）。"""
    # 先頭と判定させるにはラインが2名以上要る（1名だと単騎になる）
    rows = [("r0", "2026-01-01", 1, 0, 0, "先行", None, 1, "B", "A"),
            ("r0", "2026-01-01", 2, 0, 1, "追込", None, 2, "", "B"),
            ("r1", "2026-06-01", 1, 0, 0, "先行", None, 1, "B", "A"),
            ("r1", "2026-06-01", 2, 0, 1, "追込", None, 2, "", "B")]
    db = _db(tmp_path, rows)
    assert compute_line_stats(db, None)["A"]["pos"]["先頭"]["n"] == 2
    compute_line_stats.cache_clear()
    assert compute_line_stats(db, "2026-03-01")["A"]["pos"]["先頭"]["n"] == 1


def test_kamashi_and_tsuppari(tmp_path):
    """かまし=カマシ宣言でB取得。つっぱり=カマシ宣言者が居る中で先行がBを守った。"""
    rows = [("r0", "2026-01-01", 1, 0, 0, "先行", None, 1, "B", "A"),
            ("r0", "2026-01-01", 2, 1, 0, "カマシ", None, 4, "", "K")]
    s = compute_line_stats(_db(tmp_path, rows), None)
    assert s["K"]["kamashi"]["n"] == 1 and s["K"]["kamashi"]["k"] == 0
    assert s["A"]["tsuppari"]["n"] == 1 and s["A"]["tsuppari"]["k"] == 1


def test_seri_win_needs_rival_in_results(tmp_path):
    """競りは相手より上位で入れば勝ち。相手の着順が無ければ数えない。"""
    rows = [("r0", "2026-01-01", 7, 0, 1, "競り", 0, 2, "", "X"),
            ("r0", "2026-01-01", 1, 0, 2, "競り", 0, 5, "", "Y"),
            ("r0", "2026-01-01", 2, 0, 0, "先行", None, 1, "B", "Z")]
    s = compute_line_stats(_db(tmp_path, rows), None)
    assert s["X"]["seri"]["n"] == 1 and s["X"]["seri"]["k"] == 1
    assert s["Y"]["seri"]["n"] == 1 and s["Y"]["seri"]["k"] == 0


def test_line_cut_counts_same_event_from_both_sides(tmp_path):
    """ライン分断は先頭視点と番手視点で同じ事象を数える（率が一致するのはそのため）。"""
    rows = [("r0", "2026-01-01", 1, 0, 0, "先行", None, 1, "B", "A"),
            ("r0", "2026-01-01", 2, 0, 1, "追込", None, 7, "", "B")]
    s = compute_line_stats(_db(tmp_path, rows), None)
    assert s["A"]["cut_head"]["k"] == 1      # 先頭3着内・番手着外
    assert s["B"]["cut_mate"]["k"] == 1      # 同じ事象を番手側から


def test_missing_position_is_skipped(tmp_path):
    """失格・欠車で着順が無い行は集計に入れない（着順 None で落ちない）。"""
    rows = [("r0", "2026-01-01", 1, 0, 0, "先行", None, None, "", "A")]
    s = compute_line_stats(_db(tmp_path, rows), None)
    assert "A" not in s
