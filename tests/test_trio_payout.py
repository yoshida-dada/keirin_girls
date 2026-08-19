"""三連複払戻のパーステスト。

払戻テーブルは1行に全券種が並び、`a=b`（枠連/車連/ワイド）と `a=b=c`（三連複）が
混在する。**イコール2個の組を拾わない**ことが要点（ワイドは1セルに3組入る）。
"""
from src.collect.gamboo_result import (parse_trio_payout, parse_trifecta_payout,
                                        parse_exacta_payout)

# 実物（2026-08-16 松戸1R）の払戻テーブルを再現したもの
HTML = """
<table class="refund_table"><tbody>
<tr><th>2 枠 連</th><td>複</td><td>1=4 190円 (1)</td>
    <th>2 車 連</th><td>複</td><td>1=5 180円 (1)</td>
    <th>3 連 勝</th><td>複</td><td>1=5=7 410円 (2)</td>
    <th>ワ イ ド</th><td>1=5 140円 (1) 1=7 390円 (7) 5=7 330円 (4)</td></tr>
<tr><td>単</td><td>4-1 310円 (1)</td><td>単</td><td>5-1 250円 (1)</td>
    <td>単</td><td>5-1-7 720円 (1)</td></tr>
</tbody></table>
"""


def test_parses_trio_not_wide_or_quinella():
    """三連複だけを拾う（ワイド・車連・枠連の a=b は拾わない）。"""
    got = parse_trio_payout(HTML)
    assert got is not None
    assert got.combo == (1, 5, 7)          # 車番昇順
    assert got.payout == 410
    assert got.popularity == 2


def test_trifecta_still_parsed():
    """既存の三連単パースを壊していない（a-b-c 側）。"""
    got = parse_trifecta_payout(HTML)
    assert got.combo == (5, 1, 7)
    assert got.payout == 720


def test_no_table_returns_none():
    assert parse_trio_payout("<html><body>払戻なし</body></html>") is None


def test_combo_is_sorted():
    """並び順が違っても車番昇順に正規化する（モデル側の三連複キーと突き合わせるため）。"""
    html = ('<table class="refund_table"><tr><td>3 連 勝 複</td>'
            '<td>7=1=5 410円 (2)</td></tr></table>')
    assert parse_trio_payout(html).combo == (1, 5, 7)


def test_popularity_optional():
    html = ('<table class="refund_table"><tr><td>3 連 勝 複</td>'
            '<td>2=3=4 1,250円</td></tr></table>')
    got = parse_trio_payout(html)
    assert got.combo == (2, 3, 4) and got.payout == 1250 and got.popularity is None


# 実物の払戻テーブルは dl/dt/dd 構造（2026-08 確認, tests/fixtures/gamboo_result_sample.html）。
# 枠単・車単ともに a-b（ハイフン2車）で並び、最後が車単＝二車単。9車立ての例では
# 枠番≠車番のため枠単 3-6 と車単 3-9 が別組になる。
REAL_HTML = """
<table class="refund_table"><tbody>
<tr>
  <th rowspan="2">2<br>枠<br>連</th><td>複</td>
    <td><dl class="cf"><dt>3=6</dt><dd>820円<span>(2)</span></dd></dl></td>
  <th rowspan="2">2<br>車<br>連</th><td>複</td>
    <td><dl class="cf"><dt>3=9</dt><dd>3,550円<span>(15)</span></dd></dl></td>
  <th rowspan="2">3<br>連<br>勝</th><td>複</td>
    <td><dl class="cf"><dt>3=6=9</dt><dd>12,580円<span>(55)</span></dd></dl></td>
  <th rowspan="2">ワ<br>イ<br>ド</th>
    <td class="wide" rowspan="2">
      <dl class="cf"><dt>3=6</dt><dd>820円<span>(12)</span></dd></dl>
      <dl class="cf"><dt>3=9</dt><dd>950円<span>(13)</span></dd></dl>
      <dl class="cf"><dt>6=9</dt><dd>3,740円<span>(36)</span></dd></dl></td>
</tr>
<tr>
  <td>単</td><td><dl class="cf"><dt>3-6</dt><dd>1,310円<span>(2)</span></dd></dl></td>
  <td>単</td><td><dl class="cf"><dt>3-9</dt><dd>5,310円<span>(24)</span></dd></dl></td>
  <td>単</td><td><dl class="cf"><dt>3-9-6</dt><dd>74,450円<span>(269)</span></dd></dl></td>
</tr>
</tbody></table>
"""


def test_exacta_is_last_hyphen_pair_not_bracket():
    """二車単＝車単。枠単(3-6)ではなく最後のハイフン2車組(3-9)を採る。"""
    got = parse_exacta_payout(REAL_HTML)
    assert got is not None
    assert got.combo == (3, 9)             # 着順どおり（昇順化しない）
    assert got.payout == 5310
    assert got.popularity == 24


def test_exacta_ignores_trifecta_triple():
    """三連単 3-9-6 をハイフン2車と誤認しない。"""
    assert parse_exacta_payout(REAL_HTML).combo == (3, 9)


def test_trifecta_and_trio_from_real_dl_html():
    """実 dl 構造でも三連単・三連複が壊れず取れる。"""
    assert parse_trifecta_payout(REAL_HTML).combo == (3, 9, 6)
    assert parse_trio_payout(REAL_HTML).combo == (3, 6, 9)


def test_exacta_small_field_single_pair():
    """枠のない少数立て（車単のみ）でも車単を採る。"""
    html = ('<table class="refund_table"><tr>'
            '<td>単</td><td><dl class="cf"><dt>2-5</dt><dd>1,800円<span>(7)</span></dd></dl></td>'
            '<td>単</td><td><dl class="cf"><dt>2-5-3</dt><dd>9,900円<span>(30)</span></dd></dl></td>'
            '</tr></table>')
    got = parse_exacta_payout(html)
    assert got.combo == (2, 5) and got.payout == 1800 and got.popularity == 7


def test_exacta_no_table_returns_none():
    assert parse_exacta_payout("<html><body>払戻なし</body></html>") is None


# 2着同着（実物 3720260816030004）: 車単が 4-3 と 4-5 の2つ、三連単も 4-3-5 と 4-5-3。
# 三連単パーサが採る1着-2着 (4,3) に対応する車単 4-3 を返す（三連単と整合させる）。
DEADHEAT_HTML = """
<table class="refund_table"><tbody>
<tr>
  <td><dl class="cf"><dt>3=4</dt><dd>140円<span>(1)</span></dd></dl></td>
  <td><dl class="cf"><dt>4=5</dt><dd>210円<span>(2)</span></dd></dl></td>
  <td><dl class="cf"><dt>3=4=5</dt><dd>410円<span>(2)</span></dd></dl></td>
</tr>
<tr>
  <td><dl class="cf"><dt>4-3</dt><dd>250円<span>(1)</span></dd></dl></td>
  <td><dl class="cf"><dt>4-5</dt><dd>430円<span>(4)</span></dd></dl></td>
  <td><dl class="cf"><dt>4-3-5</dt><dd>640円<span>(2)</span></dd></dl></td>
  <td><dl class="cf"><dt>4-5-3</dt><dd>940円<span>(5)</span></dd></dl></td>
</tr>
</tbody></table>
"""


def test_exacta_deadheat_matches_trifecta_first_two():
    """2着同着では三連単が採る1・2着 (4,3) に対応する車単 4-3 を返す。"""
    assert parse_trifecta_payout(DEADHEAT_HTML).combo == (4, 3, 5)
    got = parse_exacta_payout(DEADHEAT_HTML)
    assert got.combo == (4, 3) and got.payout == 250
