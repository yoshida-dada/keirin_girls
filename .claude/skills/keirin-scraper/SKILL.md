---
name: keirin-scraper
description: KEIRINプロジェクトでガールズケイリンのデータ取得スクレイパー（オッズ/出走表/結果/選手成績）を新規追加・修正するときの手順。GambooBET等のHTMLをパースしDBスキーマへ載せる。
---

# keirin-scraper

ガールズケイリン期待値予測AI（KEIRINプロジェクト）のデータ取得を追加・修正するための手順。
主系は GambooBET（楽天Kドリーム keirin.kdreams.jp）。取得元の選定根拠は CLAUDE.md「データ取得元」参照。

## 原則（必ず守る）
1. **取得は `src/collect/base.py: fetch()` を通す。** ホスト単位のスロットリング（既定1秒以上）・
   指数バックオフ・`apparent_encoding` 自動判定が入っている。生の requests.get を各所で書かない。
2. **個人利用・低頻度。** 4xxは即中断（robots/規約シグナル）。連続アクセスでBANされる設計にしない。
   netkeirin(keirin.netkeiba.com)は規約でスクレイピング禁止のため**使わない**。
3. **パースは純関数に切り出す。** `parse_*(html)->dict` をネットワークから分離し、保存HTMLフィクスチャで
   オフラインテストする（下記手順）。ネットワークテストはCIで回さない。
4. **DBスキーマ（`db/schema.sql`）に合わせて出力する。** 車番は1始まりint。三連単comboは
   `(1着,2着,3着)` のintタプル。DB保存時は "-" 連結文字列（例 "3-1-5"）。
5. **欠損検知を必ず入れる。** 三連単は `base.detect_missing_trifecta(odds, field_size)` で
   N*(N-1)*(N-2)点そろっているか確認（7車=210点）。

## 新しいスクレイパーを追加する手順
1. **実HTMLを1回だけ取得してフィクスチャ保存**（サイトへの負荷を最小化）:
   ```python
   from src.collect.base import fetch
   open("tests/fixtures/<name>.html","w",encoding="utf-8").write(fetch(url).text)
   ```
2. **構造を調べる**: BeautifulSoupでテーブルのclass・行/セル構成をダンプしてから書く。
   憶測でセレクタを書かない。日本語はcp932コンソールで文字化けするが解析には無害。
3. **`parse_<kind>(html) -> dict` を `src/collect/<site>_<kind>.py` に実装。**
4. **`tests/fixtures/<name>.html` に対するテストを書く**（実測値2〜3点をハードコードで固定）。
5. `python -m pytest tests/` で緑を確認。

## GambooBET 三連単オッズの構造（実装済み: `src/collect/gamboo_odds.py`）
- URL: `.../race-card/odds/{開催コード}/{開催日コード}/{R}/3rentan/`
- 1着車番ごとに `<table class="odds_table bt5 {1着車番}">` が1枚。
- ヘッダ行に純数字thが並ぶ（=3着車番の列）。**前後に空thが付く**ので数字thだけ拾う。
- データ行: 先頭th=2着車番、続くtd=各3着列のオッズ。`class="empty"` は不可能な組合せ。
  **行末尾に2着thが再掲される**ためtdだけ数える。
- 出走表・結果ページも同系のURL規則（`/race-card/` `/race-card/result/`）。EUC-JP/UTF-8はページ依存なので
  `apparent_encoding` 任せにする（`fetch()`が処理済み）。

## 実装済み（S1オッズ収集パイプライン。実データで210点/7車ガールズを検証済）
- `gamboo_schedule.py`: 日付→ガールズ開催発見(`icon_girls`)、開催→レース番号一覧。
- `gamboo_odds.py`: 三連単オッズ全点＋締切時刻(`<dt>締切</dt>`)パース。
- `snapshot.py`: 締切60分前〜締切の窓判定＋1レース収集。`db/repository.py`(SQLite暫定)へ保存。
- `scripts/collect_odds_snapshot.py`: 当日ガールズ全レースを窓内だけ収集するランナー（cron 5分毎想定）。
- race_id = 開催日コード + R番号2桁。combo表記は "a-b-c"（着順どおり）。

- `gamboo_racecard.py`: 出走表パーサ。**オッズページ同梱**の`racecard_table[0]`から
  車番/枠番/選手名/府県/年齢/期別/級班/脚質/ギヤ/**競走得点**を抽出。選手名セルは
  `名前 + <span class="home">府県/年齢/期別</span>`。名前セルの直後4td=級班/脚質/ギヤ/競走得点。
  車7は枠番セルが欠落しうるのでclass(`td.num`/`td.rider`)基準で辿る。登録番号はこのページに無い。
- **ガールズ判定はレース単位**（`is_girls_race`＝全員の級班がLで始まる）。開催のicon_girlsは
  「その開催にガールズ回がある」だけで、個々のレースは男子戦のことがある（実例: 会場11 R1はA級）。
  snapshot収集は `only_girls=True` でL級レースだけ保存する。

- `keirin_station.py`: 競輪ステーション選手成績（robots一般UAは/keirindb/許可）。
  - `parse_player_detail`: 選手詳細から 直近4ヶ月(出走回数/B/H/S・勝率/2連対/3連対・着順・**決まり手率=脚質率**・競走得点平均/最高/最低)＋年度別通算。URL `/keirindb/player/detail/{登録番号}/`。
  - 選手検索: `/keirindb/search/player/` GET。`player_profile[name_1]`(姓)/`[name_2]`(名)/`[girls_flag]=1`＋
    `submit[btn][player_profile][get]`が必須。結果は名前行(anchorに登録番号)＋直後の登録番号行の2行組。
  - `resolve_rider_id(name,pref,term)`: GambooBET出走表→登録番号の名前突合（空白除去＋府県正規化＋期別）。
    GambooBETは県サフィックス無し/検索結果も無し/詳細は"徳島県"付き → `normalize_prefecture`で吸収。
  - ★タイムは競輪学校時代の記録のみ（直近レースのタイムは無い）。バンク別勝率テーブルも無い→オッズパーク補完。
  - ライブ検証済: 青木美保(埼玉,118期)→015485、得点53.6・決まり手率取得。

- `gamboo_result.py`: レース結果。URL `/race-card/result/{開催}/{開催日}/{R}/`。
  - `result_table`: 着順/車番/選手名/着差/**上り(上がりタイム秒)**/決まり手/S・B/勝敗因。
  - `refund_table`: 全券種払戻。**券種はセパレータで判定**（`a-b-c`=三連単/`a=b=c`=三連複/`a-b`=車単/`a=b`=車連）。
    `parse_trifecta_payout`が三連単の(組合せ,払戻金,人気)を返す＝学習ラベル＋ROI決済。
  - ★全210点の機械買いシミュレーションには refund(的中1点)では足りず、オッズページの確定オッズ(odds_final_trifecta)が別途必要。
  - 検証済: 平塚2025-12-28 R11、三連単3-9-6=74,450円(人気269)、上がりタイム取得。

## 残タスク（着手時に本skillへ追記）
- オッズパーク: バンク別（競輪場別）成績の補完
- 直近5走/10走の個別着順（競輪ステーション「最近の成績」レース単位のパース）
- 本番PostgreSQLへの接続層差し替え（repositoryのDDL/接続のみ。combo表記・主キーは流用）
- cron/APScheduler常駐（Raspberry Pi）と欠損アラート通知
- 実L級レースの出走表フィクスチャ追加（現状の出走表フィクスチャは7車A級。構造は同一）
