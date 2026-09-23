# 新PC移行ガイド（KEIRIN / girls_keirin_ai）（2026-09-22作成）

`競馬予想\migration\MIGRATION.md` と同じ新PC（ミニPC・常時起動）への移行が対象。方針・切替の考え方は共通なので、
ここではKEIRIN固有の差分だけをまとめる。**両プロジェクトを同じ新PCに移す場合、winget導入とGitHub認証は
どちらか一方の`02_setup_new_pc.ps1`が済ませていればもう一方は自動でスキップされる**（インストール済み/認証済みチェック）。

## 0. 競馬予想との違い（重要）

| 項目 | 競馬予想 | KEIRIN |
|---|---|---|
| コード管理 | 非公開GitHub（[banei-keiba](https://github.com/yoshida-dada/banei-keiba)） | 非公開GitHub（[keirin_girls](https://github.com/yoshida-dada/keirin_girls)、以前から運用済み） |
| ダッシュボード | 別リポジトリ（`banei-dashboard`）、GitHub Actionsなし | **同じリポジトリ内**`dashboard/`。GitHub Actionsで自動デプロイ・手動リフレッシュ |
| 常駐プロセス | Task Scheduler 9タスク | **スタートアップフォルダの.bat 1本**（`KeirinGirlsLive.bat` → `scripts\start_live_scheduler.bat` → `live_scheduler.py`が予測更新・ローカル配信・Pagesへのpushを一つのプロセスで担当） |
| DB | SQLite 1ファイル（717MB） | **SQLite 6ファイル 計約1.9GB**（本番/男子/男子probe/v1backup/v3/オッズ時系列。全て`data/`配下・gitignore対象） |
| Python実行環境 | 専用venv（`~/.virtualenvs/pythonProject-umvgXkVN`） | **これまでは素のPython 3.10に直接インストール**。今回の移行を機に**新PCでは専用venv（`KEIRIN\.venv`）に切替**（`scripts/start_live_scheduler.bat`は`.venv`があれば優先、無ければ旧来の絶対パスにフォールバックするよう修正済み・旧PCへの影響なし） |
| .env / 秘密情報 | `.voting_token` / `.vapid_*` | `.env`（VAPID鍵・通知設定）＋ `data/push_subs.json`（Push購読先、端末固有）／`data/notified.json`（通知済みレース記録、端末固有） |

## 1. 事前に実施済みの整理（2026-09-22）

- [x] 未コミットだった分析・検証スクリプト14本をcommit・push済み
- [x] `scripts/start_live_scheduler.bat`を`.venv`優先・旧絶対パスへのフォールバック方式に修正済み（旧PCの挙動は不変）
- [x] **`.git`履歴の圧縮 完了(2026-09-22)**：`git filter-repo`で`dashboard/data.json`・`dashboard/data_men.json`
      の履歴を除去し、現在の内容を1コミットで復元してforce push済み。`live_scheduler.py`が発走前に数分間隔で
      この2ファイルをコミットし続けていたため、2.1万コミット中2.1万件・1.8万件がこの2ファイルの変更のみで
      占められ、`.git`が**2.6GB→6.1MB**に肥大化していた（実質的なコード変更コミットは182件のみ残存）。
      実行前にミラーバックアップを取得済み（ローカル、`git clone --mirror`、`04_verify.ps1`で.gitサイズを
      継続監視）。GitHub上の元の履歴は force push により参照できなくなっている（単独ユーザーのprivate
      repoのため影響なし）。
- [x] pytestベースライン計測（旧PC・素のPython 3.10、`--ignore=api`）: **147 passed, 1 pre-existing failed**
      （`tests/test_gamboo_odds.py::test_race_meta_missing_is_none`。移行と無関係の既存のテスト/コードの
      ズレで、今回の作業で発生したものではない。新PCでも同じ1件が失敗するのが正しい状態）
- [ ] `api/`配下（FastAPI予測配信）は`fastapi`が未インストールで現状動いていない・テストも実行不可
      （`requirements.txt`には記載があるが実際には使われていない模様）。移行では**現状維持**（無視）とし、
      使う場合は別途`pip install fastapi uvicorn`が必要

## 2. 移行対象の一覧

### 2-A. コード（GitHub）
[yoshida-dada/keirin_girls](https://github.com/yoshida-dada/keirin_girls)（非公開）。`dashboard/`もこのリポジトリの中にあり、
`.github/workflows/pages.yml`（push時に自動デプロイ）・`refresh.yml`（手動トリガーのみ。定期実行は無効化済み、
理由はコメントに明記: 「オッズ更新はローカル常駐`live_scheduler.py`が主担当、Actionsとの二重pushを避けるため」）。
新PCでは`git clone`するだけで揃う。

### 2-B. データ（gitで運べないもの）

| 対象 | サイズ | 方法 |
|---|---|---|
| `data/keirin.sqlite`（本番・女子） | 116MB | online backup（`db_tool.py`、WAL対応） |
| `data/keirin_men.sqlite`（男子） | 558MB | 同上 |
| `data/odds_snapshots.sqlite`（オッズ時系列） | 1.1GB | 同上。**再取得不能** |
| `data/keirin_v3.sqlite` | 105MB | 同上（旧バージョン。全部移行する方針で確定） |
| `data/keirin_v1_backup.sqlite` | 35MB | 同上 |
| `data/keirin_men_probe.sqlite`（男子Phase0判断用の一時DB） | 3.4MB | 同上 |
| `.env`（VAPID鍵・通知設定） | — | 同じ値をコピー（変えるとPush購読が無効化） |
| `data/push_subs.json` / `data/notified.json` | — | 端末固有の状態。コピーで引き継ぐ |
| `data/_bt_records.pkl`・各種`.log` | 34MB+ | **移行しない**（再生成可能なキャッシュ・ログ） |

### 2-C. 常駐プロセス
Windows起動時、スタートアップフォルダの`KeirinGirlsLive.bat`（3行、秘匿情報なし）が
`scripts\start_live_scheduler.bat`を呼び、`live_scheduler.py`が常駐する。Task Schedulerではないため
`03_register_tasks.ps1`相当は不要（`02`がStartupフォルダへ.batを配置するだけ）。

**既知の懸念**: 旧PCで確認したところ、`live_scheduler.py`は現在動いておらず、ログ（`data/live_scheduler.log`）の
最終更新も2026-08-20で1ヶ月以上前だった。スタートアップ起動が何らかの理由で機能していない可能性があるため、
新PC移行後は`04_verify.ps1`の該当項目（プロセス起動確認）を必ず見て、動いていなければ手動で
`scripts\start_live_scheduler.bat`を実行して原因を確認すること。

## 3. 手順

**両プロジェクトを移す場合の推奨順序**：先に`競馬予想`のPhase 1を済ませる（Python/Git/GitHub CLI/Claude Codeの
導入とGitHub認証が済む）→ その後にKEIRINのPhase 1を行うと、下記の「共通ブートストラップ」が全部スキップされて
KEIRIN側は「clone→DB復元→専用venv作成」だけで終わる。単独でKEIRINだけ先に行っても支障はない
（`02_setup_new_pc.ps1`は両方とも「インストール済み／認証済みなら何もしない」判定を持つ）。

### Claude Codeに任せる場合（推奨）

競馬予想と同じ新PCで、先にそちらのブートストラップ（Windows初期設定→Git・Claude Codeの導入→`/login`）を
済ませていれば、KEIRIN側はもう人間の手を介さずに済む。

1. **人間が行う（競馬予想を未セットアップの場合のみ・代行不可・5〜10分）**：
   ```powershell
   winget install --id Git.Git -e --silent --accept-package-agreements --accept-source-agreements
   irm https://claude.ai/install.ps1 | iex
   ```
   `claude`を起動し`/login`（ブラウザOAuth）。※競馬予想を先に設定済みならこの手順は不要。
2. **Claude Codeに依頼する（例）**：「`https://github.com/yoshida-dada/keirin_girls`をcloneして、
   `migration/MIGRATION.md`のPhase 1に従ってセットアップして。詰まったら聞いて」
   → `gh auth login`のブラウザコード入力（未認証の場合のみ人間に一度依頼）→ clone → `02_setup_new_pc.ps1`を
   **まず`-Bundle`無しで**実行（専用venv作成・依存インストールまで進む。`-Bundle`は必須パラメータではない）
   → `04_verify.ps1`の結果報告。旧PCのbundleが用意でき次第、同じコマンドに`-Bundle <path> -SkipInstall`を
   付けて再実行するよう伝えれば、DB6ファイル・`.env`・Startupランチャーも追加で復元される。
3. **それでも人間が必要な箇所**：GitHub認証のブラウザ操作（未認証の場合のみ）、
   `live_scheduler.py`のStartup自動起動が実際に動くかのログオン確認（既知の懸念、§2-C参照）、
   Phase 2（切替日、新旧PC両方を見る必要がある）。

### Phase 0（旧PCで）
- [x] `git status`がクリーン・push漏れ無しを確認（`01`が自動チェック）
- [x] 履歴圧縮を実施済み（§4。ローカルミラーバックアップは保持中）
- [ ] リハーサル: `.\migration\01_export_old_pc.ps1 -Dest D:\rehearsal -SkipDb -DryRun`
- [ ] 外付けSSD（**BitLocker暗号化推奨**。bundleに`.env`等の秘匿情報が入る）を用意。bundle本体は
      DBが主体で約1.9GB（`-SkipDb`ならKB単位）

### Phase 1：新PCセットアップ（旧PCは通常稼働のまま。手動で行う場合の詳細手順）
1. 新PC初期設定：競馬予想と共通（同じMicrosoftアカウント、リージョン=日本、電源接続）。
   **すでに競馬予想のPhase 1を終えているなら、この手順は完了済み。**
2. 旧PCでbundle作成：
   ```powershell
   cd "C:\Users\yoshi\PycharmProjects\pythonProject\KEIRIN"
   .\migration\01_export_old_pc.ps1 -Dest E:\keirin_bundle
   ```
   （警告なしで完了すること。`repo_git_url`が空という警告が出たら、pushを先に済ませてから再実行）
3. 新PCで（PowerShell、管理者不要。UACは出る）：`02_setup_new_pc.ps1`は`keirin_girls`リポジトリの
   `migration\`配下にあり、bundleには含まれない（コード＝GitHubの原則どおり）。競馬予想を先にセットアップ
   済みなら`git`・`gh`は既に使えるので、そのまま次のコマンドで足りる：
   ```powershell
   git clone https://github.com/yoshida-dada/keirin_girls.git "$env:TEMP\keirin-bootstrap"
   Set-ExecutionPolicy -Scope Process Bypass
   ```
   競馬予想を未セットアップの場合は、先に上の「Claude Codeに任せる場合」手順1のブートストラップと
   `gh auth login`→`gh auth setup-git`を行ってから上記を実行する。

   **`-Bundle`は省略可**。旧PCのbundle（外付けドライブ等）がまだ手元に無い段階でも、まずコードだけの
   ブートストラップを進められる：
   ```powershell
   & "$env:TEMP\keirin-bootstrap\migration\02_setup_new_pc.ps1"
   ```
   この場合の流れ：winget導入（未導入分のみ）→ gh認証（未認証の場合のみ）→ `git clone`でリポジトリ復元 →
   専用venv（`KEIRIN\.venv`）作成・依存インストール → `04_verify.ps1`（DB6ファイル・`.env`・Startup
   ランチャーはWARNになる）。bundleが用意でき次第、同じコマンドに`-Bundle <path> -SkipInstall`を付けて
   再実行すれば（冪等なので）その部分だけ追加で復元される：
   ```powershell
   & "$env:TEMP\keirin-bootstrap\migration\02_setup_new_pc.ps1" -Bundle E:\keirin_bundle -SkipInstall
   ```
4. 手動作業（自動化不可）：
   - ログオンし直すか、Startupフォルダの`KeirinGirlsLive.bat`を手動実行して`live_scheduler.py`が
     起動することを確認（既知の懸念、§2-C参照。起動しない場合はコンソールのエラーを確認する）
   - GitHub Pagesダッシュボードがこのリポジトリの`.github/workflows/pages.yml`で自動デプロイされる
     ことを確認（`git push`後、数分待って`https://yoshida-dada.github.io/keirin_girls/`相当を開く）
5. リハーサル確認：`.\migration\04_verify.ps1`が0 FAIL（WARNは新PCでは0件になるはず。旧PCで見えた
   2件のWARNは「.venvが無い」「live_scheduler.pyが動いていない」で、新PCなら両方解消される想定）。

### Phase 2（切替日。競馬予想の切替日と合わせるのが望ましい）
1. 旧PC：スタートアップフォルダから`KeirinGirlsLive.bat`を削除（または退避）し、実行中の`live_scheduler.py`を終了
2. 旧PC：最終bundle作成（DB最新化）：`.\migration\01_export_old_pc.ps1 -Dest E:\keirin_bundle`
3. 新PC：`02_setup_new_pc.ps1 -Bundle E:\keirin_bundle -SkipInstall -SkipPython`（DB上書き復元＋`git pull`）
4. 新PC：`04_verify.ps1`全PASS → ログオンし直すか手動で`scripts\start_live_scheduler.bat`を実行して起動確認
5. 確認：GitHub Pagesダッシュボードが新PCからのpushで更新される／スマホPush通知が届く

## 4. `.git`履歴圧縮の手順（実行者向けメモ・2026-09-22実施済み。今後同様の肥大化が起きた場合の再手順として残す）

```powershell
# 1. ミラーバックアップ（実施済み・保持）
git clone --mirror https://github.com/yoshida-dada/keirin_girls.git <backup先>

# 2. 現在の内容を退避
Copy-Item dashboard\data.json      <tmp>\data.json
Copy-Item dashboard\data_men.json  <tmp>\data_men.json

# 3. 履歴から除去（origin remoteは filter-repo が自動で外すので後で再設定）
python -m pip install --user git-filter-repo
python -m git_filter_repo --path dashboard/data.json --path dashboard/data_men.json --invert-paths --force

# 4. 現在の内容を1コミットで復元
Copy-Item <tmp>\data.json      dashboard\data.json
Copy-Item <tmp>\data_men.json  dashboard\data_men.json
git add dashboard/data.json dashboard/data_men.json
git commit -m "Restore live dashboard data after history cleanup"

# 5. remote再設定してforce push
git remote add origin https://github.com/yoshida-dada/keirin_girls.git
git push origin --force --all
git push origin --force --tags
```

単独ユーザーのprivateリポジトリのため、force pushによる影響（他者のcloneとの不整合）は無い。
実行後は`.git`サイズが1.7GB→数十MB程度まで縮む見込み（`04_verify.ps1`が500MB超で警告する）。

## 5. 自動化の構成

| ファイル | 役割 |
|---|---|
| `01_export_old_pc.ps1` | bundle作成（DB6ファイル/`.env`/Startup.bat＋manifest。コードはGitHub） |
| `02_setup_new_pc.ps1` | winget→gh認証→`git clone`→DB/secrets/Startup復元→専用venv作成→依存インストール→検証 |
| `04_verify.ps1` | Python/依存/DB6ファイル/secrets/Startup配置/git状態(サイズ含む)/pytest |
| `db_tool.py` | 競馬予想と共通（SQLite online backup＋integrity_check＋件数manifest照合） |
| `requirements.lock.txt` | 旧PCの素のPython環境から freeze した56パッケージ |

テスト状況：旧PC上で`db_tool.py backup/check`（3.4MBのprobe DBで動作確認済み）、pytestベースライン取得済み
（`--ignore=api`で147 passed, 1 pre-existing failed）、`01`/`04`は構文解析＋実行確認（`04_verify.ps1 -SkipTests`
は0 FAIL・2 WARN=想定どおり）。`02`は構文解析のみ（**一度、日本語コメントがBOM無しUTF-8のためWindows
PowerShell 5.1で構文エラーになるバグを作り込み、pushしてから発見・修正した**。ASCII化して再push済み）。
履歴圧縮（§4）は本番実行・force push・GitHub側反映まで確認済み。

### 修正履歴

- **2026-09-23（旧PC側で修正・push済み）**：競馬予想の`02_setup_new_pc.ps1`を新PC（A7_MAX / ユーザーdada）で
  実際に試したところ、`-Bundle`が必須パラメータのためbundle（外付けドライブ）が無い段階では**コードのclone
  すら実行できない**ことが判明。KEIRIN側の`02_setup_new_pc.ps1`も同一設計だったため、先回りで同様に修正。
  `-Bundle`を省略可に変更し、GitHub URL（`https://github.com/yoshida-dada/keirin_girls.git`）を既定値として
  直接持つようにした。以後は：
  1. bundleなしで`02_setup_new_pc.ps1`を実行 → winget/gh認証/`git clone`/専用venv構築まで進む
  2. bundleが用意でき次第、`-Bundle <path> -SkipInstall`を付けて同じコマンドを再実行 → DB6ファイル/`.env`/
     Startupランチャーを追加復元
  という2段階で進められる（§3 Phase 1の手順は更新済み）。**新PC側で使っているコピーがこの修正より前の
  ものなら、`git pull`するか改めてGitHubからcloneし直すこと**（MIGRATION.mdの差し替えだけでは
  `02_setup_new_pc.ps1`本体の修正は反映されない）。**新PCでの`02`本実行はまだ未テスト**（構文解析のみ確認）。

## 6. 今後の改善（未着手）
- `live_scheduler.py`のStartup自動起動が機能していない疑い（ログが1ヶ月停止）の原因調査
- `api/`（FastAPI予測配信）を使うか、`requirements.txt`から外すか整理
- Postgres移行（`.env.example`记载の本番想定）は未着手のまま。当面SQLiteで運用継続
