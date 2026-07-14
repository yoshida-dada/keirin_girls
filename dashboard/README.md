# ガールズケイリン 期待値AI ダッシュボード

ガールズケイリン3連単の期待値予測AI（KEIRINプロジェクト）の可視化PWA。
フレームワーク不要・外部ネットワークアクセスなしの素の HTML/CSS/JS で、
GitHub Pages などの静的ホスティングでそのまま動きます。姉妹プロジェクト
`競馬予想/dashboard/` のPWA作法（manifest / service worker / data.json 契約）を踏襲しています。

## ファイル一覧

| ファイル | 役割 |
|---|---|
| `index.html` | ダッシュボード本体。CSS/JSインライン・自己完結。`data.json` を fetch して描画。ライト/ダーク両対応・レスポンシブ。 |
| `data.json` | データ契約（JSONスキーマ）。`scripts/build_dashboard_data.py` が生成する形。現状はサンプル値＋実データ集計。 |
| `manifest.json` | PWAマニフェスト（インストール可能）。 |
| `service-worker.js` | オフラインキャッシュ。HTML/`data.json`＝network-first、アイコン＝cache-first。 |
| `icon-192.png` / `icon-512.png` / `icon-maskable-512.png` / `apple-touch-icon.png` | アイコン類（姉妹プロジェクトから流用）。 |
| `README.md` | このファイル。 |

## 各セクションの意味

1. **本日の推奨買い目** … レース・買い目・モデル確率・必要オッズ（＝1/モデル確率）・
   市場オッズ・EV・推奨投票額（quarter Kelly）。対象バケット内でEV閾値を超えた買い目のみ掲載。
2. **バケット別 累積ROI（レースタイプ×オッズ帯）** … 「軸堅／標準／混戦 × オッズ帯」の
   実現ROIをヒートマップ／表で表示。**ROI100%超（緑）のゾーンだけが実運用候補**。本戦略の核。
3. **レースタイプ分布** … 確率分布の形状から分類した軸堅／標準／混戦の構成比。
4. **キャリブレーション（予測 vs 実現）** … 1点あたり予測確率と実現確率の対応（reliability curve）＋
   Brier score。対角線に近いほど正確、線より下＝過信。
5. **全体 累積ROI 推移** … EV閾値超えの買い目を quarter Kelly で購入した想定の累積損益（haircut後オッズ）。
6. **データ収集状況** … `keirin.sqlite` からの実データ集計（収集レース数・期間・オッズ点数・
   払戻カバレッジ・券種別サンプル数・車立て分布）。

> モデル・バックテスト未完成のあいだ、1〜5はサンプル値（バッジ `サンプル`）または未算出（`未算出`）を表示し、
> 6のデータ収集状況のみ実データ集計です。**狙い目マップ完成までは実弾投入しません。**

## data.json の更新方法

`keirin.sqlite` の実データからデータ収集状況を再集計して `data.json` を再生成します。

```bash
# プロジェクトルート（KEIRIN/）で実行
python scripts/build_dashboard_data.py
```

- 出力先はデフォルトで `dashboard/data.json`。
- DBは **読み取り専用** で短時間だけ開いて即クローズします（収集プロセスと競合しません）。
- 現時点ではモデル依存項目（推奨買い目・バケットROI・レースタイプ分布・キャリブレーション・
  累積ROI）は `status="pending"`（未算出）で出力されます。モデル/バックテスト完成後に
  このスクリプトを拡張して実データ化してください。
- オプション: `--db path/to.sqlite` `--out path/to/data.json`（いずれも相対パス可）。

> リポジトリに同梱の `data.json` はUIを確認できるようサンプル値入りです。
> `build_dashboard_data.py` を実行すると、実データ集計＋未算出プレースホルダに置き換わります。

## GitHub Pages での公開手順

1. `dashboard/` の中身を公開したいリポジトリに置く（例: リポジトリ直下、または `docs/` 配下）。
2. GitHub の **Settings → Pages** で Source を「Deploy from a branch」にし、
   ブランチと公開フォルダ（リポジトリ直下なら `/root`、`docs/` に置いたなら `/docs`）を選択。
3. 数十秒後に `https://<ユーザー名>.github.io/<リポジトリ>/` で公開されます。
   `index.html` がルートに来るように配置してください。
4. 更新時は `python scripts/build_dashboard_data.py` で `data.json` を再生成し、commit & push。
   service worker が `data.json` を network-first で取得するため、再訪時に最新が反映されます。
   外殻（index.html）を更新した場合は `service-worker.js` の `CACHE` 名を上げると確実に更新されます。

### ローカル確認

`file://` 直開きでは fetch と service worker が動きません。簡易サーバ経由で開いてください。

```bash
cd dashboard
python -m http.server 8000
# ブラウザで http://localhost:8000/ を開く
```

## 制約・設計メモ

- 外部CDN・外部ネットワークアクセスなし（CSS/JS/アイコンすべてローカル）。
- 絶対パスのハードコードなし。ビルド不要。秘匿値なし。
- 日本語UI・円/％表示。
- テーマは OS 設定に追従＋右上ボタンで手動切替（localStorage 永続化）。
