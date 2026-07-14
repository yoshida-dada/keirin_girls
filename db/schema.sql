-- ガールズケイリン 期待値予測AI DBスキーマ (PostgreSQL)
-- girls_keirin_ai_spec.md Phase1「PostgreSQLスキーマ設計」準拠。
--
-- 設計方針:
--   * 7車立て固定 → 3連単は 7*6*5 = 210 通り。全点のオッズ・払戻を保持する（バケット分析には
--     的中以外の買い目のオッズも必要 = 課題A）。
--   * 選手成績詳細は時点(as_of)で変動するため、rider_stats は snapshot_date を持ち
--     「レース時点で最新の状態」を時間軸結合で引く（リーク防止 = 課題E/2.7）。
--   * オッズは締切直前に下落する(課題D)ため、時系列スナップショット(odds_snapshots_trifecta)と
--     確定オッズ(odds_final_trifecta)を分離。haircut係数は両者の差から推定する。

-- ============================================================
-- マスタ / レース基本
-- ============================================================

CREATE TABLE IF NOT EXISTS races (
    race_id         TEXT PRIMARY KEY,            -- 例: YYYYMMDD + 場コード + R番号
    race_date       DATE        NOT NULL,
    venue           TEXT        NOT NULL,        -- 競輪場名
    venue_code      TEXT,                        -- 場コード
    race_number     INTEGER     NOT NULL,
    race_name       TEXT,
    grade           TEXT,                        -- 開催格 (F2/F1/G等)
    bank_length_m   INTEGER,                     -- バンク周長 333/400/500 (課題E, 2.3)
    is_indoor       BOOLEAN,                     -- 屋内/屋外
    session_type    TEXT,                        -- normal/night/midnight (ナイター/ミッドナイト適性 2.3)
    field_size      INTEGER     NOT NULL DEFAULT 7,
    scraped_at      TIMESTAMPTZ
);

-- 選手マスタ（登録番号で一意）
CREATE TABLE IF NOT EXISTS riders (
    rider_id        INTEGER PRIMARY KEY,         -- 選手登録番号
    rider_name      TEXT        NOT NULL,
    prefecture      TEXT,                        -- 登録地
    term            INTEGER,                     -- 期別
    birth_date      DATE
);

-- 出走表（レース×選手）: 発走前に確定する情報のみ
CREATE TABLE IF NOT EXISTS entries (
    race_id         TEXT        NOT NULL REFERENCES races(race_id),
    rider_id        INTEGER     NOT NULL REFERENCES riders(rider_id),
    bracket_number  INTEGER     NOT NULL,        -- 枠番 (Tier1特徴量 2.9)
    car_number      INTEGER     NOT NULL,        -- 車番
    racing_score    REAL,                        -- 競走得点 (Tier1土台 2.1)
    class_rank      TEXT,                        -- 級班 (2.1)
    leg_type        TEXT,                        -- 脚質 逃/捲/差/マーク (2.4)
    age             INTEGER,
    PRIMARY KEY (race_id, rider_id)
);

-- 結果（決まり手・B/H回数含む = Phase1要件）
CREATE TABLE IF NOT EXISTS results (
    race_id         TEXT        NOT NULL REFERENCES races(race_id),
    rider_id        INTEGER     NOT NULL REFERENCES riders(rider_id),
    position        INTEGER,                     -- 着順（目標変数）
    kimarite        TEXT,                        -- 決まり手 逃/捲/差/マーク
    b_count         INTEGER,                     -- B回数（バック先頭通過）
    h_count         INTEGER,                     -- H回数（ホーム先頭通過）
    passing_order   TEXT,                        -- 周回ごとの位置取り生文字列 (2.4)
    is_dsq          BOOLEAN     DEFAULT FALSE,    -- 失格/落車
    PRIMARY KEY (race_id, rider_id)
);

-- ============================================================
-- 選手成績詳細（時点スナップショット, as_of結合で使う = 課題E/リーク防止）
-- ============================================================
-- バンク別成績・直近走・脚質率・中何日・落車/失格/欠場歴などをまとめて保持。
-- 「そのレース日以前で最新の snapshot_date」を引くことで未来情報リークを防ぐ。
CREATE TABLE IF NOT EXISTS rider_stats (
    rider_id            INTEGER   NOT NULL REFERENCES riders(rider_id),
    snapshot_date       DATE      NOT NULL,
    -- 基礎能力 (2.1)
    win_rate            REAL,     -- 勝率
    top2_rate           REAL,     -- 連対率
    top3_rate           REAL,     -- 3連対率
    avg_finish          REAL,     -- 平均着順
    -- 直近コンディション (2.2)
    recent5_positions   TEXT,     -- 直近5走着順（"-"連結）
    recent10_positions  TEXT,     -- 直近10走着順
    win_rate_30d        REAL,
    win_rate_90d        REAL,
    days_since_last     INTEGER,  -- 中何日
    fell_last_race      BOOLEAN,  -- 前日落車/失格歴
    returning_from_rest BOOLEAN,  -- 欠場明け
    -- 脚質率 (2.4)
    escape_rate         REAL,     -- 逃げ率
    dash_rate           REAL,     -- 捲り率
    closing_rate        REAL,     -- 差し率
    mark_rate           REAL,     -- マーク率
    -- 成長率算出用 (2.8)
    racing_score_90d_ago REAL,
    fetched_at          TIMESTAMPTZ,
    PRIMARY KEY (rider_id, snapshot_date)
);

-- バンク別成績（競輪場×選手, as_of, 2.3）
CREATE TABLE IF NOT EXISTS rider_bank_stats (
    rider_id        INTEGER   NOT NULL REFERENCES riders(rider_id),
    venue_code      TEXT      NOT NULL,
    snapshot_date   DATE      NOT NULL,
    bank_win_rate   REAL,
    bank_top3_rate  REAL,
    starts          INTEGER,
    PRIMARY KEY (rider_id, venue_code, snapshot_date)
);

-- ============================================================
-- オッズ（3連単全210点）
-- ============================================================
-- 締切前の時系列スナップショット（締切60分前〜締切まで5〜10分刻み, 課題D/Phase1）
CREATE TABLE IF NOT EXISTS odds_snapshots_trifecta (
    race_id     TEXT        NOT NULL REFERENCES races(race_id),
    combo       TEXT        NOT NULL,   -- 車番を "-" 連結（着順どおり, 例 "3-1-5"）
    odds        REAL        NOT NULL,
    taken_at    TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (race_id, combo, taken_at)
);

-- 確定オッズ（全210点 = 的中以外も保持。バケット分析の必須要件 課題A）
CREATE TABLE IF NOT EXISTS odds_final_trifecta (
    race_id     TEXT        NOT NULL REFERENCES races(race_id),
    combo       TEXT        NOT NULL,   -- 車番を "-" 連結
    odds        REAL        NOT NULL,
    PRIMARY KEY (race_id, combo)
);

-- 実払戻（的中組合せのみ, ROI集計の照合用）
CREATE TABLE IF NOT EXISTS payouts_trifecta (
    race_id     TEXT        NOT NULL REFERENCES races(race_id),
    combo       TEXT        NOT NULL,   -- 車番を "-" 連結（着順どおり）
    payout      INTEGER     NOT NULL,   -- 100円あたり払戻金
    popularity  INTEGER,
    PRIMARY KEY (race_id, combo)
);

-- ============================================================
-- 運用（買い目記録 / バケット定義）
-- ============================================================
-- 買い目エンジンが生成した注文の記録（実購入は行わずまずペーパートレード, 課題F）
CREATE TABLE IF NOT EXISTS bet_orders (
    order_id        BIGSERIAL PRIMARY KEY,
    race_id         TEXT      NOT NULL REFERENCES races(race_id),
    combo           TEXT      NOT NULL,
    stake_yen       INTEGER   NOT NULL,
    odds_at_bet     REAL,                        -- 投票時点の表示オッズ
    model_prob      REAL,                        -- モデル確率
    expected_value  REAL,
    race_type       TEXT,                        -- 軸堅/標準/混戦（判定結果）
    odds_bucket     TEXT,                        -- オッズ帯ラベル
    reason          TEXT,
    status          TEXT      NOT NULL DEFAULT 'paper',  -- paper/confirmed/cancelled
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_races_date_number      ON races(race_date, race_number);
CREATE INDEX IF NOT EXISTS idx_entries_rider          ON entries(rider_id);
CREATE INDEX IF NOT EXISTS idx_results_rider          ON results(rider_id);
CREATE INDEX IF NOT EXISTS idx_rider_stats_asof       ON rider_stats(rider_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_rider_bank_stats_asof  ON rider_bank_stats(rider_id, venue_code, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_odds_snap_tri          ON odds_snapshots_trifecta(race_id, taken_at);
CREATE INDEX IF NOT EXISTS idx_bet_orders_race        ON bet_orders(race_id);
