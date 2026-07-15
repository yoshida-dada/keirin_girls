"""参考分析: 開催時間帯(セッション)と季節がガールズ競輪の着順にどの程度効くか。

読み取り専用。data/keirin.sqlite を集計し、以下を報告する:
  1. セッション別・季節別の分布（レース数/延べ選手数）
  2. セッション別・季節別の成績差（平均着順/上り(last_lap)/1着率）— 全体＋選手内ばらつき
  3. 「選手×セッション」「選手×季節」の as-of 成績（発走前・リーク無し）を1特徴として
     既存の簡易PLモデルに足し、time_split で ece/logloss/top1 が改善するか軽く確認
  4. 特徴化する価値があるか（Yes/No＋根拠）

セッションは races.deadline の「時」から、季節は race_id から復元した実施日の「月」から決める。
実施日復元は src.features.rider_history._race_date_from_id をそのまま使う（race_date は初日固定バグ）。

実行: PYTHONIOENCODING=utf-8 python scripts/analyze_session_season.py
"""
from __future__ import annotations

import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.rider_history import _race_date_from_id
from src.model.training_data import load_samples, PL_FEATURES_FULL
from src.model.train_pl import train_pl, PLModel
from src.model.evaluate import evaluate, time_split

DB_PATH = ROOT / "data" / "keirin.sqlite"


# ---------------------------------------------------------------------------
# 区分ロジック
# ---------------------------------------------------------------------------
def session_of(deadline: str | None) -> str | None:
    """発走/締切時刻 "HH:MM" の「時」からセッションを分類する。

    deadline のヒストグラム（8-23時に分布, 空白帯は 11時台/14時台）を踏まえた閾値:
      morning   : ~10:59        (hour <= 10)
      day       : 11:00-15:59   (hour 11-15)
      night     : 16:00-20:59   (hour 16-20, ナイター)
      midnight  : 21:00-        (hour >= 21, ミッドナイト/未明)
    """
    if not deadline:
        return None
    try:
        h = int(deadline.split(":")[0])
    except (ValueError, IndexError):
        return None
    if h <= 10:
        return "morning"
    if h <= 15:
        return "day"
    if h <= 20:
        return "night"
    return "midnight"


def season_of(race_id: str) -> str | None:
    """実施日（race_id から復元）の「月」から季節を分類する。3-5春/6-8夏/9-11秋/12-2冬。"""
    d = _race_date_from_id(race_id)
    if not d:
        return None
    m = int(d[5:7])
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "winter"


SESSIONS = ["morning", "day", "night", "midnight"]
SEASONS = ["spring", "summer", "autumn", "winter"]
SESSION_JP = {"morning": "モーニング", "day": "デイ", "night": "ナイター", "midnight": "ミッドナイト"}
SEASON_JP = {"spring": "春", "summer": "夏", "autumn": "秋", "winter": "冬"}


# ---------------------------------------------------------------------------
# 生データ読み込み（レース単位のメタ＋結果行）
# ---------------------------------------------------------------------------
def load_rows():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        races = conn.execute(
            "SELECT race_id, deadline, field_size FROM races"
        ).fetchall()
        results = conn.execute(
            "SELECT race_id, position, car_number, rider_name, last_lap"
            " FROM results WHERE position IS NOT NULL AND rider_name IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return races, results


# ---------------------------------------------------------------------------
# 1. 分布  ＋  2. 成績差
# ---------------------------------------------------------------------------
def analyze_distribution_and_diffs(races, results):
    race_meta = {}  # race_id -> (session, season, field_size)
    for rid, deadline, fs in races:
        race_meta[rid] = (session_of(deadline), season_of(rid), fs)

    # レース数分布（全レース）
    race_by_session = defaultdict(int)
    race_by_season = defaultdict(int)
    for rid, (sess, seas, fs) in race_meta.items():
        race_by_session[sess] += 1
        race_by_season[seas] += 1

    # 結果行を区分ごとに集計
    def new_agg():
        return {"n": 0, "pos": [], "lap": [], "win": 0, "riders": set()}

    by_session = {k: new_agg() for k in SESSIONS}
    by_season = {k: new_agg() for k in SEASONS}
    overall = new_agg()

    # 選手内ばらつき用: rider -> session -> [positions], rider -> season -> [positions]
    rs_pos = defaultdict(lambda: defaultdict(list))   # rider -> session -> positions
    ry_pos = defaultdict(lambda: defaultdict(list))   # rider -> season  -> positions

    for rid, pos, car, name, lap in results:
        meta = race_meta.get(rid)
        if not meta:
            continue
        sess, seas, fs = meta
        for agg, key in ((by_session, sess), (by_season, seas)):
            if key is None:
                continue
            a = agg[key]
            a["n"] += 1
            a["pos"].append(pos)
            if lap is not None:
                a["lap"].append(lap)
            if pos == 1:
                a["win"] += 1
            a["riders"].add(name)
        overall["n"] += 1
        overall["pos"].append(pos)
        if lap is not None:
            overall["lap"].append(lap)
        if pos == 1:
            overall["win"] += 1
        overall["riders"].add(name)
        if sess:
            rs_pos[name][sess].append(pos)
        if seas:
            ry_pos[name][seas].append(pos)

    return {
        "race_by_session": race_by_session,
        "race_by_season": race_by_season,
        "by_session": by_session,
        "by_season": by_season,
        "overall": overall,
        "rs_pos": rs_pos,
        "ry_pos": ry_pos,
    }


def fmt_agg_table(title, keys, jp, race_counts, agg_map, overall):
    lines = [f"\n### {title}"]
    lines.append(f"{'区分':<12} {'レース数':>7} {'延べ選手':>8} {'実選手数':>7} "
                 f"{'平均着順':>7} {'平均上り':>7} {'1着率':>7}")
    for k in keys:
        a = agg_map[k]
        n = a["n"]
        if n == 0:
            lines.append(f"{jp[k]:<12} {race_counts[k]:>7} {'0':>8}")
            continue
        avg_pos = statistics.mean(a["pos"])
        avg_lap = statistics.mean(a["lap"]) if a["lap"] else float("nan")
        winrate = a["win"] / n
        lines.append(f"{jp[k]:<12} {race_counts[k]:>7} {n:>8} {len(a['riders']):>7} "
                     f"{avg_pos:>7.3f} {avg_lap:>7.3f} {winrate:>7.3f}")
    o = overall
    lines.append(f"{'全体':<12} {sum(race_counts.values()):>7} {o['n']:>8} {len(o['riders']):>7} "
                 f"{statistics.mean(o['pos']):>7.3f} "
                 f"{statistics.mean(o['lap']):>7.3f} {o['win']/o['n']:>7.3f}")
    return "\n".join(lines)


def within_rider_spread(pos_map, keys, jp, min_per_cell=10, min_cells=2):
    """同一選手の区分別平均着順の（区分内ばらつき）を集計する。

    各選手について、min_per_cell 走以上ある区分が min_cells 以上あるとき、その選手の
    「区分別平均着順の最大-最小(レンジ)」と標準偏差を取り、全選手で平均する。
    区分自体の効果が大きいほどこのレンジは大きくなる（選手固定で見た区分差）。
    """
    ranges = []
    stdevs = []
    n_riders = 0
    for name, cells in pos_map.items():
        means = {k: statistics.mean(v) for k, v in cells.items()
                 if k in keys and len(v) >= min_per_cell}
        if len(means) < min_cells:
            continue
        vals = list(means.values())
        ranges.append(max(vals) - min(vals))
        stdevs.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        n_riders += 1
    if not ranges:
        return None
    return {
        "n_riders": n_riders,
        "mean_range": statistics.mean(ranges),
        "mean_std": statistics.mean(stdevs),
    }


# ---------------------------------------------------------------------------
# 3. as-of 特徴の信号チェック
# ---------------------------------------------------------------------------
def build_asof_feature(samples, kind: str, mode: str = "cell"):
    """各サンプルの各選手に「選手×区分」の as-of 平均着順を割り当てて返す。

    kind: "session" または "season"。
    mode:
      "cell"    : その選手のその区分での過去平均着順（区分特化 + 選手実力が混ざる）
      "overall" : 区分を無視した その選手の全走過去平均着順（純粋な選手実力の対照群）
      "delta"   : cell - overall（区分特化の“ずれ”だけを取り出す。選手実力を相殺）
    as-of 作法: レースを実施日昇順に処理し、当該レース**より前**の走のみ集計（発走前値）。
    未経験セルは overall/delta で 0、cell で overall へフォールバック、全て無なら 4.0/0.0。

    返り値: {(race_id, car_number): feature_value}
    """
    # サンプルを実施日→race_id で厳密整列（load_samples は date昇順だが date は初日なので同日内で
    # 実施日順が崩れうる。ここでは race_id 先頭14桁の実施日で並べ替えて as-of を厳密化）。
    ordered = sorted(samples, key=lambda s: (_race_date_from_id(s.race_id) or "", s.race_id))

    classify = session_of_by_id if kind == "session" else season_of
    # session は deadline が要るので race_id からは引けない → 事前に deadline マップを作る
    hist_cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # name -> cellkey -> [sum_pos, cnt]
    hist_all = defaultdict(lambda: [0, 0])                         # name -> [sum_pos, cnt]

    feat = {}
    name_of = {}  # (race_id, car) -> name  （車番→氏名は entries から: samples に無いので後で補完）

    # samples には氏名が無い。entries から (race_id, car)->name を引く。
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only=1")
    try:
        rids = tuple(s.race_id for s in ordered)
        # まとめて取得
        rid_set = set(rids)
        rows = conn.execute(
            "SELECT race_id, car_number, rider_name FROM entries"
        ).fetchall()
        for rid, car, nm in rows:
            if rid in rid_set:
                name_of[(rid, car)] = nm
        # session 用 deadline
        deadline_of = dict(conn.execute("SELECT race_id, deadline FROM races").fetchall())
    finally:
        conn.close()

    for s in ordered:
        if kind == "session":
            cell = session_of(deadline_of.get(s.race_id))
        else:
            cell = season_of(s.race_id)
        # 1) 発走前の as-of 値を各選手に割り当て
        for car in s.car_numbers:
            nm = name_of.get((s.race_id, car))
            default = 0.0 if mode in ("overall", "delta") else 4.0
            if nm is None:
                feat[(s.race_id, car)] = default
                continue
            cell_avg = (hist_cell[nm][cell][0] / hist_cell[nm][cell][1]
                        if cell is not None and hist_cell[nm][cell][1] > 0 else None)
            over_avg = (hist_all[nm][0] / hist_all[nm][1]
                        if hist_all[nm][1] > 0 else None)
            if mode == "overall":
                feat[(s.race_id, car)] = over_avg if over_avg is not None else 0.0
            elif mode == "delta":
                if cell_avg is not None and over_avg is not None:
                    feat[(s.race_id, car)] = cell_avg - over_avg
                else:
                    feat[(s.race_id, car)] = 0.0
            else:  # cell
                if cell_avg is not None:
                    feat[(s.race_id, car)] = cell_avg
                elif over_avg is not None:
                    feat[(s.race_id, car)] = over_avg
                else:
                    feat[(s.race_id, car)] = 4.0
        # 2) このレースの結果で履歴を更新（発走後）。order は上位3車のみなので着順が分かる分だけ。
        for rank, car in enumerate(s.order, start=1):
            nm = name_of.get((s.race_id, car))
            if nm is None:
                continue
            hist_all[nm][0] += rank
            hist_all[nm][1] += 1
            if cell is not None:
                hist_cell[nm][cell][0] += rank
                hist_cell[nm][cell][1] += 1
    return feat


def session_of_by_id(_):  # placeholder (未使用; build_asof_feature 内で deadline 経由に切替)
    return None


def augment_samples(samples, feat, name):
    """samples の各 X に feat 列を1本追加した新しいサンプル列を返す（元は破壊しない）。"""
    import copy
    out = []
    for s in samples:
        col = np.array([[feat.get((s.race_id, car), 4.0)] for car in s.car_numbers], dtype=float)
        new = copy.copy(s)
        new.X = np.hstack([s.X, col])
        new.feature_names = list(s.feature_names) + [name]
        out.append(new)
    return out


def eval_model(train, test):
    model = train_pl(train)
    return evaluate(lambda X, cars: model.strengths(X, cars), test)


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("開催時間帯(セッション) × 季節 の着順への効き — 参考分析")
    print(f"DB: {DB_PATH}")
    print("=" * 78)

    races, results = load_rows()
    print(f"\n総レース数(=ガールズ全て): {len(races)}  / 結果行: {len(results)}")

    # ---- 1 & 2 ----
    d = analyze_distribution_and_diffs(races, results)

    print("\n" + "=" * 78)
    print("【1・2】 分布と成績差")
    print("=" * 78)
    print(fmt_agg_table("セッション別", SESSIONS, SESSION_JP,
                        d["race_by_session"], d["by_session"], d["overall"]))
    print(fmt_agg_table("季節別", SEASONS, SEASON_JP,
                        d["race_by_season"], d["by_season"], d["overall"]))

    print("\n### 選手内ばらつき（同一選手を固定して見た区分差）")
    print("  各選手の『区分別 平均着順』のレンジ(最大-最小)と標準偏差を全選手平均。")
    print("  区分内 min10走・2区分以上を持つ選手が対象。区分の効果が大きいほどレンジが大きい。")
    for label, pm, keys in (("セッション", d["rs_pos"], SESSIONS),
                            ("季節", d["ry_pos"], SEASONS)):
        w = within_rider_spread(pm, keys, None)
        if w:
            print(f"  [{label}] 対象選手={w['n_riders']}  "
                  f"平均レンジ={w['mean_range']:.3f}着  平均std={w['mean_std']:.3f}")
        else:
            print(f"  [{label}] 対象選手が不足")

    # ---- 3 ----
    print("\n" + "=" * 78)
    print("【3】 as-of 特徴の信号チェック（time_split・リーク無し）")
    print("=" * 78)
    print("ベース特徴: PL_FEATURES_FULL  / field_size=7")
    samples = load_samples(DB_PATH, field_size=7, features=PL_FEATURES_FULL)
    print(f"学習可能サンプル数: {len(samples)}")
    train, test = time_split(samples, test_frac=0.2)
    print(f"train={len(train)}  test={len(test)}")

    base = eval_model(train, test)
    print(f"\n[baseline]              {base}")

    # 各 as-of 特徴を構築（cell=区分特化, overall=選手実力の対照, delta=区分ずれのみ）
    f_sess = build_asof_feature(samples, "session", "cell")
    f_seas = build_asof_feature(samples, "season", "cell")
    f_overall = build_asof_feature(samples, "session", "overall")   # kindは無関係(区分無視)
    f_dsess = build_asof_feature(samples, "session", "delta")
    f_dseas = build_asof_feature(samples, "season", "delta")

    def run(feats):
        aug = samples
        for feat, nm in feats:
            aug = augment_samples(aug, feat, nm)
        tr, te = time_split(aug, 0.2)
        return eval_model(tr, te)

    r_overall = run([(f_overall, "asof_overall_pos")])
    r_sess = run([(f_sess, "asof_session_pos")])
    r_seas = run([(f_seas, "asof_season_pos")])
    r_both = run([(f_sess, "asof_session_pos"), (f_seas, "asof_season_pos")])
    # 対照: 選手実力(overall)を先に入れた上で 区分ずれ(delta) が“上乗せ”で効くか
    r_ov_dsess = run([(f_overall, "asof_overall_pos"), (f_dsess, "asof_dsession")])
    r_ov_dseas = run([(f_overall, "asof_overall_pos"), (f_dseas, "asof_dseason")])

    print(f"[+ 選手実力(overall)]   {r_overall}   ← 対照群(区分非依存の選手実力)")
    print(f"[+ 選手×セッション]     {r_sess}")
    print(f"[+ 選手×季節]           {r_seas}")
    print(f"[+ 両方(cell)]          {r_both}")
    print(f"[+ overall+Δseason]     {r_ov_dseas}  ← 実力込みで季節ずれの純寄与")
    print(f"[+ overall+Δsession]    {r_ov_dsess}  ← 実力込みでセッションずれの純寄与")

    def delta(a, b, key):
        return b.get(key, float('nan')) - a.get(key, float('nan'))

    print("\n### baseline比 改善幅（負=改善: logloss/ece/brier, 正=改善: top1_acc）")
    rows = (("overall(対照)", r_overall), ("session", r_sess), ("season", r_seas),
            ("both", r_both), ("ov+Δsession", r_ov_dsess), ("ov+Δseason", r_ov_dseas))
    for label, r in rows:
        print(f"  {label:<14} d_logloss={delta(base, r, 'logloss'):+.4f}  "
              f"d_ece={delta(base, r, 'ece'):+.5f}  "
              f"d_brier={delta(base, r, 'brier'):+.5f}  "
              f"d_top1={delta(base, r, 'top1_acc'):+.4f}")
    print("\n※ session/season が overall(選手実力) を超えて改善するか、"
          "および Δ(区分ずれ) が overall の上に上乗せ改善するかが判断の要。")

    print("\n分析完了。")


if __name__ == "__main__":
    main()
