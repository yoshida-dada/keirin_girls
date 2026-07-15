"""レース展開特徴（S2拡張・as-of＝リーク無し）。

rider_tactics.compute_pre_race_tactics が返す「選手個人の絶対値」展開指数
（lead_index / lead_index_sb / sikake / avg_last_lap / escape_survival / leg_change_rate）を、
**各レースの出走者集合の中で相対化・交互作用**して「レースの展開そのもの」を数値化する。

★設計方針（Sub-1 の最重要指摘）:
    選手個人の絶対値は、レース内シフト不変のランキングモデル（PL線形の softmax / lambdarank）
    には直接効かない。「主導権を取れるか」「想定ペース」「逃げ切れるか」は本質的に
    **レース内の相対量**なので、必ず (a) レース内平均を引く / (b) レース内最大との差 /
    (c) 他車との交互作用 のいずれかへ変換する。

    ※ 注意: pace_mean / pace_max / pace_n600 / pace_std / lead_contest は「レース定数」
      （同一レースの全車で同値）。これらは softmax／group内ランキングに対しては
      シフト不変で**寄与しない**（Sub-1 の核心指摘そのもの）。透明性・解釈用として per-car に
      展開して返すが、学習に足しても順位付けは変わらないため、compare_tactics.py の
      ablation では「レース内で変動する列」のみをモデルに投入する（レース定数列は除外）。

as-of / リーク:
    rider_tactics が既に as-of（当該レースを含まない）なので、その相対化である本モジュールも
    as-of。学習 train / 評価 test の分割は呼び出し側（compare_tactics.py）の責務。

返す各値のキー（(race_id, car_number) ごと）:
    lead_margin    : lead_index − レース内「他車の最大 lead_index」。
                     先頭候補の突出度。首位車は +（2番手との差＝どれだけ楽に主導権）、
                     それ以外は −（首位車にどれだけ届かないか）。★レース内変動・主要相対量。
    lead_contest   : 首位 lead_index − 2番手 lead_index（主導権争い指数、小=接戦=混戦寄り）。
                     【レース定数】全車同値。解釈用。
    leadidx_rel    : lead_index − レース内平均 lead_index。★レース内変動。
    sikake_rel     : sikake − レース内平均 sikake（自分は平均より前で仕掛けるか）。★レース内変動。
    pace_mean      : レース内 sikake 平均（想定ペース水準）。【レース定数】
    pace_max       : レース内 sikake 最大（最も前で仕掛ける車の激しさ）。【レース定数】
    pace_n600      : sikake≥600（逃げ・捲り主体）の人数。【レース定数】
    pace_std       : レース内 sikake の標準偏差（脚質のばらつき＝隊列の乱れやすさ）。【レース定数】
    escape_success : 逃げ成功確率(flagship)。各車を先頭候補とみなし
                     escape_survival × (1 − 他車の最大攻撃力)。★レース内変動・交互作用。
    last_lap_rel   : avg_last_lap − レース内平均（負=上がりが速い＝相対的に脚がある）。
                     ★レース内変動。レース内相対化により周長(バンク)水準は自動で相殺される
                     （同一レースは同一バンク）ので追加のバンク正規化は不要。
    escape_survival: rider_tactics の逃げ残率（絶対値・診断用に素通し）。

欠損の埋め方:
    rider_tactics が None を返す列（lead_index/sikake ~13%, leg_change ~11% 等）は、
    そのレースの present 値の平均で埋める（eval_rolling._rel_column と同じ穏当な作法。
    相対化後は 0＝レース平均扱い）。present が皆無なら 0。escape_survival は常に値がある。
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.features.rider_tactics import compute_pre_race_tactics

# 「捲り力(攻撃力)」の近似に lead_index(逃げ率+0.3捲り率)を用いる。前を積極的に取りに行く
# 力の代理。逃げ成功確率の割引で、後方（他車）の最大攻撃力として集計する。
_SIKAKE_FRONT = 600.0   # sikake がこの値以上＝逃げ・捲り主体（前受け系）とみなすしきい値


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def _fill_race(raw: list[float | None]) -> list[float]:
    """None を present 値の平均で埋める（present 皆無なら 0）。"""
    present = [v for v in raw if v is not None]
    m = _mean(present)
    return [v if v is not None else m for v in raw]


def dynamics_for_cars(cars: list[int], t_list: list[dict]) -> dict[int, dict]:
    """1レースの出走者（cars と、その並び順の rider_tactics 値 t_list）から展開特徴を作る純関数。

    学習(compute_pre_race_dynamics)と推論(predict時の単一レース)で**同一ロジック**を共有し、
    train/inference skew を防ぐ。返すキーの定義はモジュール docstring を参照。
    """
    lead = _fill_race([x.get("lead_index") for x in t_list])
    sik = _fill_race([x.get("sikake") for x in t_list])
    lap = _fill_race([x.get("avg_last_lap") for x in t_list])
    surv = _fill_race([x.get("escape_survival") for x in t_list])   # 実質常に present

    lead_mean = _mean(lead)
    sik_mean = _mean(sik)
    lap_mean = _mean(lap)

    srt = sorted(lead, reverse=True)
    lead_contest = (srt[0] - srt[1]) if len(srt) >= 2 else 0.0
    pace_mean = sik_mean
    pace_max = max(sik) if sik else 0.0
    pace_std = _std(sik)
    pace_n600 = float(sum(1 for v in sik if v >= _SIKAKE_FRONT))

    out: dict[int, dict] = {}
    for i, c in enumerate(cars):
        others = [lead[j] for j in range(len(cars)) if j != i]
        max_other = max(others) if others else 0.0
        threat = min(1.0, max(0.0, max_other))       # 攻撃力[~0,1.3]を[0,1]にクランプ
        escape_success = surv[i] * (1.0 - threat)
        out[c] = {
            "lead_margin": lead[i] - max_other,
            "leadidx_rel": lead[i] - lead_mean,
            "sikake_rel": sik[i] - sik_mean,
            "escape_success": escape_success,
            "last_lap_rel": lap[i] - lap_mean,
            "lead_contest": lead_contest,
            "pace_mean": pace_mean, "pace_max": pace_max,
            "pace_n600": pace_n600, "pace_std": pace_std,
            "escape_survival": surv[i],
        }
    return out


def compute_pre_race_dynamics(db_path: str | Path) -> dict[tuple[str, int], dict]:
    """各エントリ (race_id, car_number) の発走前(as-of)レース展開特徴を返す。

    rider_tactics.compute_pre_race_tactics を出走者集合ごとに dynamics_for_cars で相対化する。
    """
    tactics = compute_pre_race_tactics(db_path)
    by_race: dict[str, list[int]] = defaultdict(list)
    for (rid, car) in tactics:
        by_race[rid].append(car)

    out: dict[tuple[str, int], dict] = {}
    for rid, cars in by_race.items():
        cars = sorted(cars)
        per_car = dynamics_for_cars(cars, [tactics[(rid, c)] for c in cars])
        for c, d in per_car.items():
            out[(rid, c)] = d
    return out


if __name__ == "__main__":  # 簡易サニティ: PYTHONIOENCODING=utf-8 python -m src.features.race_dynamics
    import statistics
    from config.settings import DATA_DIR

    feats = compute_pre_race_dynamics(DATA_DIR / "keirin.sqlite")
    print("entries:", len(feats))
    for key in ["lead_margin", "leadidx_rel", "sikake_rel", "escape_success",
                "last_lap_rel", "lead_contest", "pace_mean", "pace_max",
                "pace_n600", "pace_std"]:
        vals = [v[key] for v in feats.values() if v.get(key) is not None]
        if vals:
            print(f"{key:16s} n={len(vals):6d} "
                  f"min={min(vals):8.3f} med={statistics.median(vals):8.3f} max={max(vals):8.3f}")
