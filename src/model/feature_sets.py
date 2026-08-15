"""男女それぞれのモデルが使う特徴セットの定義（唯一の正）。

**ガールズと男子は特徴が完全に異なる。** ライン概念の有無・級班階層・車立て・決まり手構造が
違うため、同一モデルにフラグを足す形ではなくモデルを分ける。コードは共有し、
「どの列を使うか」だけをここで分ける。

  ガールズ（38列, data/models/pl_model.pkl）
      拡張20 + rel_elo + 展開10 + 並び5 + バンク交互作用2
  男子（39列, data/models/pl_model_men.pkl）
      拡張20 + rel_elo + ライン8 + 展開10

男子で並び5列(NARABI_KEYS)を使わないのは、ライン特徴に包含されるため:
  narabi_pos/narabi_lead/narabi_mid → ln_head/ln_mate/ln_third/ln_solo が上位互換
  narabi_leg（脚質前がかり度）      → 検証で tri10 -0.28pt(0/5) と不採用
ガールズでバンク交互作用を使うのは 2026-07-29 の検証（tri10 +0.66〜0.92pt）による。
男子のバンク交互作用は未検証（`kimarite_stats.json` がガールズ実測なのでそのままでは使えない）。

検証の根拠は men_keirin_plan.md 4.8/4.9節。
"""
from __future__ import annotations

from src.model.training_data import PL_FEATURES_FULL
from src.features.tactics_features import TACTIC_NAMES
from src.features.rider_narabi import NARABI_KEYS
from src.features.bank_features import BANK_KEYS
from src.features.line_features import LINE_KEYS, LEGOH_KEYS, SERI_KEYS


def girls_features() -> list[str]:
    """ガールズ本番モデルの特徴（38列）。現行 pl_model.pkl と一致する。"""
    return (list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
            + list(NARABI_KEYS) + list(BANK_KEYS))


def men_features() -> list[str]:
    """男子本番モデルの特徴（46列）。

    ライン8列が主役（tri10 +4.61pt・5/5fold）。展開10列はその上にも純増する
    （+0.40pt・4/5fold）。並び5列(NARABI_KEYS)は不採用。

    脚質7列(LEGOH_KEYS)は **one-hot で再検証して採用**（tri10 +0.25pt・4/5fold,
    top1 +0.03pt・4/5fold）。以前スカラー化(ln_leg)で不採用にしたのは情報ではなく
    符号化の問題で、LEG_AGGR_MEN が 先行/押え先/カマシ を全て2.0に潰していた
    （実測のライン先頭主導権率は 55.2%/34.7%/33.0% と20pt違う）。
    効果量はライン特徴の1/18と小さいが、事前基準は充足している。

    競り(SERI_KEYS + ライン内位置の共有)は **不採用**（2026-08-15, scripts/validate_seri.py）。
    競りレースの tri10 は 1/5fold 改善・平均 -3.52pt で事前基準の主基準を満たさなかった。
    ただし競りレースはテストfoldあたり35〜43件しかなく、この検証はそもそも検定力が足りない
    （tri10 の1レース = 約2.7pt）。「差が無い」ことを示せたわけではなく「測れなかった」。
    データ自体（narabi.seri_group）は収集済みなので、母数が増えたら再検証する。
    """
    return (list(PL_FEATURES_FULL) + ["rel_elo"] + list(LINE_KEYS)
            + list(TACTIC_NAMES) + list(LEGOH_KEYS))


# 学習・推論の両方から参照する定数
MEN_MODEL_NAME = "pl_model_men.pkl"
MEN_ELO_NAME = "elo_state_men.json"


def load_for(is_girls: bool):
    """レースの種別に応じたモデルとEloを返す → (model, elo_state, label)。

    男子モデルが無い場合はガールズモデルへフォールバックせず (None, None, ...) を返す。
    特徴セットが違うので取り違えると無言で誤った推論になるため、落とす方が安全。
    """
    from pathlib import Path
    from src.model.persist import load_model, load_elo_state, DEFAULT_MODEL_PATH

    if is_girls:
        return load_model(), load_elo_state(), "ガールズ"
    mp = Path(DEFAULT_MODEL_PATH).parent / MEN_MODEL_NAME
    ep = Path(DEFAULT_MODEL_PATH).parent / MEN_ELO_NAME
    if not mp.exists():
        return None, None, "男子(モデル未配置)"
    model = load_model(mp)
    try:
        elo = load_elo_state(ep)
    except Exception:
        elo = {}
    return model, elo, "男子"
