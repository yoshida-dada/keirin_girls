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
from src.features.line_features import LINE_KEYS


def girls_features() -> list[str]:
    """ガールズ本番モデルの特徴（38列）。現行 pl_model.pkl と一致する。"""
    return (list(PL_FEATURES_FULL) + ["rel_elo"] + list(TACTIC_NAMES)
            + list(NARABI_KEYS) + list(BANK_KEYS))


def men_features() -> list[str]:
    """男子本番モデルの特徴（39列）。

    ライン8列が主役（tri10 +4.61pt・5/5fold）。展開10列はその上にも純増する
    （+0.40pt・4/5fold）。脚質と並び5列は不採用。
    """
    return (list(PL_FEATURES_FULL) + ["rel_elo"] + list(LINE_KEYS)
            + list(TACTIC_NAMES))


# 学習・推論の両方から参照する定数
MEN_MODEL_NAME = "pl_model_men.pkl"
MEN_ELO_NAME = "elo_state_men.json"
