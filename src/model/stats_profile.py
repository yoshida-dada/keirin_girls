"""ガールズ／男子で参照する「実測で焼き固めた統計」を切り替える唯一の場所。

**なぜ要るか**: 展開パターンの分岐比・バンク別決まり手・紐補正パラメータ・参考フォーメーション
の実測値は、すべてガールズのデータで作られている。男子は決まり手構造が真逆で
（男子 差50%/捲30%/逃19% vs ガールズ 逃21%/捲47%/差32%）、ライン決着という別の力学も入る。
ガールズの数字を男子に当てると、確率も「読み」も系統的に誤る。

各モジュールが `Path(__file__).with_name(...)` で統計を直読みしていたため、男子を通した
時点で無言でガールズ値が適用されていた。参照先の決定をここへ集約し、
呼び出し側は `profile(is_girls)` を渡すだけにする。

**未実測のものは None を返す**（ガールズ値へフォールバックしない）。取り違えると
無言で誤るので、出さない方が安全＝表示側は None を見て非表示にする。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class StatsProfile:
    """この性別で使う統計の所在。None は「未実測につき使わない」を意味する。"""

    label: str
    is_girls: bool
    kimarite_stats: Path | None      # バンク別の決まり手実測（bank_profile / kimarite_hint）
    dev_pattern_stats: Path | None   # 展開6パターンの分岐比（dev_patterns）
    himo_params: dict | None         # 紐補正(himo_adjust)のパラメータ。None=適用しない
    pocket_stats: bool               # 参考フォーメーションの過去実測(的中率/回収率)を出せるか

    def path(self, name: str) -> Path | None:
        return getattr(self, name)


GIRLS = StatsProfile(
    label="ガールズ",
    is_girls=True,
    kimarite_stats=_MODEL_DIR / "kimarite_stats.json",        # 6,485R 実測
    dev_pattern_stats=_MODEL_DIR / "dev_pattern_stats.json",  # 5,795R 実測
    himo_params=None,      # 下で DEFAULT_PARAMS を入れる（循環importを避けるため後段で設定）
    pocket_stats=True,     # validate_himo_roi out-of-sample 3,432R
)

MEN = StatsProfile(
    label="男子",
    is_girls=False,
    kimarite_stats=_MODEL_DIR / "kimarite_stats_men.json",        # 25,228R 実測
    dev_pattern_stats=_MODEL_DIR / "dev_pattern_stats_men.json",  # 男子DBで実測
    # 男子実測で再推定済み（A-3, 2026-08-13）。番手は記者の並び予想のライン基準で判定する。
    himo_params=None,      # 下で MEN_PARAMS を入れる
    # 参考フォーメーションの的中率/回収率はガールズのout-of-sample実測。男子では出さない。
    pocket_stats=False,
)


def profile(is_girls: bool) -> StatsProfile:
    return GIRLS if is_girls else MEN


# 循環importを避けるためここで注入する（himo_adjust は stats_profile を参照しない）。
def _init_params() -> None:
    from src.model.himo_adjust import DEFAULT_PARAMS, MEN_PARAMS
    object.__setattr__(GIRLS, "himo_params", DEFAULT_PARAMS)
    object.__setattr__(MEN, "himo_params", MEN_PARAMS)


_init_params()
