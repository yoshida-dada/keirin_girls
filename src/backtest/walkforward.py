"""Walk-forward 多フォールド分割（S5。src/model/evaluate.py: time_split の多fold化）。

time_split は「前train / 後test」の単一分割で、1分割のROIは高分散。本モジュールは
時系列（date昇順）サンプルを複数フォールドに切り、各フォールドで
「過去だけで学習 → 直後ブロックで検証」を回すための（train, test）列を生成する
薄いヘルパー。学習・検証の中身（train_pl / build_records / bucket_roi）は呼び出し側が担う。

  window="expanding": train は先頭から test 直前まで拡大（既定。データを捨てない）。
  window="rolling"  : train は直近 warmup 件の固定幅ウィンドウ（古いレジームを忘れる）。

リーク防止: test は必ず train より未来（インデックス昇順＝date昇順が前提）。
移植思想の出典: ../../競馬予想/backtest.py（as_of 走査）＋ scripts/walk_forward.py（拡大窓）。
"""
from __future__ import annotations

from typing import Iterator, Sequence, TypeVar

T = TypeVar("T")


def fold_boundaries(
    n: int, n_folds: int = 4, warmup_frac: float = 0.40, window: str = "expanding"
) -> list[tuple[int, int, int]]:
    """サンプル数 n を (train_start, train_end, test_end) の境界リストへ分割する。

    test は [train_end, test_end)。最終フォールドは端数を全て test に含める（取りこぼしなし）。
    window="rolling" のとき train_start = train_end - warmup（固定幅）、それ以外は 0（拡大窓）。
    """
    if window not in ("expanding", "rolling"):
        raise ValueError(f"unknown window: {window!r}（'expanding' か 'rolling'）")
    if n <= 0 or n_folds < 1:
        return []
    warmup = max(1, int(n * warmup_frac))
    if warmup >= n:                       # warmup が大きすぎて検証域が残らない
        return []
    block = max(1, (n - warmup) // n_folds)

    out: list[tuple[int, int, int]] = []
    for i in range(n_folds):
        train_end = warmup + i * block
        test_start = train_end
        test_end = (warmup + (i + 1) * block) if i < n_folds - 1 else n
        if test_start >= n:               # これ以上テストブロックが取れない
            break
        test_end = min(test_end, n)
        if test_end <= test_start:
            break
        train_start = max(0, train_end - warmup) if window == "rolling" else 0
        out.append((train_start, train_end, test_end))
    return out


def walk_forward_folds(
    samples: Sequence[T],
    n_folds: int = 4,
    warmup_frac: float = 0.40,
    window: str = "expanding",
) -> list[tuple[list[T], list[T]]]:
    """date昇順サンプルを [(train, test), ...] に分割して返す（各 test は train より未来）。

    引数は fold_boundaries と同じ。学習/検証の実処理は呼び出し側で行う薄いヘルパー。
    """
    n = len(samples)
    folds: list[tuple[list[T], list[T]]] = []
    for tr_start, tr_end, te_end in fold_boundaries(n, n_folds, warmup_frac, window):
        train = list(samples[tr_start:tr_end])
        test = list(samples[tr_end:te_end])
        if test:
            folds.append((train, test))
    return folds


def iter_walk_forward(
    samples: Sequence[T],
    n_folds: int = 4,
    warmup_frac: float = 0.40,
    window: str = "expanding",
) -> Iterator[tuple[int, list[T], list[T]]]:
    """walk_forward_folds を (fold_index, train, test) で逐次yieldするジェネレータ版。"""
    for i, (train, test) in enumerate(
        walk_forward_folds(samples, n_folds, warmup_frac, window)
    ):
        yield i, train, test


if __name__ == "__main__":
    # 自己テスト: 境界の妥当性（未来性・端数吸収・rolling固定幅）を確認。
    N = 100
    data = list(range(N))

    exp = walk_forward_folds(data, n_folds=4, warmup_frac=0.40)
    print(f"[expanding] folds={len(exp)}")
    for i, (tr, te) in enumerate(exp):
        print(f"  fold{i}: train[{tr[0]}..{tr[-1]}]({len(tr)}) test[{te[0]}..{te[-1]}]({len(te)})")
        assert te[0] > tr[-1], "test は train より未来でなければならない（リーク防止）"
        assert tr[0] == 0, "expanding は先頭から拡大"
    assert exp[-1][1][-1] == N - 1, "最終フォールドは末尾まで検証（端数吸収）"
    assert len(exp) == 4

    roll = walk_forward_folds(data, n_folds=4, warmup_frac=0.40, window="rolling")
    print(f"[rolling] folds={len(roll)}")
    widths = {len(tr) for tr, _ in roll}
    for i, (tr, te) in enumerate(roll):
        print(f"  fold{i}: train[{tr[0]}..{tr[-1]}]({len(tr)}) test[{te[0]}..{te[-1]}]({len(te)})")
        assert te[0] > tr[-1]
    assert widths == {40}, f"rolling の train 幅は warmup=40 で固定のはず: {widths}"

    # 縮退ケース
    assert walk_forward_folds([], 4) == []
    assert fold_boundaries(10, n_folds=4, warmup_frac=1.0) == []  # warmup>=n で検証域が残らない
    print("OK: walkforward self-test passed")
