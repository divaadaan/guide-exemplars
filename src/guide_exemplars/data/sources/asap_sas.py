"""ASAP-SAS (Kaggle "asap-sas", ``train.tsv``): short constructed responses. ``Score1`` is the label of record
and ``Score2`` is the second rater, used only to report the human ceiling."""

from __future__ import annotations

from pathlib import Path

from guide_exemplars.data.sources.asap_aes import RawItem, SourceReport, human_agreement


def load(
    path: str | Path,
    essay_set: int,
    score_range: tuple[int, int],
    score_col: str = "Score1",
    second_col: str = "Score2",
) -> tuple[list[RawItem], SourceReport]:
    import pandas as pd

    df = pd.read_csv(path, sep="\t", encoding="latin-1")
    for col in ("Id", "EssaySet", "EssayText", score_col, second_col):
        if col not in df.columns:
            raise ValueError(f"{path} has no column {col!r}; is this ASAP-SAS train.tsv?")
    df = df[df["EssaySet"] == essay_set].dropna(subset=[score_col, "EssayText"])
    lo, hi = score_range
    observed = (int(df[score_col].min()), int(df[score_col].max())) if len(df) else (lo, hi)
    if observed[0] < lo or observed[1] > hi:
        raise ValueError(f"set {essay_set}: scores span {observed}, expected within {score_range}")
    items = [
        RawItem(
            str(row.Id),
            " ".join(str(row.EssayText).split()),
            int(getattr(row, score_col)),
            int(getattr(row, score_col)),
            int(getattr(row, second_col)),
        )
        for row in df.itertuples(index=False)
        if str(row.EssayText).strip()
    ]
    n_labels = hi - lo + 1
    acc, qwk = human_agreement([i.rater1 - lo for i in items], [i.rater2 - lo for i in items], n_labels)
    counts = {str(v): sum(i.label == v for i in items) for v in range(lo, hi + 1)}
    return items, SourceReport(len(df), len(items), len(df) - len(items), counts, acc, qwk)
