from __future__ import annotations

import pytest

from guide_exemplars.core.scale import LabelScale, canonical_label

NAMED = LabelScale(("Beginning", "Developing", "Proficient", "Advanced"))
INT4 = LabelScale.from_spec([0, 1, 2, 3])
NONUNIFORM = LabelScale.from_spec([0, 10, 20, 30])


def test_canonical_label_forms() -> None:
    assert canonical_label(2) == "2"
    assert canonical_label("2") == "2"
    assert canonical_label(" 2.0 ") == "2"
    assert canonical_label(3.0) == "3"
    assert canonical_label("Proficient") == "Proficient"
    with pytest.raises(TypeError):
        canonical_label(True)
    with pytest.raises(ValueError):
        canonical_label(2.5)


def test_named_scale_rank_and_neighbours() -> None:
    assert NAMED.rank("Proficient") == 2
    assert NAMED.label_at(0) == "Beginning"
    assert NAMED.neighbours("Beginning") == ["Developing"]
    assert NAMED.neighbours("Developing") == ["Beginning", "Proficient"]
    assert NAMED.neighbours("Advanced") == ["Proficient"]
    assert NAMED.is_adjacent("Developing", "Proficient")
    assert not NAMED.is_adjacent("Beginning", "Proficient")
    assert NAMED.distance("Beginning", "Advanced") == 3


def test_integer_scale_accepts_int_and_str() -> None:
    assert INT4.rank(2) == INT4.rank("2") == 2
    assert INT4.labels == ("0", "1", "2", "3")
    assert 3 in INT4
    assert 4 not in INT4


def test_nonuniform_scale_adjacency_is_by_rank() -> None:
    assert NONUNIFORM.is_adjacent(10, 20)
    assert NONUNIFORM.distance(0, 30) == 3
    assert not NONUNIFORM.is_adjacent(0, 20)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" proficient.", "Proficient"),
        ('"ADVANCED"', "Advanced"),
        ("Developing!", "Developing"),
        ("Score: Proficient", None),
        ("Expert", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_named(raw: object, expected: str | None) -> None:
    assert NAMED.parse(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("2", "2"), ("2.0", "2"), (" 3 ", "3"), (1, "1"), (2.0, "2"), (2.5, None), ("4", None), (True, None)],
)
def test_parse_integer(raw: object, expected: str | None) -> None:
    assert INT4.parse(raw) == expected


def test_unknown_label_raises() -> None:
    with pytest.raises(ValueError, match="unknown label"):
        NAMED.rank("Expert")
    with pytest.raises(IndexError):
        NAMED.label_at(4)


def test_invalid_scales_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        LabelScale(("only",))
    with pytest.raises(ValueError, match="duplicate"):
        LabelScale.from_spec([0, 1, "1"])
    with pytest.raises(ValueError, match="case"):
        LabelScale(("low", "LOW"))
    with pytest.raises(ValueError, match="align"):
        LabelScale(("a", "b"), ("only one",))


def test_from_spec_with_descriptions() -> None:
    scale = LabelScale.from_spec([{"value": 0, "description": "none"}, {"value": 1}])
    assert scale.description(0) == "none"
    assert scale.description("1") == ""
    assert LabelScale.from_spec([0, 1]).description(0) == ""


def test_scales_are_hashable_and_comparable() -> None:
    assert LabelScale.from_spec([0, 1, 2, 3]) == INT4
    assert hash(LabelScale.from_spec([0, 1, 2, 3])) == hash(INT4)
