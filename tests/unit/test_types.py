from __future__ import annotations

from guide_exemplars.core.types import Exemplar, Item, Prediction, Subset


def _ex(item_id: str = "a", rationale: str = "Because.", **kw: object) -> Exemplar:
    return Exemplar(item_id=item_id, text="text", label="1", rationale=rationale, **kw)


def test_item_coerces_id_and_label() -> None:
    item = Item(item_id=7, text="t", label=2)  # type: ignore[arg-type]
    assert item.item_id == "7"
    assert item.label == "2"


def test_uid_distinguishes_rationale_variants_of_one_item() -> None:
    a, b = _ex(rationale="Clear errors."), _ex(rationale="Few errors.")
    assert a.uid != b.uid
    assert a.uid.startswith("a#") and b.uid.startswith("a#")


def test_uid_ignores_whitespace_and_case() -> None:
    assert _ex(rationale="Clear  errors.\n").uid == _ex(rationale="clear errors.").uid


def test_uid_independent_of_metadata() -> None:
    assert _ex(origin="phase0", round=0).uid == _ex(origin="phase2", round=3).uid


def test_exemplar_roundtrip_through_dump() -> None:
    ex = _ex(is_expert=True)
    dumped = ex.model_dump(mode="json")
    assert dumped["uid"] == ex.uid
    assert Exemplar.model_validate(dumped) == ex


def test_from_item() -> None:
    item = Item(item_id="x", text="hello", label="3")
    ex = Exemplar.from_item(item, "r", origin="given")
    assert (ex.item_id, ex.text, ex.label, ex.origin) == ("x", "hello", "3", "given")


def test_subset_key_is_order_independent() -> None:
    s1, s2 = Subset.of(["b#1", "a#2", "c#3"]), Subset.of(["c#3", "b#1", "a#2"])
    assert s1 == s2
    assert s1.key() == s2.key()
    assert list(s1) == ["a#2", "b#1", "c#3"]
    assert Subset.of(["a#2"]).key() != s1.key()
    assert "a#2" in s1 and len(s1) == 3


def test_prediction_flags() -> None:
    assert Prediction("i", "1", "1").correct
    assert not Prediction("i", "1", "2").correct
    failed = Prediction("i", "1", None, error="parse")
    assert failed.failed and not failed.correct
