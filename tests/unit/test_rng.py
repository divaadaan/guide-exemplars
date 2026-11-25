from __future__ import annotations

from guide_exemplars.core.rng import RngStreams


def test_same_seed_and_path_reproduce() -> None:
    a = RngStreams(3).get("round", 1, "ops").random(5)
    b = RngStreams(3).get("round", 1, "ops").random(5)
    assert (a == b).all()


def test_paths_and_seeds_are_independent() -> None:
    base = RngStreams(3).get("round", 1, "ops").random(5)
    assert not (base == RngStreams(3).get("round", 2, "ops").random(5)).all()
    assert not (base == RngStreams(4).get("round", 1, "ops").random(5)).all()


def test_drawing_from_one_stream_does_not_shift_another() -> None:
    streams = RngStreams(11)
    expected = RngStreams(11).get("cap", 2).random(3)
    streams.get("bo", 2, 0).random(1000)
    assert (streams.get("cap", 2).random(3) == expected).all()


def test_int_seed_deterministic_and_in_range() -> None:
    s = RngStreams(5)
    assert s.int_seed("gp", 1) == RngStreams(5).int_seed("gp", 1)
    assert 0 <= s.int_seed("gp", 1) < 2**31 - 1
    assert s.int_seed("gp", 1) != s.int_seed("gp", 2)
