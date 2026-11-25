from __future__ import annotations

import json
from pathlib import Path

import pytest

from guide_exemplars.core.scale import LabelScale
from guide_exemplars.core.types import Item
from guide_exemplars.data.presets import Preset, load_prepared, prepare
from guide_exemplars.data.sources import asap_aes, asap_sas, generic
from guide_exemplars.data.splits import carve_experts, limit_split, split_hash, stratified_split

SCALE = LabelScale.from_spec([0, 1, 2, 3])


def items(n: int = 200) -> list[Item]:
    return [Item(item_id=f"i{i:03d}", text=f"text {i}", label=[0, 1, 2, 2, 3][i % 5]) for i in range(n)]


# ---- label policies ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r1", "r2", "policy", "expected"),
    [
        (2, 2, "min_drop_gap2", 2),
        (2, 3, "min_drop_gap2", 2),
        (1, 3, "min_drop_gap2", None),
        (1, 3, "rater1", 1),
        (1, 2, "mean_round", 2),
        (1, 3, "mean_round", 2),
        (1, 2, "agree_only", None),
        (3, 3, "agree_only", 3),
    ],
)
def test_combine(r1: int, r2: int, policy: str, expected: int | None) -> None:
    assert asap_aes.combine(r1, r2, policy) == expected  # type: ignore[arg-type]


def test_unknown_policy() -> None:
    with pytest.raises(ValueError):
        asap_aes.combine(1, 1, "bogus")  # type: ignore[arg-type]


# ---- splits -----------------------------------------------------------------------------------------


def test_stratified_split_ratios_and_determinism() -> None:
    data = items(500)
    a = stratified_split(data, SCALE, (3, 1, 1), seed=0, experts_per_label=1)
    b = stratified_split(data, SCALE, (3, 1, 1), seed=0, experts_per_label=1)
    assert a == b
    assert a != stratified_split(data, SCALE, (3, 1, 1), seed=1, experts_per_label=1)
    counts = {s: sum(v == s for v in a.values()) for s in ("train", "val", "test", "expert")}
    assert counts["expert"] == 4
    assert counts["val"] == pytest.approx(100, abs=4) and counts["test"] == pytest.approx(100, abs=4)
    for label in SCALE.labels:
        group = [it for it in data if it.label == label]
        val_share = sum(a[it.item_id] == "val" for it in group) / len(group)
        assert val_share == pytest.approx(0.2, abs=0.03)


def test_experts_one_per_label_from_train_only() -> None:
    data = items(100)
    splits = stratified_split(data, SCALE, seed=3, experts_per_label=1)
    experts = [it for it in data if splits[it.item_id] == "expert"]
    assert sorted(it.label for it in experts) == ["0", "1", "2", "3"]


def test_carve_experts_keeps_a_training_item_per_label() -> None:
    tiny = [Item(item_id="a", text="a", label=0), Item(item_id="b", text="b", label=1)]
    out = carve_experts(tiny, {"a": "train", "b": "train"}, LabelScale.from_spec([0, 1]), 1, 0)
    assert out == {"a": "train", "b": "train"}


def test_split_hash_changes_with_assignment() -> None:
    data = items(50)
    s = stratified_split(data, SCALE, seed=0)
    flipped = dict(s)
    flipped[data[0].item_id] = "test" if s[data[0].item_id] != "test" else "train"
    assert split_hash(data, s) != split_hash(data, flipped)


def test_limit_split_stratified_and_deterministic() -> None:
    data = items(200)
    a = limit_split(data, 50, seed=0, scale=SCALE)
    assert len(a) == 50 and a == limit_split(data, 50, seed=0, scale=SCALE)
    assert sum(it.label == "2" for it in a) == pytest.approx(20, abs=1)
    assert limit_split(data, None, 0, SCALE) == data


# ---- sources ----------------------------------------------------------------------------------------


def _write_aes(path: Path) -> None:
    rows = ["essay_id\tessay_set\tessay\trater1_trait4\trater2_trait4"]
    for i in range(40):
        r1 = i % 4
        r2 = r1 if i % 3 else min(3, r1 + 1)
        if i == 7:
            r2 = (r1 + 2) % 4 if r1 < 2 else r1 - 2  # a 2-point gap
        rows.append(f"{1000 + i}\t7\tEssay   number {i}\t{r1}\t{r2}")
    rows.append("9999\t3\tother set\t1\t1")
    path.write_text("\n".join(rows) + "\n", encoding="latin-1")


def test_asap_aes_loader(tmp_path: Path) -> None:
    f = tmp_path / "aes.tsv"
    _write_aes(f)
    raw, report = asap_aes.load(f, essay_set=7, trait=4)
    assert report.n_raw == 40 and report.n_dropped == 1 and len(raw) == 39
    assert raw[0].text == "Essay number 0"
    assert 0 < report.human_acc < 1
    with pytest.raises(ValueError, match="no column"):
        asap_aes.load(f, essay_set=7, trait=5)


def test_asap_sas_loader_checks_scale(tmp_path: Path) -> None:
    f = tmp_path / "sas.tsv"
    rows = ["Id\tEssaySet\tScore1\tScore2\tEssayText"] + [
        f"{i}\t3\t{i % 3}\t{i % 3}\tanswer {i}" for i in range(30)
    ]
    f.write_text("\n".join(rows) + "\n", encoding="latin-1")
    raw, report = asap_sas.load(f, essay_set=3, score_range=(0, 2))
    assert len(raw) == 30 and report.human_acc == 1.0
    with pytest.raises(ValueError, match="expected within"):
        asap_sas.load(f, essay_set=3, score_range=(0, 1))


def test_generic_loader_csv_and_jsonl(tmp_path: Path) -> None:
    csv_path = tmp_path / "d.csv"
    csv_path.write_text("item_id,text,label,split\na,hello,1,train\nb,world,2,\n", encoding="utf-8")
    loaded, splits = generic.load_file(csv_path)
    assert [it.label for it in loaded] == ["1", "2"] and splits == {"a": "train"}
    jl = tmp_path / "d.jsonl"
    jl.write_text(json.dumps({"item_id": 1, "text": "t", "label": "Low"}) + "\n", encoding="utf-8")
    assert generic.load_file(jl)[0][0].item_id == "1"
    with pytest.raises(ValueError, match="duplicate"):
        generic.items_from_records([{"item_id": "x", "text": "", "label": 1}] * 2)
    with pytest.raises(ValueError, match="missing"):
        generic.items_from_records([{"item_id": "x"}])
    with pytest.raises(ValueError, match="split"):
        generic.items_from_records([{"item_id": "x", "text": "", "label": 1, "split": "holdout"}])


# ---- presets ----------------------------------------------------------------------------------------


def test_prepare_and_load_roundtrip(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "asap-aes").mkdir(parents=True)
    _write_aes(raw / "asap-aes" / "training_set_rel3.tsv")
    preset = Preset.load("aes-p7")
    [task_dir] = prepare(preset, raw_dir=raw, out_dir=tmp_path / "prepared")
    report = json.loads((task_dir / "report.json").read_text(encoding="utf-8"))
    assert report["n_kept"] == 39 and report["split_counts"]["expert"] == 4
    prepared = load_prepared(preset, prepared_dir=tmp_path / "prepared")
    assert len(prepared.train) + len(prepared.val) + len(prepared.test) + len(prepared.experts) == 39
    small = prepared.limited(train=10, val=4)
    assert len(small.train) == 10 and len(small.val) == 4 and len(small.experts) == 4


def test_shipped_presets_parse() -> None:
    assert Preset.load("aes-p7").task_entry(None).id == "aes-p7"
    sas = Preset.load("sas")
    assert [t.id for t in sas.tasks] == ["sas-1", "sas-5", "sas-3", "sas-7"]
    with pytest.raises(ValueError, match="pick one"):
        sas.task_entry(None)
    with pytest.raises(ValueError, match="no task"):
        sas.task_entry("sas-9")


def test_bundled_demo_preset_prepares_from_configs(tmp_path: Path) -> None:
    preset = Preset.load("demo")
    assert preset.bundled and not Preset.load("aes-p7").bundled
    # raw_dir points nowhere: the demo source lives under configs/, not data/raw
    [task_dir] = prepare(preset, raw_dir=tmp_path / "missing", out_dir=tmp_path / "prepared")
    report = json.loads((task_dir / "report.json").read_text(encoding="utf-8"))
    assert report["n_kept"] == 150 and report["split_counts"]["expert"] == 4
    prepared = load_prepared(preset, prepared_dir=tmp_path / "prepared")
    assert {it.label for it in prepared.items} == {"0", "1", "2", "3"}
    assert all(it.text.strip() for it in prepared.items)
