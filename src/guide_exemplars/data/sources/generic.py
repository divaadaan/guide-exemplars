"""Bring-your-own data: CSV or JSONL with ``item_id``, ``text``, ``label`` and an optional ``split`` column."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from guide_exemplars.core.types import Item

VALID_SPLITS = {"train", "val", "test", "expert"}


def parse_records(content: str, fmt: str) -> list[dict]:
    if fmt == "jsonl":
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    if fmt == "csv":
        return list(csv.DictReader(io.StringIO(content)))
    raise ValueError(f"unsupported format {fmt!r}; use csv or jsonl")


def items_from_records(records: list[dict]) -> tuple[list[Item], dict[str, str]]:
    """Items plus any explicit ``split`` assignments. Raises on missing columns or duplicate ids."""
    items: list[Item] = []
    splits: dict[str, str] = {}
    seen: set[str] = set()
    for n, rec in enumerate(records, start=1):
        missing = {"item_id", "text", "label"} - rec.keys()
        if missing:
            raise ValueError(f"record {n} missing {sorted(missing)}")
        item = Item(item_id=rec["item_id"], text=str(rec["text"]), label=rec["label"])
        if item.item_id in seen:
            raise ValueError(f"duplicate item_id {item.item_id!r}")
        seen.add(item.item_id)
        items.append(item)
        split = (rec.get("split") or "").strip()
        if split:
            if split not in VALID_SPLITS:
                raise ValueError(f"record {n}: split {split!r} not in {sorted(VALID_SPLITS)}")
            splits[item.item_id] = split
    return items, splits


def load_file(path: str | Path) -> tuple[list[Item], dict[str, str]]:
    path = Path(path)
    fmt = "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson"} else "csv"
    return items_from_records(parse_records(path.read_text(encoding="utf-8"), fmt))
