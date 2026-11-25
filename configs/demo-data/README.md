# Synthetic demo data

`patience-synthetic.jsonl` holds 150 **made-up** grade 7 essays on the ASAP-AES prompt 7 topic (patience). None of
them is real student writing. Each was written by `gpt-4o-mini` to a target Conventions score (0–3), with ideas and
organization quality drawn independently of that score. The generator is
[`scripts/make_demo_data.py`](../../scripts/make_demo_data.py).

The `label` field is the **generation target**, not a human rating. Use this set to try the tool, the service and the
dashboard without a Kaggle account (`demo` preset). Don't use it to measure grading accuracy; the benchmark results
come from the real ASAP data.

Fields: `item_id`, `text`, `label`, and `scenario` (the story prompt given to the generator).
