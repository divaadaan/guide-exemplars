"""Generate the synthetic demo set: made-up grade 7 essays on the ASAP-AES prompt 7 topic (patience).

Each essay is written by gpt-4o-mini to a *target* Conventions score (0-3). The label is that target, not a human
rating, so the set is for demos and screenshots only; real results come from the ASAP benchmark runs. Content
quality (ideas, organization) is drawn independently of the conventions level so a grader cannot score the
trait from content alone.

    uv run --no-sync python scripts/make_demo_data.py            # writes configs/demo-data/patience-synthetic.jsonl
    uv run --no-sync python scripts/make_demo_data.py --dry-run  # print three sample plans and the cost estimate

Needs OPENAI_API_KEY (read from .env). Costs about $0.05 for the default 150 essays and stops at --max-usd.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "configs" / "demo-data" / "patience-synthetic.jsonl"
MODEL = "gpt-4o-mini"
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6  # USD per token, from configs/pricing.yaml

COUNTS = {0: 20, 1: 40, 2: 55, 3: 35}  # skewed toward 2 like the real data, but with enough 0s to see

PROMPT = (
    "Write about patience. Being patient means that you are understanding and tolerant. A patient person "
    "experience difficulties without complaining. Do only one of the following: write a story about a time when "
    "you were patient OR write a story about a time when someone you know was patient OR write a story in your "
    "own way about patience."
)

PERSPECTIVES = [
    "a story about a time the writer was patient",
    "a story about a time someone the writer knows was patient",
    "a story in the writer's own way about patience (it may be invented or told in third person)",
]

SCENARIOS = [
    "waiting weeks for a rescue dog to be ready for adoption",
    "learning to fish with a grandparent who never rushes",
    "practicing a hard song on the guitar for a talent show",
    "a flight delayed for nine hours at an airport",
    "a younger sibling in the hospital and the family waiting for news",
    "standing in a long line for a new roller coaster",
    "teaching a little brother to ride a bike",
    "a parent calmly helping with math homework night after night",
    "growing tomatoes from seeds in a small garden",
    "waiting for a broken arm to heal during soccer season",
    "a long car trip with a carsick cousin",
    "saving allowance for months to buy a bike",
    "a teacher patiently helping a new student who speaks little English",
    "waiting for a turn at the doctor's office with a crying baby nearby",
    "building a model ship that kept falling apart",
    "a grandmother learning to use a smartphone",
    "waiting out a snowstorm with no power",
    "training a stubborn puppy to sit",
    "a friend who waited to make the basketball team for two years",
    "a coach who stayed calm when the team kept losing",
    "baking bread that had to rise for hours",
    "waiting for a pen pal's letter from another country",
    "a parent stuck in traffic on the way to an important game",
    "a kitten that hid under the bed for days after coming home",
    "a cashier dealing politely with a rude customer",
    "learning to swim after being afraid of the water",
    "a long wait at the DMV with a parent",
    "a bus that broke down on a school field trip",
    "a neighbor rebuilding a fence after a storm",
    "an older sister helping with a science fair project that failed twice",
]

IDEAS = [
    "strong: a focused story with specific, vivid details and a clear point about patience",
    "average: a mostly focused story with a mix of specific and general details",
    "weak: a thin story that wanders a little and uses general details",
]

LEVELS = {
    3: (
        "Conventions score 3 of 3 (consistent, appropriate for grade 7). At most one or two minor slips in the "
        "whole essay. Dialogue, commas and apostrophes are handled correctly. Sentences vary and are correctly "
        "punctuated. It should still sound like a capable 12-year-old, not an adult writer.",
        (180, 380),
    ),
    2: (
        "Conventions score 2 of 3 (adequate for grade 7). Roughly one error every 30-40 words: a few misspellings "
        "of harder words, missing or extra commas, an occasional run-on or fragment, a capitalization slip, a "
        "missing apostrophe. Meaning is always clear.",
        (150, 330),
    ),
    1: (
        "Conventions score 1 of 3 (limited for grade 7). Errors are frequent, about one every 10-15 words: "
        "phonetic misspellings of common words, run-on sentences joined with 'and' or commas, lowercase 'i', "
        "missing apostrophes and end punctuation, tense switches, wrong homophones (there/their, to/too). "
        "Some sentences take effort to follow.",
        (100, 250),
    ),
    0: (
        "Conventions score 0 of 3 (ineffective). Errors are pervasive and severe: almost no punctuation or "
        "sentence boundaries, heavy phonetic spelling, random capitalization, missing words. The meaning is often "
        "hard to recover, though the topic is recognizable.",
        (60, 180),
    ),
}

SYSTEM = (
    "You write realistic practice essays that imitate real grade 7 student writing for a rubric-training set. "
    "Follow the conventions level exactly: the errors must be the kind a real student makes, spread naturally "
    "through the text, never flagged or explained. Use anonymization tokens the way the public ASAP essays do: "
    "@PERSON1, @PERSON2 for names, @LOCATION1 for places, and occasionally @DATE1, @NUM1 or @CAPS1. Output only "
    "the essay text: no title, no notes."
)


def plans(seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for level, n in COUNTS.items():
        for _ in range(n):
            lo, hi = LEVELS[level][1]
            out.append(
                {
                    "label": level,
                    "perspective": rng.choice(PERSPECTIVES),
                    "scenario": rng.choice(SCENARIOS),
                    "ideas": rng.choice(IDEAS),
                    "words": rng.randint(lo, hi),
                }
            )
    rng.shuffle(out)
    for i, p in enumerate(out, start=1):
        p["item_id"] = f"demo-{i:04d}"
    return out


def user_message(p: dict) -> str:
    return (
        f"Assignment the student is answering:\n{PROMPT}\n\n"
        f"Write {p['perspective']}, about {p['scenario']}. About {p['words']} words.\n"
        f"Ideas and organization: {p['ideas']}.\n"
        f"{LEVELS[p['label']][0]}"
    )


async def generate(plan: list[dict], max_usd: float, concurrency: int) -> list[dict]:
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    load_dotenv(REPO / ".env")
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    spent = 0.0
    rows: list[dict] = []

    async def one(i: int, p: dict) -> None:
        nonlocal spent
        async with sem:
            if spent > max_usd:
                raise RuntimeError(f"spent ${spent:.4f}, over --max-usd {max_usd}")
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message(p)}],
                temperature=1.0,
                seed=i,
                max_tokens=900,
            )
        u = resp.usage
        spent += u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT
        text = (resp.choices[0].message.content or "").strip()
        rows.append({"item_id": p["item_id"], "text": text, "label": p["label"], "scenario": p["scenario"]})
        if len(rows) % 25 == 0:
            print(f"  {len(rows)}/{len(plan)}  ${spent:.4f}", file=sys.stderr)

    await asyncio.gather(*(one(i, p) for i, p in enumerate(plan)))
    print(f"done: {len(rows)} essays, ${spent:.4f}", file=sys.stderr)
    return sorted(rows, key=lambda r: r["item_id"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-usd", type=float, default=0.25)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    plan = plans(args.seed)
    if args.dry_run:
        for p in plan[:3]:
            print(user_message(p), end="\n\n---\n\n")
        n_out = sum(p["words"] for p in plan) * 1.4
        est = len(plan) * 450 * PRICE_IN + n_out * PRICE_OUT
        print(f"{len(plan)} essays, labels {COUNTS}, est. ${est:.3f}")
        return

    rows = asyncio.run(generate(plan, args.max_usd, args.concurrency))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
