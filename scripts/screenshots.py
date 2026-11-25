"""Capture the dashboard screenshots used in docs/walkthrough/README.md.

Drives a headless Chrome/Edge over the DevTools protocol (no extra dependencies beyond `websockets`, which ships
with uvicorn[standard]). Start the service first and have a finished job on the bundled `demo` preset:

    uv run --no-sync guide serve                                   # in another terminal
    uv run --no-sync python scripts/screenshots.py --job <job_id>  # writes docs/walkthrough/img/*.png

Without --job it picks the most recent succeeded job. The script refuses to capture a job whose dataset is not the
synthetic demo set, and the grade panel is filled with held-out demo essays, so no real student writing can appear
in a screenshot. `--themes light,dark` also writes dark-theme copies (suffix -dark).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "walkthrough" / "img"
DEMO = REPO / "configs" / "demo-data" / "patience-synthetic.jsonl"
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
]
WIDTH, HEIGHT = 1400, 900


def find_chrome(explicit: str | None) -> str:
    for c in [explicit] if explicit else CHROME_CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return c
    sys.exit("no Chrome/Edge found; pass --chrome PATH")


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


class CDP:
    def __init__(self, ws):
        self.ws = ws
        self.ids = itertools.count(1)

    async def send(self, method: str, **params):
        msg_id = next(self.ids)
        await self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str):
        r = await self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"JS error in {expr[:60]!r}: {r['exceptionDetails'].get('text')}")
        return r.get("result", {}).get("value")

    async def wait_for(self, expr: str, timeout: float = 30, what: str = "") -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if await self.js(expr):
                return
            await asyncio.sleep(0.25)
        raise TimeoutError(f"timed out waiting for {what or expr}")

    async def shot(self, path: Path, selector: str | None = None, pad: int = 0) -> None:
        params: dict = {"format": "png", "captureBeyondViewport": True}
        if selector:
            box = await self.js(
                f"""(() => {{ const el = document.querySelector({json.dumps(selector)});
                if (!el) return null; scrollTo(0, 0);  // keep the sticky header at the top of the document
                const r = el.getBoundingClientRect();
                return {{x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height}}; }})()"""
            )
            if not box:
                raise RuntimeError(f"no element {selector}")
            await asyncio.sleep(0.3)
            params["clip"] = {
                "x": max(box["x"] - pad, 0),
                "y": max(box["y"] - pad, 0),
                "width": box["w"] + 2 * pad,
                "height": box["h"] + 2 * pad,
                "scale": 1,
            }
        data = await self.send("Page.captureScreenshot", **params)
        path.write_bytes(base64.b64decode(data["data"]))
        print(f"  {path.relative_to(REPO)}")


def demo_essays(n: int = 3) -> list[str]:
    """Held-out (test split) demo essays at different levels, flattened to one line each for the grade box."""
    from guide_exemplars.data.presets import Preset, load_prepared, prepare

    preset = Preset.load("demo")
    try:
        prepared = load_prepared(preset)
    except FileNotFoundError:
        prepare(preset)
        prepared = load_prepared(preset)
    picked, seen = [], set()
    for it in sorted(prepared.test, key=lambda it: it.label):
        if it.label not in seen and it.label != prepared.spec.scale.labels[0]:  # skip the rare bottom level
            seen.add(it.label)
            picked.append(" ".join(it.text.split()))
        if len(picked) == n:
            break
    if not picked:
        raise RuntimeError("no demo essays found for the grade panel")
    return picked


async def capture(base: str, job: str, chrome: str, themes: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    essays = demo_essays()
    port = 9333
    profile = tempfile.mkdtemp(prefix="guide-shots-")
    proc = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}",
         f"--user-data-dir={profile}", f"--window-size={WIDTH},{HEIGHT}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(40):
            try:
                targets = get_json(f"http://127.0.0.1:{port}/json")
                break
            except OSError:
                await asyncio.sleep(0.25)
        else:
            raise RuntimeError("Chrome did not open its debugging port")
        page = next(t for t in targets if t["type"] == "page")
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=None) as ws:
            cdp = CDP(ws)
            await cdp.send("Page.enable")
            await cdp.send("Runtime.enable")
            await cdp.send(
                "Emulation.setDeviceMetricsOverride", width=WIDTH, height=HEIGHT, deviceScaleFactor=1, mobile=False
            )
            for theme in themes:
                print(theme)
                sfx = "" if theme == "light" else f"-{theme}"
                await cdp.send(
                    "Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": theme}]
                )
                await cdp.send("Page.navigate", url=f"{base}/#/jobs/{job}")
                await cdp.wait_for("document.readyState === 'complete'", what="page load")
                await cdp.js("location.reload()")  # re-render charts with the emulated theme's colors
                await cdp.wait_for("document.readyState === 'complete'", what="reload")
                await cdp.wait_for(
                    "document.querySelector('#t-status')?.textContent.trim() === 'succeeded'", what="job status"
                )
                await cdp.wait_for(
                    "document.querySelector('#convergence .main-svg') && document.querySelector('#poolmap .main-svg')"
                    " && document.querySelectorAll('#exemplars > *').length > 0",
                    what="charts and exemplars",
                )
                await asyncio.sleep(1.0)  # let Plotly finish transitions
                await cdp.js(
                    "(() => { const s = document.querySelector('#backbone-select');"
                    " if ([...s.options].some(o => o.value === 'gpt4o-mini')) s.value = 'gpt4o-mini'; })()"
                )
                await cdp.js("scrollTo(0, 0)")
                data = await cdp.send("Page.captureScreenshot", format="png")  # the first screenful
                (OUT / f"01-dashboard{sfx}.png").write_bytes(base64.b64decode(data["data"]))
                print(f"  {(OUT / f'01-dashboard{sfx}.png').relative_to(REPO)}")
                await cdp.shot(OUT / f"02-new-job{sfx}.png", "section.panel:has(#job-form)")
                await cdp.shot(OUT / f"03-search-progress{sfx}.png", "section.card:has(#convergence)")
                await cdp.shot(OUT / f"04-pool-map{sfx}.png", "section.card:has(#poolmap)")
                await cdp.shot(OUT / f"05-exemplar-set{sfx}.png", "section.card:has(#exemplars)")
                await cdp.js(
                    f"""(() => {{ const t = document.querySelector('#grade-input');
                    t.value = {json.dumps(chr(10).join(essays))};
                    document.querySelector('#grade-form').requestSubmit(); }})()"""
                )
                await cdp.wait_for(
                    f"document.querySelectorAll('#grade-table tbody tr').length >= {len(essays)}",
                    timeout=120, what="grade results",
                )
                await cdp.shot(OUT / f"06-grade{sfx}.png", "section.card:has(#grade-form)")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(profile, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--job", help="job id; default: most recent succeeded job")
    ap.add_argument("--chrome", help="path to Chrome or Edge")
    ap.add_argument("--themes", default="light")
    args = ap.parse_args()

    job = args.job
    if not job:
        done = [j for j in get_json(f"{args.base}/api/jobs") if j.get("status") == "succeeded"]
        if not done:
            sys.exit("no succeeded job; run one first or pass --job")
        job = max(done, key=lambda j: j.get("created_at") or 0)["id"]
    dataset = get_json(f"{args.base}/api/datasets/{get_json(f'{args.base}/api/jobs/{job}')['dataset_id']}")
    if not str(dataset.get("name", "")).startswith("demo/"):
        sys.exit(
            f"refusing to capture job {job}: its dataset {dataset.get('name')!r} is not the synthetic demo set, "
            "and screenshots must never show real student essays"
        )
    asyncio.run(capture(args.base, job, find_chrome(args.chrome), args.themes.split(",")))


if __name__ == "__main__":
    main()
