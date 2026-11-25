"""Download raw datasets with the Kaggle CLI.

Both ASAP datasets are Kaggle competitions. You need a Kaggle account, an API token
(``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``), and you must accept the competition rules on
the competition page once. Nothing downloaded here is ever committed.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from guide_exemplars.paths import DATA_DIR

DATASETS = {
    "asap-aes": {"competition": "asap-aes", "expect": "training_set_rel3.tsv"},
    "asap-sas": {"competition": "asap-sas", "expect": "train.tsv"},
}

MANUAL_HELP = (
    "Download it from https://www.kaggle.com/c/{competition}/data (accept the rules first) and put "
    "{expect} in {dest}"
)


def fetch(name: str, raw_dir: Path | None = None) -> Path:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    info = DATASETS[name]
    dest = (raw_dir or DATA_DIR / "raw") / name
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / info["expect"]
    if target.exists():
        return target
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI not found (`pip install kaggle`). " + MANUAL_HELP.format(dest=dest, **info)
        )
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", info["competition"], "-p", str(dest)], check=True
    )
    for archive in dest.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if Path(member).name == info["expect"] or member.endswith(".zip"):
                    zf.extract(member, dest)
    for nested in dest.rglob("*.zip"):
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(dest)
    found = next(dest.rglob(info["expect"]), None)
    if found is None:
        raise RuntimeError(
            f"download finished but {info['expect']} not found. " + MANUAL_HELP.format(dest=dest, **info)
        )
    if found != target:
        found.replace(target)
    return target
