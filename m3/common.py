"""Shared paths and helpers (standard library only, so both virtual environments can import it)."""
import gzip
import hashlib
import json
import os
import random
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA_M3 = DATA / "m3"
RESULTS = ROOT / "results" / "m3"
EVIDENCE_M3 = ROOT / "evidence" / "m3"
ARTIFACTS = ROOT / "artifacts"

SEED = 553
EXTRA_SEEDS = (554, 555)
PII_TYPES = ("PERSON", "EMAIL", "PHONE", "ADDRESS")
PLACEMENTS = ("prefix", "suffix")
METHODS = ("delete", "generic", "typed")
REPLACEMENT = {"delete": "", "generic": "[REDACTED]"}  # typed -> "[" + TYPE + "]"


def norm(text):
    """Same normalization as prepare_data.norm (NFC + whitespace collapse)."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def replacement(method, pii_type):
    return REPLACEMENT[method] if method in REPLACEMENT else "[" + pii_type + "]"


def read_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        # mtime=0 keeps the gzip bytes reproducible
        with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
            for r in rows:
                gz.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
    else:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def config_hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_injections():
    return {r["record_id"]: r for r in read_jsonl(DATA / "pii_injections.jsonl")}


def load_splits():
    """{record_id: 'train'|'val'|'test'} for the 13,072 modeled records."""
    return read_json(DATA_M3 / "splits.json")["assignment"]
