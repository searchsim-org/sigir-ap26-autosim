import json
import random
from pathlib import Path


def load_real_behavior(fix: str | Path) -> list[dict]:
    return json.loads((Path(fix) / "real_behavior.json").read_text())


def split_holdout(records: list[dict], *, frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - frac))
    cal = [records[i] for i in idx[:cut]]
    hold = [records[i] for i in idx[cut:]]
    return cal, hold
