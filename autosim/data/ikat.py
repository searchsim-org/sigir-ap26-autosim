import json
from pathlib import Path
from autosim.schemas import Persona


def load_topics(fix: str | Path) -> list[Persona]:
    raw = json.loads((Path(fix) / "topics.json").read_text())
    return [Persona(id=t["id"], attributes=t.get("attributes", {}),
                    info_need=t["info_need"], ptkb=t.get("ptkb", [])) for t in raw]


def load_system_scores(fix: str | Path) -> dict[str, dict[str, float]]:
    return json.loads((Path(fix) / "runs.json").read_text())


def load_official_ranking(fix: str | Path) -> list[str]:
    return json.loads((Path(fix) / "qrels_ranking.json").read_text())["order"]
