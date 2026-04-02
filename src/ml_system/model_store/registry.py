from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ModelRegistry:
    """File-backed registry: production pointer and version list."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "registry.json"

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {"production": None, "versions": []}
        return json.loads(self._path.read_text())

    def _write(self, d: dict[str, Any]) -> None:
        self._path.write_text(json.dumps(d, indent=2))

    def register(self, model_version: str, *, promote_production: bool = False) -> None:
        d = self._read()
        if model_version not in d["versions"]:
            d["versions"].append(model_version)
        if promote_production:
            d["production"] = model_version
        self._write(d)

    def get_production_version(self) -> str | None:
        return self._read().get("production")

    def set_production(self, model_version: str) -> None:
        d = self._read()
        if model_version not in d["versions"]:
            d["versions"].append(model_version)
        d["production"] = model_version
        self._write(d)
