# root/code/utils/paths.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

def project_root() -> Path:
    # paths.py is: root/code/utils/paths.py => parents[2] == root/
    return Path(__file__).resolve().parents[2]

@dataclass(frozen=True)
class ProjectPaths:
    dataset_name: str

    @property
    def root(self) -> Path:
        return project_root()

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def figures_dir(self) -> Path:
        # Requirement #4
        return self.root / "results" / "figures" / "integrao" / self.dataset_name

    @property
    def models_dir(self) -> Path:
        # Requirement #5
        return self.root / "models" / "integrao" / self.dataset_name

    @property
    def runs_dir(self) -> Path:
        # Optional: where IntegrAO can drop run artifacts/checkpoints
        return self.root / "results" / "runs" / "integrao" / self.dataset_name

    def ensure(self) -> None:
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
