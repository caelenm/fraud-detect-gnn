"""Pipeline stages, in order. `run.py` and `scripts/NN_*.py` both call these.

Each stage reads its inputs from data/ or outputs/ (under the active
dataset's subfolder) and writes its outputs there, so any stage can be rerun
on its own. New stages (graph features, GNNs, evaluation) are added to STAGES
as they are implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fraud_detect.config import Paths, dataset_config
from fraud_detect.runstate import CHECKPOINT_DIR, UnitStore


class StageError(RuntimeError):
    """A stage cannot run; the message says what to do."""


@dataclass(frozen=True)
class Context:
    config: dict[str, Any]
    paths: Paths
    run_dir: Path
    # True when this stage was interrupted earlier and is being continued with
    # `run.py --resume`: keep the units it already finished.
    resume: bool = False

    @property
    def seed(self) -> int:
        return int(self.config["seed"])

    @property
    def dataset(self) -> str:
        return str(self.config["dataset"])

    @property
    def data(self) -> dict[str, Any]:
        """The active dataset's config section."""
        return dataset_config(self.config)

    def unit_store(self, stage: str) -> UnitStore:
        """Per-unit checkpoints for a long stage (see fraud_detect.runstate).
        Emptied on a fresh run, kept on --resume."""
        store = UnitStore(self.paths.outputs_dir / CHECKPOINT_DIR / stage)
        if not self.resume:
            store.reset()
        return store


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    run: Callable[[Context], None]
    outputs: Callable[[Paths], list[Path]]


def _require(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            raise StageError(f"Missing input {p}. Run the earlier stages first.")


def _not_implemented(milestone: str) -> Callable[[Context], None]:
    def run(ctx: Context) -> None:
        raise StageError(
            f"This stage is not implemented yet (docs/RESEARCH_PLAN.md, {milestone})."
        )

    return run


STAGES: list[Stage] = [
    Stage("download", "Fetch and checksum the CARE-GNN .mat file",
          _not_implemented("M3"), lambda p: [p.mat]),
    Stage("load", "Node table, relation edge lists, count checks",
          _not_implemented("M3"), lambda p: [p.nodes, p.graph_manifest]),
    Stage("split", "Grouped stratified train/test split and CV folds",
          _not_implemented("M4"),
          lambda p: [p.train_ids, p.test_ids, p.cv_folds, p.split_summary]),
    Stage("features", "Own-feature block and column types (training nodes)",
          _not_implemented("M4"), lambda p: [p.block_file("own"), p.column_spec]),
    Stage("audit", "Counts, feature statistics, duplicates, homophily",
          _not_implemented("M4"), lambda p: [p.reports_dir / "audit.md"]),
    Stage("df_analyze_input", "df-analyze train/test tables for the feature set",
          _not_implemented("M5"),
          lambda p: [p.df_analyze_input_dir / "train.parquet",
                     p.df_analyze_input_dir / "test.parquet"]),
    Stage("df_analyze", "Run df-analyze (Model A); verify split and types",
          _not_implemented("M6"), lambda p: [p.model_reports_dir / "split_check.json"]),
    Stage("select_model", "Score every tuned model on the grouped CV folds",
          _not_implemented("M6"),
          lambda p: [p.model_reports_dir / "model_selection_cv.csv"]),
    Stage("df_analyze_report", "Model A test metrics, threshold, bootstrap CIs",
          _not_implemented("M7"), lambda p: [p.model_reports_dir / "model_a.json"]),
    Stage("web_report", "HTML report: parameters, results, confidence cards",
          _not_implemented("M7"), lambda p: [p.web_report]),
]  # fmt: skip

STAGE_NAMES = [s.name for s in STAGES]


def get_stage(name: str) -> Stage:
    for stage in STAGES:
        if stage.name == name:
            return stage
    raise KeyError(f"Unknown stage {name!r}; choose from {STAGE_NAMES}")
