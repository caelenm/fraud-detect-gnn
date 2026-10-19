"""Write the results of a tiny df-analyze run, using df-analyze's own classes.

RUNS INSIDE DF-ANALYZE'S ENVIRONMENT (tests/test_export_model_dfa.py calls it
with `uv run --directory <df-analyze>`). Not collected by pytest.

The input tables are the synthetic df-analyze input tables a test made in
tmp_path (from tests/synthetic.py data); no real data is involved. A dummy
classifier and a logistic regression are fit with df-analyze's own wrappers
and saved with df-analyze's own EvaluationResults.save, exactly as a real run
saves its tuned models (eval_htune_results_jsonpickle_00.json,
prediction_results_00.json, X/y exports) plus the tuned_models_00.csv table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DFA_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(DFA_ROOT / "src"))
sys.path.insert(1, str(DFA_ROOT))

import pandas as pd  # noqa: E402
from df_analyze.enumerables import ClassifierScorer  # noqa: E402
from df_analyze.hypertune import EvaluationResults, HtuneResult  # noqa: E402
from df_analyze.models.dummy import DummyClassifier  # noqa: E402
from df_analyze.models.linear import LRClassifier  # noqa: E402

# (df-analyze class, "tuned" parameters, invented tuning score)
MODELS = [
    (DummyClassifier, {"strategy": "prior"}, 0.5),
    (LRClassifier, {"C": 1.0, "l1_ratio": 0.5}, 0.6),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="<run>/train/<hash>")
    args = parser.parse_args()
    train, test = pd.read_parquet(args.train), pd.read_parquet(args.test)
    X_train, y_train = train.drop(columns="target"), train["target"]
    X_test, y_test = test.drop(columns="target"), test["target"]

    results = []
    for cls, params, score in MODELS:
        model = cls()
        model.tuned_args = params
        model.refit_tuned(X=X_train, y=y_train, tuned_args=params)
        results.append(
            HtuneResult(
                selection="none",
                selected_cols=X_train.columns.to_list(),
                embed_select_model=None,
                model_cls=cls,
                model=model,
                params=params,
                metric=ClassifierScorer.BalancedAccuracy,
                score=score,
                preds_test=model.tuned_predict(X_test),
                preds_train=model.tuned_predict(X_train),
                probs_test=model.predict_proba(X_test),
                probs_train=model.predict_proba(X_train),
            )
        )
    evaluation = EvaluationResults(
        df=pd.DataFrame({"model": [cls.shortname for cls, _, _ in MODELS]}),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        results=results,
        is_classification=True,
    )
    evaluation.save(args.out / "results", fold_idx=0)
    tuning = args.out / "tuning" / "test00"
    tuning.mkdir(parents=True, exist_ok=True)
    evaluation.hp_table(fold_idx=0).to_csv(tuning / "tuned_models_00.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
