"""Narrative embeddings (via df-analyze's df-embed.py) and PCA reduction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from fraud_detect import columns as C

EMBED_PREFIX = "embed"
PCA_PREFIX = "text_pc"
# df-embed.py loads the NLP model from here (relative to the df-analyze clone).
DF_EMBED_MODEL_DIR = Path("downloaded_models") / "intfloat_multi_large"


def df_embed_input(sample: pd.DataFrame) -> pd.DataFrame:
    """The two-column table df-embed.py expects: `text` and integer `label`."""
    return pd.DataFrame(
        {
            "text": sample[C.NARRATIVE].astype(str).to_numpy(),
            "label": sample[C.LABEL].astype("int64").to_numpy(),
        }
    )


def attach_ids_to_embeddings(
    embedded: pd.DataFrame, ids: pd.Series, labels: pd.Series
) -> pd.DataFrame:
    """Attach Complaint IDs to df-embed output by row position.

    df-embed.py keeps input row order and returns the label as `target`. We
    check the row count and that `target` equals our labels row by row before
    trusting the positional join.
    """
    embed_cols = [c for c in embedded.columns if c.startswith(EMBED_PREFIX)]
    if len(embedded) != len(ids):
        raise ValueError(f"df-embed returned {len(embedded)} rows for {len(ids)} inputs")
    if "target" not in embedded.columns or not embed_cols:
        raise ValueError(f"Unexpected df-embed columns: {embedded.columns.tolist()[:5]}")
    returned = embedded["target"].astype("int64").to_numpy()
    if not np.array_equal(returned, labels.astype("int64").to_numpy()):
        raise ValueError("df-embed output labels do not match the input order")
    out = embedded[embed_cols].reset_index(drop=True).astype("float32")
    out.insert(0, C.COMPLAINT_ID, ids.to_numpy())
    return out


def fit_pca(
    embeddings: pd.DataFrame, train_ids: pd.Series, n_components: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit PCA on training complaints only and transform every complaint.

    Returns (components table, explained-variance table).
    """
    embed_cols = [c for c in embeddings.columns if c.startswith(EMBED_PREFIX)]
    is_train = embeddings[C.COMPLAINT_ID].isin(train_ids)
    if int(is_train.sum()) != len(train_ids):
        raise ValueError("Some training Complaint IDs have no embedding")
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(embeddings.loc[is_train, embed_cols].to_numpy(dtype=np.float64))
    comps = pca.transform(embeddings[embed_cols].to_numpy(dtype=np.float64))
    width = len(str(n_components))
    names = [f"{PCA_PREFIX}{i + 1:0{max(width, 2)}d}" for i in range(n_components)]
    out = pd.DataFrame(comps, columns=names)
    out.insert(0, C.COMPLAINT_ID, embeddings[C.COMPLAINT_ID].to_numpy())
    variance = pd.DataFrame(
        {
            "component": names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    return out, variance
