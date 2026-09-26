"""Run df-embed's NLP embedding on a chosen device (GPU if available).

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT, not this project's:

    uv run --directory <df-analyze> --python '>=3.13.11,<3.14' \
        python <this file> --data in.parquet --out out.parquet --device cuda

df-embed.py itself always runs on the CPU. This script reuses df-embed's own
code unchanged (model loader, dataset class, and `get_nlp_embeddings`: the
"query: " prefix, 512-token truncation, mean pooling and output layout) and
only wraps the model so each batch is computed on the requested device. To
confirm the outputs match, the first `--verify-rows` rows are also embedded
with the unwrapped CPU code path and compared.

`--cuda-info` only reports whether this environment can see a CUDA GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Import df-analyze from its source tree, as df-embed.py does (the model
# paths in df_analyze.embedding.download are relative to the source tree).
DFA_ROOT = Path.cwd().resolve()
sys.path.insert(0, str(DFA_ROOT / "src"))
sys.path.insert(1, str(DFA_ROOT))

import torch  # noqa: E402  (must be imported before transformers, as in df-embed.py)

# Tolerance for GPU vs CPU float32 differences in the pooled embeddings.
ATOL = 1e-3
RTOL = 1e-3


def cuda_info() -> dict:
    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


class OnDevice(torch.nn.Module):
    """Runs the wrapped model on `device`; returns hidden states on the CPU so
    df-embed's pooling code (which keeps the attention mask on the CPU) is
    unchanged."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, **inputs):
        moved = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**moved)
        return SimpleNamespace(last_hidden_state=out.last_hidden_state.float().cpu())


def embed(
    data: Path,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    batch_size: int,
    verify_rows: int,
) -> tuple:
    """Embed every row of `data` with df-embed's code, running `model` (loaded
    on the CPU) on `device`. If `verify_rows` > 0, the first rows are first
    embedded with the unwrapped model on the CPU for comparison.

    Returns (embeddings, reference or None, report)."""
    from df_analyze.embedding.datasets import NLPDataset
    from df_analyze.embedding.embed import get_nlp_embeddings

    # df-analyze's dataset object caches the first frame it loads and ignores
    # `limit` on later calls, so every pass gets its own dataset object.
    def dataset() -> NLPDataset:
        return NLPDataset(datapath=data, name=None)

    n_rows = len(dataset().X())
    report: dict = {"device_used": str(device), "n_rows": n_rows}
    reference = None
    n_verify = min(verify_rows, n_rows)
    if n_verify > 0:
        print(f"Embedding {n_verify} rows on CPU with df-embed's code for comparison")
        reference = get_nlp_embeddings(
            ds=dataset(), tokenizer=tokenizer, model=model,
            batch_size=batch_size, load_limit=n_verify,
        )  # fmt: skip
        if len(reference) != n_verify:
            raise RuntimeError(
                f"CPU comparison embedded {len(reference)} rows, expected {n_verify}"
            )

    print(f"Embedding {n_rows} rows on {device}")
    start = time.perf_counter()
    embedded = get_nlp_embeddings(
        ds=dataset(), tokenizer=tokenizer, model=OnDevice(model.to(device), device),
        batch_size=batch_size,
    )  # fmt: skip
    report["seconds"] = round(time.perf_counter() - start, 1)
    if len(embedded) != n_rows:
        raise RuntimeError(f"Embedded {len(embedded)} rows, expected {n_rows}")

    if reference is not None:
        cols = [c for c in reference.columns if c.startswith("embed")]
        ref = torch.tensor(reference[cols].to_numpy())
        got = torch.tensor(embedded[cols].iloc[:n_verify].to_numpy())
        report["verify_rows"] = n_verify
        report["verify_max_abs_diff"] = float((ref - got).abs().max())
        report["verify_passed"] = bool(torch.allclose(got, ref, atol=ATOL, rtol=RTOL))
    return embedded, reference, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--verify-rows", type=int, default=64)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cuda-info", action="store_true")
    args = parser.parse_args()

    info = cuda_info()
    if args.cuda_info:
        args.report.write_text(json.dumps(info, indent=2))
        print(json.dumps(info))
        return 0

    if args.device == "cuda" and not info["cuda_available"]:
        print("CUDA was requested but is not available in df-analyze's environment.")
        return 3
    use_cuda = args.device == "cuda" or (args.device == "auto" and info["cuda_available"])
    device = torch.device("cuda" if use_cuda else "cpu")
    # Full float32 on the GPU (no TF32) so results match the CPU path closely.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    from df_analyze.embedding.download import load_nlp_intfloat_ml_model_offline

    model, tokenizer = load_nlp_intfloat_ml_model_offline()
    model.eval()
    # The CPU comparison only makes sense when the main pass is on the GPU.
    embedded, _, result = embed(
        args.data, model, tokenizer, device, args.batch_size,
        verify_rows=args.verify_rows if use_cuda else 0,
    )  # fmt: skip
    report = {**info, **result}
    if report.get("verify_passed") is False:
        args.report.write_text(json.dumps(report, indent=2))
        print("GPU embeddings differ from df-embed's CPU embeddings:", report)
        return 4

    embedded.to_parquet(args.out)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Saved embeddings to {args.out}: {json.dumps(report)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
