"""Report whether df-analyze's environment can see a CUDA GPU.

THIS SCRIPT RUNS INSIDE DF-ANALYZE'S ENVIRONMENT, not this project's:

    uv run --directory <df-analyze> --python '>=3.13.11,<3.14' \\
        python <this file> --report cuda.json

df-analyze trains CatBoost and GANDALF on the GPU when torch sees CUDA. The
df_analyze stage runs this first, so a missing GPU is reported (or, with
`gpu.require: true`, stops the run) before hours of tuning on the CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def cuda_info() -> dict:
    available = torch.cuda.is_available()
    return {
        "cuda_available": available,
        "device_name": torch.cuda.get_device_name(0) if available else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    info = cuda_info()
    args.report.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
