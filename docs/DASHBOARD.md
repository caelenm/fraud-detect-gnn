# Results dashboard

One local page with every method we ran, per dataset and across datasets, the pre-registered verdicts with their caveats, and a button that classifies a random test node with the best model.

```bash
uv run scripts/dashboard.py      # then open http://localhost:8555
```

It has two parts, and they need different things:

| Part | Needs | Works on a fresh clone? |
|---|---|---|
| Results: every method, the verdicts, the cross-dataset comparison, the checks | Only `results/runs/`, which is in the repository | Yes |
| Classify a random test node | The runs' outputs, which are never in the repository, and df-analyze's clone | No, see [step 5](#step-5-optional-the-classify-panel) |

## Running it on another Windows machine (WSL2)

Everything except the browser runs inside WSL. These steps were tested from a fresh clone and real bundles on the group laptop, under WSL2 with Debian. They were not tested on a second physical machine, or on macOS.

### Step 1: Install WSL

In PowerShell, as administrator, then restart if it asks:

```powershell
wsl --install -d Debian
```

Open the **Debian** app from the Start menu and create a user when it asks. Run every command below in that terminal. If you already have Ubuntu under WSL, that should work too.

### Step 2: Install the tools

```bash
sudo apt update && sudo apt install -y git curl
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close the terminal and open it again, then check that this prints 0.9.16 or newer:

```bash
uv --version
```

### Step 3: Clone the repository into the WSL home folder

Use `~/...`, not `/mnt/c/...`, which is much slower.

```bash
mkdir -p ~/projects && git clone https://github.com/caelenm/fraud-detect-gnn.git ~/projects/fraud-detect-gnn
```

```bash
cd ~/projects/fraud-detect-gnn && uv sync
```

`uv sync` downloads the right Python and the packages, which takes a few minutes the first time.

### Step 4: Start the results sections

```bash
uv run scripts/dashboard.py --no-model
```

It prints the sanity checks, then the address. Open **http://localhost:8555** in your normal Windows browser; WSL2 forwards it. Leave the terminal open, and press Ctrl+C there to stop.

That is everything except live classification. Without outputs, the classify section says it is not available here, and the other sections are complete.

### Step 5 (optional): The classify panel

**5a. Get the bundles.** On the machine that ran the pipeline, pack each dataset:

```bash
uv run pack_artifacts.py --dataset amazon
```

```bash
uv run pack_artifacts.py --dataset yelpchi
```

The bundles are about 930 MB (Amazon) and 560 MB (YelpChi). They hold row-level data, so share them through OneDrive or Teams within the group only, never through git. Copy both into the repository's root on the new machine. From a Windows folder that looks like this (use your own user name and folder):

```bash
cp /mnt/c/Users/<you>/OneDrive/<folder>/fraud_artifacts_*.tgz ~/projects/fraud-detect-gnn/
```

**5b. Unpack them.** This verifies every file's checksum:

```bash
uv run unpack_artifacts.py --dataset amazon
```

```bash
uv run unpack_artifacts.py --dataset yelpchi
```

**5c. Install df-analyze next to the repository, at the tested commit.** The models need its environment. No GPU is needed to predict.

```bash
uv python install '>=3.13.11,<3.14'
```

```bash
git clone https://github.com/stfxecutables/df-analyze.git ../df-analyze
```

```bash
git -C ../df-analyze checkout 199e5638620693c267dac715784f1fd0e33fa796
```

```bash
uv sync --locked --python '>=3.13.11,<3.14' --directory ../df-analyze
```

**5d. Start it with the model.**

```bash
uv run scripts/dashboard.py
```

The results sections are up at once. The classify panels take a little longer, because each model is loaded and then checked on its whole test set. A panel is ready when it says **Live, verified on all N test rows**.

If you skip 5c, `uv run scripts/dashboard.py --no-model` still shows the panels, using the saved test predictions, and says so.

### If something goes wrong

| What you see | What to do |
|---|---|
| `cannot listen on 127.0.0.1:8555` | The port is in use. Add `--port 8556` and open that port instead. |
| The Windows browser cannot reach `localhost:8555` | Check that the terminal is still running the server. If it is, open `C:\Users\<you>\.wslconfig` and make sure it does not set `localhostForwarding=false`, then run `wsl --shutdown` in PowerShell and start again. |
| `uv ... is too old` | Run `uv self update`, then open a new terminal. |
| A panel says **Not available here** | The message names the missing file. Usually a bundle has not been unpacked. |
| A panel says **Saved predictions** with a note | The note gives the reason, such as df-analyze not being found. Set `DF_ANALYZE_DIR` if your clone is not at `../df-analyze`. The worker's own log is `outputs/dashboard/worker.log`. |
| `df-analyze ... is at commit ...` | Run the `git -C ../df-analyze checkout` and `uv sync` lines from step 5c again. |
| A red banner at the top of the page | A sanity check failed. Do not present these numbers until it is explained. Run `uv run scripts/dashboard.py --check` to see the details. |
| WSL stops by itself after a while | WSL shuts down when idle. Keep the terminal with the server open. |

## What the page shows

The toggles at the top scope the first sections, and they are kept in the URL, so a link opens the same view.

1. **Read this first.** The limits of the results, before the results: the budget, the single machine and seed, Amazon's `f19` warning, YelpChi's sensitivity to tuning, what was not run, and how a verdict is decided.
2. **The questions, answered.** The project's questions, each with its contrast, Δ, CV p and verdict.
3. **Every method on this dataset.** The setup with the highest CV score and how close the others are, a chart of test and CV PR-AUC per setup, and the table behind it, including each run's database entry. It also lists what was not run for this view.
4. **Pre-registered contrasts.** The primary family with Holm-adjusted p values, the other contrasts, and the one-split descriptive ones kept apart. On YelpChi, also the default-settings sensitivity check.
5. **Across the two datasets.** Normalised skill, the gain over B per dataset, and whether the verdicts agree.
6. **Classify a random test node.** See below.
7. **Sanity checks and coverage.** Every check with its result, and anything started but not finished.

### Reading the classify panel

Each panel holds its dataset's best setup by CV, and that setup's Model A on split 0. Amazon has two, one for all 25 features and one without `f19`.

- **Random test node** draws from the whole test set. **Random fraud** and **Random not fraud** pick the class, which makes for a better demo but is not representative. A **seed** makes a draw reproducible, for slides.
- A draw shows P(fraud), the saved threshold, the decision, the true label, what the baseline B said for the same node, the node's most unusual features and its neighbourhood per relation.
- **One draw is an illustration, not evidence.** The whole-test-set numbers at the top of the panel are the performance.
- **"Most unusual features" are not an explanation.** They are the node's highest values against the training nodes. They describe the node, not the model's reasoning.
- **Neighbour colours use training labels** for that picture only. No model input uses a neighbour's label.

## How the numbers are checked

**Every time the server starts,** and with `uv run scripts/dashboard.py --check`, twelve checks run on the results database. Each compares results with what they must be, or two parts of the database written separately:

| Check | What would make it fail |
|---|---|
| Results database is valid | A row-level column, a table too long, a bad UTC time |
| Every run maps to a setup | A feature set the registry cannot read |
| Corrections point at real entries | A `supersedes` naming no entry |
| No run is counted twice | One (dataset, feature set, split) recorded twice; this one stops the server |
| Metrics are in range | A PR-AUC outside [0, 1] or its interval, or not above no skill |
| Each split has one test set | Setups on one split with different test sizes or base rates |
| Budget labels match the logged settings | A "pilot" run that did not log 10 trials and 3 classifiers, or a "full" one that did not log 100 and 6 |
| Model A was chosen by shared CV, not by test | A Model A without the highest shared-CV score in its own results table |
| Verdicts follow the pre-registered rule | A stored verdict the rule does not give from its deltas and p |
| Contrast arithmetic adds up | A mean or SD that does not match its deltas, or the wrong CV fold count |
| Contrasts agree with the runs | A per-split Δ that differs from the two runs' own test PR-AUC |
| No verdict was replaced by one on fewer splits | A replacement with fewer splits than what it replaced |

**For each classify panel,** before any node is shown:

- no test node was trained on;
- the saved predictions reproduce the recorded PR-AUC, and the run matches its database entry;
- the baseline B has the same test nodes;
- the exported model is this run's Model A, and the prepared table lines up with the test IDs, row by row, in the model's column order;
- the running model reproduces every saved test probability, within the tolerance the export check uses (1e-9, or 1e-5 for models trained on a GPU).

A panel that fails the last check shows saved predictions and says so. Any other failed check stops the panel, with the reason. **Each click** compares its own live probability with the saved one, and stops rather than show a score that disagrees.

**The tests** check that every one of these fails on the fault it exists to catch, and that the dashboard's means reproduce `docs/GNN_RESULTS.md`, `docs/RQ_RESULTS.md` and the README's table to 3 decimals (`tests/test_dashboard_*.py`, `tests/test_serve_model.py`).

## Rules

- **Local only.** The server binds `127.0.0.1`, with no option to change it, because the classify panel shows row-level test data. It refuses requests addressed to any other host, and a Content-Security-Policy stops the page from reaching anything else. Never host it, and never commit or upload what the panel shows.
- **Not a pipeline stage.** It adds no stage outputs, so it never makes a finished run or a bundle rerun anything. It writes only the worker's log, under `outputs/`.
- **Two environments.** The server runs in this project's environment. The model worker (`scripts/dfa/serve_model.py`) is the only part in df-analyze's.
- **A new setup or variant** goes in `src/fraud_detect/dashboard/registry.py`. A test fails if a feature set in the config or a run in the database maps to nothing there.

The plan this was built from, and the decisions on it, are in [`DASHBOARD_PLAN.md`](DASHBOARD_PLAN.md).
