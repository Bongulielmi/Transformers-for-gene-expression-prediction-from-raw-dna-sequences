# CLAUDE.md (v2)

## 0) Scope (what this repo actually does)
Predict gene expression from **raw DNA promoter sequences** plus optional **auxiliary features**.

Models implemented (depending on `model_type`):
- Transformer variants (standard attention plus an FFT-mixing FNet block)
- 1D CNN
- BiLSTM
- TF-NET

### Source-of-truth directories
- `Classes_py/` contains the canonical runnable modules (cleaner than notebooks)
- `Classes/` contains notebooks; use only for exploration, not production logic

---

## 1) Reproducibility Contract (non-negotiable)
Every experiment must be reproducible from the repository + logged config.

Required artifacts per run:
- dataset version + path
- model config (including `model_type`, embed dims, heads, layers)
- random seed (default used in repo: **43**, unless explicitly changed)
- training + validation metrics

Do not commit notebooks or scripts with:
- `!pip`, `!rm`, `!wget`, `%%bash`
- `%pylab inline`, `%tensorflow_version`, or other notebook-only constructs

This rule is enforced by `tests/test_repo_contracts.py`, not just documented.

---

## 2) Data Contract (HDF5)
### Required HDF5 keys
The pipeline expects HDF5 datasets with these keys:
- `promoter`: shape `(N, 10500, 4)` (one-hot A/C/G/T)
- `halflife`: shape `(N, 8)`
- `label`: shape `(N,)` or `(N, 1)` (continuous)
- `gene`: optional but used in some flows

Optional keys (only if enabled):
- `tf`: shape `(N, 181)`
- `micro`: shape depends on dataset; must match training code expectations

### Directory defaults
Default dataset directory referenced in the repo:
- `Dataset/pM10Kb_1KTest`

Rules:
- no inference on missing keys (fail fast)
- validate shapes at load time
- keep dataset versions explicit (`v1`, `v2`, etc.)

---

## 3) Model Interface Standard
All models must expose a consistent minimal interface:

Required:
- `_build_model()` invoked in `__init__` (explicit model construction)
- Keras `compile()` applied in `__init__`, not left to the caller
- a training entry point with explicit epochs and early-stop behaviour defined
- an evaluation path returning at least:
  - Pearson correlation
  - R²

Input contract:
- promoter input: 3D tensor `(batch, 10500, 4)`
- auxiliary features concatenation ordering must be explicitly documented per `model_type`

---

## 4) Repository Architecture Map (what calls what)
Typical flow:
1. `DataManager` loads HDF5 datasets from `Dataset/` into train/val/test splits
2. A model class is constructed with the requested `model_type`
3. The model builds and compiles itself in `__init__`, then trains
4. Metrics plotted via `CI_plt` / plotting utilities

Canonical file reference points (these are flat modules, not packages):
- `Classes_py/DataManager.py` — HDF5 loading, splits, filtering (`DataManager`)
- `Classes_py/Transformer.py` — `projTransformer`, `TransformerBlock`, `FNETBlock`, embeddings, `CustomSchedule`
- `Classes_py/CNN1D.py` — `projCNN1D` (Xpresso-derived 1D CNN backbones)
- `Classes_py/BioLSTM.py` — `BioLSTM` (LSTM backbone)
- `Classes_py/TF_net.py` — `projTFNet` (transcription-factor-only network)
- `Classes_py/CI_plotter.py` — `CI_plt` confidence-interval comparison plots

Workflow notebooks live in `WORKFLOW_GPU/` and `WORKFLOW_TPU/`; filenames encode the
input combination — `P` promoter, `PH` promoter+half-life, `PHT` +transcription factors,
`PHM` +microRNA.

---

## 5) Examples (copy/paste, keep it canonical)

### Data loading
```python
from Classes_py.DataManager import DataManager

dm = DataManager(datadir="Dataset/pM10Kb_1KTest")
X_train, y_train = dm.get_train()
X_valid, y_valid = dm.get_validation()
X_test, y_test = dm.get_test()
```

### Minimal training run
```python
from Classes_py.Transformer import projTransformer

# builds and compiles the model inside __init__
model = projTransformer(
    model_type="best",
    embed_dim=32,
    num_heads=4,
    ff_dim=64,
    batch_size=32,
    learning_rate=1e-4,
    optimizer="Adam",
    loss="mse",
)
```

Available `projTransformer` model types include `"best"`, `"DeepLncLoc"`,
`"DeepLncLoc_TF"` and `"DeepLncLoc_onlyPromo"`; the auxiliary inputs each one
expects differ, so check `_build_model()` before switching.

---

## 6) Contribution Guidelines (human + AI, enforceable)

### PR checklist
- [ ] HDF5 keys and feature shapes validated via DataManager
- [ ] No notebook-only magics in `Classes_py/` (enforced by `tests/test_repo_contracts.py`)
- [ ] Added/updated docstrings: inputs, outputs, assumptions
- [ ] CI passes locally before PR:
  - `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics`
  - `pytest`
- [ ] Updated this doc if architecture changes materially

### Style rules
Prefer:
- small modules, single responsibility
- explicit configs over hidden defaults
- fail fast on data shape mismatches

---

## 7) CI Checks
Current workflows in `.github/workflows/` include:
- `.github/workflows/python-app.yml` (flake8 + pytest)
- `.github/workflows/codeql.yml` (security scanning)

Enforced today:
1. flake8 blocking gate (`E9,F63,F7,F82`) — syntax errors and undefined names fail the build
2. `pytest` runs a real suite; a no-op collection is itself a failure
3. "no notebook magics in modules" — asserted per-module in `tests/test_repo_contracts.py`
4. CLAUDE.md may not cite in-repo paths that do not exist

Deliberately deferred (would fail red on this legacy, notebook-derived codebase and
so cannot land in the same change that turns CI green):
- `ruff` as a full lint gate
- `mypy` type checking

---

## 8) Security & Privacy Notes (required)
This repo can touch sensitive biological signals; treat it as privacy-critical.

Strict rules:
- never commit real patient data
- never commit raw datasets; use `.gitignore` for `Dataset/`
- never commit credentials or API keys
- use `.env` for secrets; do not bake secrets into configs or notebooks

Threat model to avoid:
- inferred re-identification through unique promoter sequences
- accidental dataset leakage via `Saved_Models/` archives containing config metadata

---

## 9) Known Issues (keep this brutally honest)
- no `requirements.txt` in the repo; environment must be written down and version-pinned.
  CI therefore installs only `flake8` and `pytest`, which is why the test suite parses
  modules with `ast` instead of importing them
- notebooks under `Classes/`, `WORKFLOW_GPU/`, `WORKFLOW_TPU/` and `varie/` still contain
  Colab magics; only the `Classes_py/` modules are held to the no-magics rule
- README.md still describes an `src/` `data/` `results/` `configs/` layout that does not
  exist; source-of-truth is `Classes_py/` and this document
- `Saved_Models/checkpoint/` holds 60+ ad-hoc checkpoint directories with no manifest
  tying them back to configs, which undercuts section 1

---

## 10) AI Assistant Operating Rules (agent-neutral)
Absolute prohibitions:
- do not invent module names, directories, or datasets
- do not silently swap feature ordering (promoter/halflife/tf/micro)
- do not "fix" missing dependencies by adding random pip packages without context

If there's ambiguity:
- fail fast with a clear message containing: missing key, expected shape, and file path
