# BIOSTAT 707: Checkpoints 1, 2 and 3

Place the 4,000 raw patient text files in `data/set-a/` and `Outcomes-a.txt` directly in `data/`. Raw data and patient-level derived files are ignored by Git. Run commands from the repository root with Pixi installed. Dependencies are resolved in `pixi.lock` for Windows, Linux, and macOS.

```sh
pixi run --locked checkpoint1
pixi run --locked checkpoint2
```

Each command independently executes its notebook and exports a self-contained report to `output/checkpoint1.html` or `output/checkpoint2.html`. Both HTML reports are allowed through `.gitignore` and should be committed with the corresponding source notebook. Notebook outputs are stripped by the existing nbstripout Git filter. Checkpoint 2 does not depend on running Checkpoint 1 or on its generated CSVs. `pixi run lab` opens JupyterLab.

Checkpoint 2 uses only set-a, with one 0–48-hour summary per stay. `checkpoint2_support.py` reproduces Checkpoint 1's cleaning and identifier checks, and provides a reusable feature builder. Three models use the same five outer folds and three-fold inner tuning by log loss: ridge logistic regression, additive cubic-spline logistic regression, and histogram gradient boosting. SAPS-I and SOFA are standalone comparison scores. The report documents missing scores, all predictor definitions, fitted effects and limitations. The nonlinear effects use 100 bootstrap refits for pointwise confidence bands. Seeds and two numerical threads are fixed.

All generated artifacts are under `output/`:

- `checkpoint2.html`: complete report, including all tables, figures and result-driven interpretations.
- `cp2_final_model.joblib`: selected full-set-a fitted sklearn Pipeline, including preprocessing.
- `cp2_model_metadata.json`: predictor order, selected model, versions, runtime and selection rule.
- `cp2_oof_predictions.csv`: RecordID, death outcome, shared fold, three held-out probabilities, raw benchmark scores and complete-score indicator.
- `cp2_comparison.csv`, `cp2_tuning.csv`, `cp2_predictor_summary.csv`, `cp2_ridge_coefficients.csv`, `cp2_partial_effects.csv`: analysis tables.
- `cp2_modeling_table.csv`: patient-level analysis input, excluded from Git.
- `cp2_ridge_coefficients.png`, `cp2_partial_effects.png`, `cp2_roc.png`: figures embedded in the HTML.

The generated model and patient-level predictions are intentionally local artifacts. Retain or securely transfer them for Checkpoint 3; a fresh checkout recreates them by running the checkpoint2 task. Load joblib files only from a trusted source, using the locked environment. To apply the saved model without retraining in Checkpoint 3:

```python
import json
from pathlib import Path
import joblib
from checkpoint2_support import build_features

metadata = json.loads(Path('output/cp2_model_metadata.json').read_text())
model = joblib.load('output/cp2_final_model.joblib')
# future_patient_folder is supplied in Checkpoint 3; no external set is read here.
features = build_features(future_patient_folder)
features = features.reindex(columns=metadata['predictors'])
probabilities = model.predict_proba(features)[:, 1]
```


## Checkpoint 3: external mortality evaluation

Place the 4,000 patient files for each external cohort in `data/set-b/` and `data/set-c/`, and their outcomes in `data/Outcomes-b.txt` and `data/Outcomes-c.txt`. Run:

```sh
pixi run --locked checkpoint3
```

The command activates/installs the locked Pixi environment, executes `checkpoint3.ipynb` from a fresh Python kernel in notebook order, and writes `output/checkpoint3.html`, including the written report, tables, embedded figures and AI Use Statement. No manual activation or separate export command is required. Run it from the repository root. All analysis inputs are relative to that directory; filenames and capitalization must match exactly.

Required inputs, deliberately ignored by Git:

- `data/set-b/*.txt` and `data/set-c/*.txt`: 4,000 patient files per cohort, named by RecordID, with `Time,Parameter,Value` columns. Extract the patient files directly into these directories; the task does not extract archives.
- `data/Outcomes-b.txt` and `data/Outcomes-c.txt`: matching outcomes containing `RecordID`, `In-hospital_death`, `SAPS-I` and `SOFA`. Patient IDs must be unique and match the corresponding cohort exactly.
- `output/cp2_final_model.joblib`: the final fitted Checkpoint 2 pipeline, including preprocessing.
- `output/cp2_model_metadata.json`: the matching model metadata and ordered predictor list.
- `output/cp2_comparison.csv`: Set A cross-validation comparison table.
- `output/cp2_oof_predictions.csv`: Set A outcomes, out-of-fold predictions and benchmark scores.
- `output/cp2_modeling_table.csv`: Set A reference values for the case interpretation section.
- `output/cp2_partial_effects.csv`: Checkpoint 2 reference effect curves.

A fresh Git checkout does not contain these inputs. To evaluate the existing final model on another computer, transfer the six matching Checkpoint 2 artifacts from a trusted source and supply the external data in the locations above. No Set A raw data or Checkpoint 2 rerun is needed when those artifacts are present. Alternatively, missing Checkpoint 2 artifacts can be recreated beforehand with `pixi run --locked checkpoint2`, using `data/set-a/` and `data/Outcomes-a.txt`; this is a separate training step (the recorded Checkpoint 2 run took about 32 minutes), never part of the `checkpoint3` task. Keep all six artifacts from the same Checkpoint 2 run, and use the committed lockfile to preserve the model's Python/scikit-learn/NumPy/pandas versions.

`checkpoint3_support.py` reproduces the Checkpoint 2 feature builder but drops records with a blank parameter name. Set C has 2,403 such observations; their measurement type is unknown. External preprocessing and the final model are never refitted; the SOFA risk mapping is fitted on Set A only, and calibration regressions are evaluation summaries. Bootstrap sampling and permutation importance use seed 707. Results are retrospective. The report contains patient-level RecordIDs, case details and two author-review prompts in the subgroup/case commentary; review before sharing. Raw data and patient-level CSVs remain excluded from Git. Commit `checkpoint3.ipynb`, `checkpoint3_support.py` and `output/checkpoint3.html` alongside the task and documentation changes.

### Environment portability and verification

The manifest and lockfile include `win-64`, `osx-arm64`, `osx-64` and `linux-64`. The macOS lock entries were resolved with a macOS 13.0 virtual-system baseline. The main analysis versions match across these platforms: Python 3.12.14, scikit-learn 1.9.1, joblib 1.6.0, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1 and statsmodels 0.15.0. Keep `pixi.lock`; `--locked` rejects a stale lockfile instead of updating it.

This audit used Pixi 0.81.0. `pixi lock --check` verified manifest/lockfile consistency for the declared platforms. An independent `pixi lock --dry-run` in an ignored temporary workspace successfully resolved both Mac architectures from Windows, with every direct dependency pinned to its existing locked version; it did not change this project's lockfile or install Mac packages. This verifies dependency resolution, not macOS execution.

On November 24, 2026, the exact instructor command succeeded on Windows in 249.33 seconds using the existing locked environment and its Python/Jupyter kernel. The generated HTML contained eight tables, seven embedded figures, the written conclusions and AI Use Statement. All tables, figures and displayed analysis outputs matched the previous HTML. Hash checks confirmed that the Checkpoint 2 artifacts, analysis source files and lockfile were unchanged. The saved notebook has nonsequential execution counters and one stale case-perturbation output table; the task ignores saved outputs and successfully executed all eight code cells in order from a fresh kernel. The notebook was left unchanged.

On an actual Apple Silicon and Intel Mac, supply the documented inputs and run the exact command above to verify environment installation, saved-model loading, full notebook execution and the generated report. Neither macOS execution nor a fresh Windows environment installation was tested in this audit. Small floating-point or rendering differences between operating systems are possible; bitwise identical results across platforms have not been established.
