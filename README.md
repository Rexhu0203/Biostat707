# BIOSTAT 707: Checkpoints 1 and 2

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

No imputer, encoder, spline transformer, scaler or classifier should be refit on external data. The feature builder validates the supplied observation window; clinical validity and coding conventions still require review. Keep this support module alongside the notebook.

Verification on this Windows workspace: the complete Checkpoint 2 task passed, with 2.7 minutes of notebook analysis (excluding initial environment installation). An isolated copy of Checkpoint 1 also executed successfully in the updated locked environment, preserving its original notebook and HTML. Focused synthetic checks covered sentinel values, retained zeros, missing parameters, summary statistics and the 48-hour boundary. All 188 summaries/outcome columns shared with Checkpoint 1 matched across all 4,000 stays. The rendered Checkpoint 2 report contains 12 numbered tables and three figures; saved-model probabilities agree before and after serialization. Runtime varies with hardware; the grids are intentionally small. Initial Pixi environment creation adds setup time.

A second `pixi run --locked checkpoint2` passed from a fresh directory containing only the Checkpoint 2 notebook, support module, manifest/lock, and a junction to raw `data/`. Pixi created a fresh environment; no Checkpoint 1 outputs were present. Notebook analysis took 3.6 minutes after environment setup. All out-of-fold predictions and comparison tables reproduced, and full-data fitted probabilities agreed to numerical tolerance. All 25 comparison-table fold AUROCs were independently recalculated. This checks a fresh working-directory snapshot; it does not claim a clone of an uncommitted Git revision or validation on other operating systems.

The course checklist is at https://steverozen.github.io/biostat707_2026/checkpoint2-3-checklist.html. Review the rendered report and AI disclosure before submission. No automatic commit or push is performed. The new report and implementation files are staged for review. After committing, a fresh checkout with locally supplied data remains a useful final submission check.
