# Income Prediction Project

This repository contains a Jupyter notebook and small Streamlit apps used to train and evaluate an XGBoost classifier on the UCI Adult (income) dataset and to run / report on the notebook results.

Contents
 - `App.ipynb` — Primary Jupyter notebook with data loading, preprocessing, feature engineering, model training, hyperparameter tuning, saving the best model, and a prediction example.
 - `income.csv` — Local copy of the UCI Adult dataset used as the raw data source.
 - `streamlit_app.py` — Streamlit runner that executes `App.ipynb` and renders the executed notebook as HTML inside a Streamlit page.
 - `requirements.txt` — Minimal package list required to run the Streamlit runner and execute the notebook programmatically.
 - `best_xgboost_model.joblib` — Saved best model produced by the notebook (binary, not included in the text dump).

Project Goal
 - Build a robust, reproducible pipeline that loads the Adult income dataset, applies necessary preprocessing, trains an XGBoost binary classifier to predict whether income is `>50K`, evaluates model performance, and exposes a simple Streamlit interface for running the notebook and generating a project report.

How the project is organized and what was applied

1) Data loading and target encoding
 - The notebook reads `income.csv` into a pandas DataFrame and maps the target `income` values to numeric labels (e.g., `<=50K` -> `0`, `>50K` -> `1`).
 - It prints and displays dataset head & value counts for quick verification.

2) Train/test split
 - The dataset is split with `train_test_split` (default seed in the notebook) into `X_train`, `X_test`, `y_train`, and `y_test`.

3) Preprocessing and feature engineering
 - Missing value handling and label fixes:
	 - Some categorical fields use `?` for unknown values. These are replaced with `"unknown"` when appropriate (e.g., `workclass`).
	 - For numerical fields, missing values are filled with sensible statistics (noted in the notebook where applied).
 - Categorical handling:
	 - One-hot encoding is applied to high-cardinality or nominal categorical features such as `workclass`, `race`, and `native-country` using `OneHotEncoder(handle_unknown='ignore')` so that unseen categories in test/new data won't break inference.
	 - For `native-country`, rare countries are grouped into an `Other` category based on a small-threshold heuristic before one-hot encoding.
	 - The `education` column is converted to an ordered categorical according to a defined `education_order`, then converted to integer codes (1-indexed) into `education_encoded`.
 - Target-guided encoding:
	 - The notebook computes a simple target-mean encoding for `occupation` based on training data and applies the mapping to both train and test, filling unknowns with the global mean so the feature is stable at inference time.
 - Column alignment and dropping:
	 - Redundant columns (e.g., `relationship`) are dropped consistently. The pipeline resets indexes where needed and aligns feature order for prediction on new rows.

4) Robust label handling
 - The notebook contains explicit label normalization code that ensures `y_train` and `y_test` are 1-D integer arrays of 0/1. It converts string variants like `"<=50K"` and `">50K"` to numeric labels and raises a clear error if non-convertible labels are encountered.
 - Any rows with missing labels are dropped and X/y are re-aligned so training never receives NaNs as labels.

5) Model selection and training
 - XGBoost (`xgboost.XGBClassifier`) is used as the primary model.
 - The notebook includes two training phases:
	 - Baseline training of an XGBClassifier with configured hyperparameters.
	 - Grid search (`GridSearchCV`) over `n_estimators`, `learning_rate`, and `max_depth` with cv=3 to find best parameters.
 - When fitting the XGBoost model, the notebook uses a version-robust early stopping approach:
	 - Preferred: use `callbacks = [xgb.callback.EarlyStopping(rounds=10, save_best=True)]` and pass `callbacks=` into `fit()` (works with modern XGBoost builds).
	 - Fallback: if the installed XGBoost rejects the `callbacks`/`early_stopping_rounds` usage (older builds), the notebook falls back to calling `fit()` without early stopping and prints informative messages.
 - The notebook re-instantiates the classifier prior to calling `fit()` to avoid stale `classes_` attributes from previous runs in the same session.

6) Evaluation and persistence
 - After training, predictions are made on train and test sets and `accuracy_score` is used to report performance.
 - The best model (from grid search) is re-initialized with best params, trained on training data and then saved as `best_xgboost_model.joblib` using `joblib.dump`.

7) Prediction example
 - The notebook contains a reproducible example for predicting a single new row:
	 - The example constructs `new_raw_data`, applies the same fitted encoders/transformations (OHEs, target encodings, ordinal encoding, etc.), aligns the column order to `X_train.columns`, then calls `best_model.predict(...)` and `predict_proba(...)` to show class and probability.

Streamlit apps
 - `streamlit_app.py`: a lightweight runner that reads `App.ipynb`, executes it (using `nbconvert.ExecutePreprocessor`), exports rendered HTML via `nbconvert.HTMLExporter`, and embeds the HTML inside a Streamlit page. This provides a quick way to reproduce the notebook end-to-end via a web UI. Execution uses the `python3` kernel by default and a 600s timeout.

Reproduction / Quick start (PowerShell on Windows)
```powershell
cd "c:\Users\t6201\OneDrive\Documents\IMP_SUB\Machine learning\Project"
python -m pip install --upgrade pip
pip install -r requirements.txt
# XGBoost is required by the notebook; install explicitly if not present:
pip install xgboost

# Run the Streamlit notebook runner
streamlit run streamlit_app.py
```

Notes and troubleshooting
 - Kernel compatibility: `nbconvert.ExecutePreprocessor` runs the notebook in a kernel named `python3` by default. If your environment uses a different kernel name or virtual environment, adjust the `kernel_name` passed into `ExecutePreprocessor` or run Streamlit from the same environment that satisfies the notebook dependencies.
 - XGBoost API differences: older XGBoost versions do not accept `early_stopping_rounds` or `callbacks` in the same way. The notebook contains a fallback path that trains without early stopping and prints clear diagnostics. If you prefer early stopping, upgrade XGBoost with `pip install -U xgboost`.
 - Missing labels: If you see errors about invalid classes or `NaN` in labels, inspect the notebook prints — the notebook now prints `y_train`/`y_test` statistics and drops rows with missing labels so training is not attempted with NaN labels.
 - Large CSV size: `income.csv` contains ~48K rows. If notebook execution is slow, consider working on a sampled subset or increasing the `timeout` in `streamlit_app.py`.

What I changed (recent edits performed by the project collaborator)
 - Added robust label normalization and missing-label removal prior to training to avoid `ValueError: Found array with NaNs` or invalid class label errors.
 - Replaced direct `early_stopping_rounds=` usage with `callbacks`-based early stopping and added a fallback to `fit()` without early stopping for older XGBoost builds.
 - Ensured OHEs and encoders are fit on `X_train` and reused on `X_test` and new inputs to maintain consistency at inference.

Next steps / improvements
 - Add a reproducible script that runs the entire pipeline (data → model → report) without requiring the Streamlit runner.
 - Add unit tests for preprocessing transforms to guard against schema drift and missing columns at prediction time.
 - Provide a small `report_app.py` (previously included) to summarize notebook findings and export a markdown report — re-add if you need the richer report UI.

Contact / help
 - If you run into errors executing the notebook via Streamlit, copy the full traceback and paste it here — I can help adapt the notebook to your local environment (kernel name, package versions, or data path differences).

License / data
 - The `income.csv` dataset is the UCI Adult dataset. Respect the dataset license and privacy considerations when using or distributing derived models.

-----


