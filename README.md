# Calories Burnt Prediction

Predict calories burned during exercise using machine learning regression models. This repository contains the data preprocessing, training pipeline, model artifacts, and an inference utility to make predictions on new samples.

## Table of contents

- [Project overview](#project-overview)
- [Repository structure](#repository-structure)
- [Quick start](#quick-start)
- [Usage](#usage)
    - [Train models](#train-models)
    - [Run inference](#run-inference)
- [Model artifacts](#model-artifacts)
- [Implementation details](#implementation-details)
- [Dependencies](#dependencies)
- [Development & testing](#development--testing)
- [Notes & next steps](#notes--next-steps)
- [Credits & dataset](#credits--dataset)

## Project overview

This project trains regression models to estimate calories burned given exercise session attributes. It combines two CSV files (`exercise.csv` and `calories.csv`) and performs feature engineering (including BMI calculation), scaling/encoding, and model selection. Trained models and preprocessing objects are saved under `models/` for later inference.

Key goals:

- Provide a reproducible training pipeline.
- Persist preprocessing objects and models for production-ready inference.
- Offer a simple inference script to generate predictions for new samples.

## Repository structure

Top-level files:

- `README.md` - this file.
- `environment.yaml`, `requirements.txt` - dependency manifests.
- `notebooks/Calories_burnt.ipynb` - exploratory analysis and interactive experiments.

Source and data:

- `app/` - contains `app.py` (if present; can be used to expose a small demo or service).
- `data/` - input CSVs: `exercise.csv`, `calories.csv`.
- `models/` - saved preprocessing objects and trained model artifacts (joblib files).
- `src/` - package source code used by the pipeline:
    - `src/utils.py` - helper functions: I/O, preprocessing, feature engineering, sample generation, etc.
    - `src/training_pipeline.py` - orchestrates preprocessing, training, evaluation, and saving artifacts.
    - `src/inference.py` - loads saved artifacts and runs predictions on generated samples.

## Quick start

1. Create a Python environment (recommended: conda).

    PowerShell example:
```powershell
conda env create -f environment.yaml
conda activate <env-name>
# or using pip
python -m pip install -r requirements.txt
```

2. Ensure the CSV files are in the `data/` directory: `data/exercise.csv` and `data/calories.csv`.

3. Run the training pipeline to produce model artifacts:
```powershell
python src/training_pipeline.py
```


4. Run inference to see example predictions:

```powershell
python src/inference.py
```

## Usage

### Train models

Run `src/training_pipeline.py`. The script will:

- Load `data/exercise.csv` and `data/calories.csv`.
- Merge datasets and perform feature engineering (BMI calculation and any derived columns).
- Split data into train/test sets.
- Fit preprocessing pipeline (column transformer) and multiple regression models.
- Evaluate models using metrics such as R² and mean squared error (MSE).
- Save fitted objects and best models into `models/` as joblib files.

By default, artifacts saved in `models/` include:

- `column_transformer_fitted.joblib` - fitted preprocessing pipeline for numeric and categorical columns.
- `column_values_range.joblib` - saved ranges or encoding maps used for generating valid sample values.
- `trained_<ModelName>.joblib` - saved trained model files, e.g. `trained_RandomForestRegressor.joblib`.

To run the training script from the repo root:

```powershell
python src/training_pipeline.py
```

### Run inference

Run `src/inference.py` which will:

- Load the preprocessing objects and a chosen trained model from `models/`.
- Generate example input samples (the code uses a helper to randomly sample realistic values within observed ranges).
- Preprocess the samples and return predicted calories burned.

To run inference with a specific model file (example):

```powershell
python src/inference.py --model models/trained_RandomForestRegressor.joblib
```

Check `src/inference.py` for available CLI flags (model path, number of samples, random seed).

## Model artifacts

Files in `models/` (existing in repo):

- `column_transformer_fitted.joblib` - preprocessing pipeline (scalers, encoders).
- `column_values_range.joblib` - min/max or categorical value lists used to generate samples.
- `trained_KNeighborsRegressor.joblib`
- `trained_Linear_SupportVectorRegressor.joblib`
- `trained_LinearRegression.joblib`
- `trained_RandomForestRegressor.joblib`

These artifacts are produced by the training pipeline and can be loaded with `joblib.load()` for inference in other scripts or services.

## Implementation details

Design contract:

- Inputs: CSV files `data/exercise.csv` and `data/calories.csv` containing session-level and calories information.
- Outputs: saved preprocessing objects and trained model artifacts; evaluation metrics printed or logged.
- Error modes: missing files, corrupted CSVs, or mismatched schemas (scripts raise informative errors).

High-level pipeline steps (as implemented in `src/training_pipeline.py` and `src/utils.py`):

1. Load CSVs and merge on a common key (e.g., `User_ID` or session/time-based key) — see the notebook for exploratory merges.
2. Feature engineering: compute BMI from `Weight` and `Height`, encode categorical columns, scale numerical features.
3. Split into training and test sets.
4. Train several regression models (Random Forest, Linear Regression, KNN, LinearSVR, Decision Tree).
5. Evaluate and save the best-performing models along with the fitted column transformer.

Edge cases considered:

- Null/missing values are handled (imputation or row removal depending on the column).
- Categorical levels unseen at inference are handled via saved value ranges and transformers.
- Numeric ranges are preserved to allow realistic sample generation.

## Dependencies

Primary Python packages used:

- pandas
- numpy
- scikit-learn
- joblib
- matplotlib (for notebook EDA)
- seaborn (for notebook EDA)

Install using pip (from `requirements.txt`) or conda (from `environment.yaml`). Example pip install:

```powershell
python -m pip install -r requirements.txt
```

## Development & testing

- Code is organized under `src/` and can be imported as a package for unit tests.
- Add unit tests for `src/utils.py` functions (preprocessing, sample generation, metric calculation).
- Suggested CI: run tests, linting (flake8/ruff), and run a smoke test that runs `src/training_pipeline.py` on a small sample.

## Notes & next steps

- Consider adding a lightweight API (FastAPI/Flask) in `app/` to serve model predictions.
- Add deterministic unit tests and CI pipelines (GitHub Actions) to validate training and inference steps on sample data.
- Add example usage and a small sample dataset subset for quick local testing.
- Package `src/` as an installable module (setup.cfg / pyproject.toml) for easier reuse.

## Credits & dataset

- Original dataset referenced from Kaggle: [fmendes/fmendesdat263xdemos](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos)
- See `notebooks/Calories_burnt.ipynb` for full exploratory analysis and model selection notes.

