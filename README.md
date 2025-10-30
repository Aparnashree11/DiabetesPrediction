# Diabetes Prediction using Apache Beam Pipelines (MLOps Lab4)

A hands-on lab demonstrating batch and streaming ML inference using Apache Beam and scikit-learn. The project trains a Random Forest model to predict diabetes and shows two example Beam pipelines that load the trained artifacts to run offline (batch) and near-real-time (streaming) predictions.

## Repository structure

- `diabetes_prediction.ipynb`  - Main notebook with end-to-end workflow: data prep, training, model saving, custom Beam transforms, batch & streaming pipelines.
- `diabetes_prediction_dataset.csv` - Raw dataset used for training and demos.
- `data/` - Generated CSV splits used by pipelines (`diabetes_train.csv`, `diabetes_batch.csv`, `diabetes_streaming.csv`).
- `models/` - Directory for model artifacts. The repository tracks a placeholder file 
- `output/` - Pipeline outputs (batch and streaming results).

## Goals

- Show how to prepare data for training and for inference pipelines.
- Train and save a scikit-learn RandomForest model plus preprocessing objects (label encoders, scaler).
- Demonstrate how to run ML inference in parallel with Apache Beam using the `RunInference` API and a custom `ModelHandler`.
- Provide a simple streaming simulation and windowed summary metrics.

## Quick setup

1. Create a Python virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the dependencies (example):

```powershell
pip install apache-beam scikit-learn pandas numpy joblib
```

3. Open `diabetes_prediction.ipynb` in Jupyter/VS Code and run cells in order. The notebook is written so you can run sections independently:

- "Data Preparation" — creates `data/` splits
- "Model Training and storing" — fits model and saves artifacts under `models/`
- "Custom Transforms for Prediction" — Beam DoFns used by pipelines
- "Batch Pipeline" and "Streaming Pipeline" — example usages of the trained artifacts with Beam

## Running the pipelines

This lab uses the DirectRunner (local) by default. The pipeline functions are defined inside the notebook. To run them:

1. In the notebook, after training and saving artifacts, run the `run_batch_pipeline()` cell to execute the batch pipeline. This will write outputs to `output/batch_predictions/`.

2. Run `run_streaming_pipeline()` cell to simulate streaming predictions. Outputs go to `output/streaming_predictions/`.

Notes:
- For production or larger-scale testing, change `PipelineOptions` runner to Dataflow or another supported runner and provide the runner-specific options.
- The notebook's Beam code expects the following artifacts in `models/`:
  - `diabetes_model.pkl`
  - `label_encoders.pkl`
  - `scaler.pkl`
  - `feature_names.pkl`
