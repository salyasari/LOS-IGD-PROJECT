# LOS-IGD-PROJECT
Early prediction of prolonged Emergency Department Length of Stay (≥6 hours) using structured and clinician-guided feature engineering with LightGBM, SHAP explainability, and robustness analysis.

Early Prediction of Prolonged Length of Stay in Emergency Department

This repository contains the full research pipeline for early detection of prolonged Emergency Department (ED) Length of Stay (LOS ≥ 6 hours) using structured and clinician-guided feature engineering.

The study follows a CRISP-DM framework and evaluates model performance, interpretability, and robustness in a controlled experimental design.

# Research Objective

To develop a recall-oriented binary classification model capable of identifying high-risk ED visits (LOS ≥ 6 hours) at an early stage of patient arrival, using structured clinical and operational features.

# Dataset Overview

Study Period: January – December 2024

Total ED Visits: 39,087 (after cleaning)

Target Variable:

Class 0 → LOS < 6 hours

Class 1 → LOS ≥ 6 hours

Class Distribution: ~76% vs ~24% (imbalanced)

Final Feature Set: 76 features after hybrid feature engineering

All patient identifiers were pseudonymized to ensure data privacy.

# Methodology Overview

The modeling pipeline consists of:

# Structured Feature Engineering (Model A)

Diagnosis grouping

Procedure aggregation

Vital sign normalization

Operational features (shift, weekend, service load)

Categorical encoding

# Clinician-Guided Feature Engineering (Model B)

Clinical × operational interactions

Diagnosis × vital sign interactions

Diagnosis × complexity interactions

Historical × clinical interactions

Model B extends Model A while keeping the training pipeline identical to ensure controlled comparison.

# Modeling Approach

Baseline: Logistic Regression

Main Model: LightGBM

Class imbalance handling: SMOTE (applied only to training set)

Hyperparameter optimization: Optuna

Threshold optimization: Recall-oriented tuning

Evaluation Metrics:

ROC-AUC

Recall (Sensitivity)

Precision

F1-score

Confusion Matrix

# Explainability & Robustness
SHAP Analysis

Global feature importance

Local explanation (case-level)

Comparison between Model A and Model B

Ablation Study

Feature removal experiments (e.g., administrative variables)

Performance stability assessment

Dependency analysis

# Key Findings

Interaction-based features improved recall for prolonged LOS detection.

Threshold optimization significantly reduced false negatives.

Model predictions were driven by both clinical and operational factors.

Robustness analysis confirmed structural stability beyond single-feature dominance.

# Reproducibility

The repository includes:

Data preprocessing scripts

Feature engineering modules

Model training pipeline

Hyperparameter optimization scripts

SHAP analysis notebooks

Ablation study implementation

Random seeds are fixed for reproducibility.

