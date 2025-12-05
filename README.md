# Loan Default Prediction Project #

A complete end-to-end Machine Learning pipeline for predicting loan default risk using Python, Scikit-Learn, and XGBoost. 
This repository demonstrates a full ML lifecycle, including data ingestion, cleaning, EDA, feature engineering, model training, evaluation, and final prediction output. 

## Project Overview ##

Loan default prediction is a key use-case in financial risk modeling.
The goal of this project is to:

Build a reliable ML pipeline that can run end-to-end in one script

Produce cleaned data, EDA visualizations, engineered features, trained models, and a submission file

Compare multiple models and select the best one based on ROC-AUC

Demonstrate reproducible, professional ML workflow

The entire pipeline is implemented in:
```
 loan_pipeline_full.py
```

## Repository Structure ##

```
Loan-Default-Prediction_Redi/
│
├── loan_pipeline_full.py          # Full automated ML pipeline
├── train_LZV4RXX.csv              # Training dataset
├── test_XXXXXX.csv                # Test dataset
│
├── cleaned_loan_data_final.csv    # Cleaned training data
├── model_ready_loan_data.csv      # Final dataset used for modeling
│
├── submission.csv                 # Final model predictions for test set
│
├── plots/                         # All EDA & model evaluation outputs
│   ├── hist_loan_amount.png
│   ├── corr_heatmap.png
│   ├── cm_randomforest.png
│   ├── roc_comparison.png
│   ├── eval_metrics_summary.json
│   └── ... many more
│
└── EDA_Report.md            # Additional EDA summary
```
 
 ## Exploratory Data Analysis (EDA) ##

The script automatically generates visualizations including:

### Distribution Plots ###

Loan Amount

Target Distribution by Categories

Default rate by Education

## Proof submission count ##

(plots available inside /plots/)

### Summary file ###
```
Elham_EDA_Report.md
```
## Feature Engineering ##

Features added:

loan_to_asset

loan_amount_log

Outlier flags (column-wise binary indicators)

Automatic selection of meaningful categorical features

**Output**
```
model_ready_loan_data.csv
```

## Train/Validation Split ##

80/20 split

Stratified by target

Ensures both classes represented consistently

## Preprocessing Pipeline ##

Using Scikit-Learn’s ColumnTransformer:

### Numeric Pipeline ###

Median imputation

Standard scaling

### Categorical Pipeline ###

Mode imputation

One-hot encoding with dynamic category expansion

# Model Training & Evaluation #

### Models compared ###

| Model                 | Accuracy | ROC-AUC | F1     |
|-----------------------|---------:|--------:|-------:|
| LogisticRegression    |    0.555 |  0.587  | 0.517  |
| RandomForest          |    0.582 |  0.543  | 0.357  |
| XGBoost               |    —     |    —    | —      |

### Best Model Selected: Logistic Regression (based on ROC-AUC) ###

# Evaluation Plots #

ROC curve comparison

Feature Importance (RF)

# Final Model Training + Submission Generation #

Best model is retrained on full dataset

Saved as:
```
best_loan_model.joblib
```
Test data processed + predictions created

Output file:
```
submission.csv
```

# How to Run the Pipeline #
```
python loan_pipeline_full.py
```
# Requirements #
```
pip install -r requirements.txt
```
# Contributors #
| Name              | Contribution                                                                                   |
|-------------------|------------------------------------------------------------------------------------------------|
| **Yuni**          | Performed initial Exploratory Data Analysis (EDA)                                              |
| **Ktja**          | Led evaluation, interpretation of model results, and provided feedback                         |
| **Elena**         | Performed full data cleaning, built automated ML pipeline, feature engineering, modeling, and documentation |

# Future Improvements #

Hyperparameter tuning (GridSearchCV or Optuna)

Try advanced algorithms (LightGBM, CatBoost)

SMOTE / class imbalance handling

SHAP explainability dashboard

Deployment-ready API version

# Final Note #

This project demonstrates a full end-to-end ML workflow and is structured for learning, interviews, and real-world demonstration
