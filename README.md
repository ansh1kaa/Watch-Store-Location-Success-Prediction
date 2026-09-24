# Watch Store Location Success Prediction
**Machine Learning Decision-Support System for Retail Store Expansion**

> **Academic Disclosure Notice:**  
> Synthetic dataset created for academic machine-learning practice. It does not represent actual watch-store businesses or real-world locations.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Objective](#objective)
4. [Dataset Description](#dataset-description)
5. [Features](#features)
6. [Data Generation Methodology & Leakage Safeguards](#data-generation-methodology--leakage-safeguards)
7. [Data Preprocessing](#data-preprocessing)
8. [Models Used](#models-used)
9. [Evaluation Metrics & Actual Results](#evaluation-metrics--actual-results)
10. [Model Selection Decision](#model-selection-decision)
11. [Web Application & Vercel API Endpoint](#web-application--vercel-api-endpoint)
12. [Project Structure](#project-structure)
13. [Local Setup & Run Instructions](#local-setup--run-instructions)
14. [Vercel Deployment Instructions](#vercel-deployment-instructions)
15. [Limitations](#limitations)
16. [Future Improvements](#future-improvements)

---

## 🎯 Project Overview
Opening a physical retail watch store requires substantial capital investment in long-term lease commitments, high-end interior fitting, security infrastructure, inventory, and staffing. Choosing an sub-optimal location leads to low foot traffic, high operating overhead, and potential site failure.

This project implements an end-to-end binary classification machine-learning system using **Python, Pandas, NumPy, and Scikit-learn** to predict whether a proposed store location will be **Profitable (1)** or **Not Profitable (0)**. In addition to binary classification, the system outputs dynamic **Probability of Profitability (%)** scores using `predict_proba()`, enabling business stakeholders to perform multi-site comparison ranking and sensitivity analysis.

---

## 💼 Business Problem
A retail business owner wants an objective, data-driven decision-support tool to evaluate potential store locations before committing capital.

Without quantitative evaluation, retail expansion decisions rely heavily on subjective intuition, increasing financial exposure. This project models location, demographic, competitive, customer, and cost factors to provide empirical probability estimates and decision support.

---

## 🎯 Objective
1. Classify candidate retail locations into binary outcome categories: `1` (Profitable) or `0` (Not Profitable).
2. Compute dynamic probability estimates (e.g., `93.00%` probability of profitability) using `predict_proba()`.
3. Prevent data leakage by separating latent economic variables from observable input features.
4. Provide practical business tools: Multi-Site Location Ranking, What-If Sensitivity Analysis, and Financial Break-Even Estimates.

---

## 📊 Dataset Description
The dataset contains **5,000 synthetic location records** (`data/watch_store_locations.csv`) generated with realistic retail microeconomic relationships.

- **Total Records:** 5,000
- **Total Columns:** 13 (12 input features + 1 target variable)
- **Target Variable:** `profitable` (`1` = Profitable: 58.68%, `0` = Not Profitable: 41.32%)
- **Data Format:** Standard clean CSV without inline header comments

---

## 📐 Features

| Feature Name | Type | Range / Unit | Description |
| :--- | :--- | :--- | :--- |
| `population` | Integer | 10,000 – 500,000 | Resident population within 5km radius |
| `average_monthly_income` | Float | $1,500 – $12,000 | Average monthly household income |
| `daily_foot_traffic` | Integer | 500 – 25,000 | Estimated daily pedestrian/foot traffic |
| `nearby_competitors` | Integer | 0 – 15 | Competing watch/jewelry stores within 3km |
| `monthly_rent` | Float | $1,000 – $20,000 | Proposed monthly lease cost |
| `distance_to_mall_km` | Float | 0.1 – 15.0 km | Distance to nearest major shopping mall |
| `nearby_retail_stores` | Integer | 5 – 150 | Complementary retail stores in vicinity |
| `estimated_monthly_customers` | Integer | 100 – 3,500 | Projected converting monthly visitors |
| `average_purchase_value` | Float | $40 – $850 | Expected average order value per transaction |
| `monthly_operating_cost` | Float | $2,000 – $25,000 | Non-rent monthly operational overhead |
| `local_demand_score` | Float | 1.0 – 10.0 | Demand index for luxury/fashion watches |
| `target_age_group_score` | Float | 1.0 – 10.0 | Demographic alignment score for target age group |

---

## 🔬 Data Generation Methodology & Leakage Safeguards

### Data Generation Logic
Latent revenue and cost dynamics are computed during data generation incorporating primary attributes (`foot_traffic`, `income`, `operating_cost`, `rent`) and modest realistic multipliers for secondary attributes (`population`, `distance_to_mall_km`, `nearby_retail_stores`). Gaussian noise is added to create a realistic, non-deterministic decision boundary.

### Strict Data Leakage Safeguards
- Latent variables (`latent_revenue`, `latent_total_cost`, `latent_net_profit`, `adjusted_profit`) were **never stored** in the CSV dataset or passed to Scikit-learn models.
- Derived profit features like `monthly_profit` or `profit_margin` do **NOT** exist in model features.
- The model learns exclusively from raw, observable location, demographic, and cost attributes.

---

## 🧹 Data Preprocessing & Train/Test Split

1. **Data Quality Check:** Checked for missing values (0 missing) and duplicate rows (0 duplicates).
2. **Stratified Train/Test Split:** Used `train_test_split(test_size=0.2, random_state=42, stratify=y)` (4,000 train / 1,000 test records) to preserve exact target proportions across splits.
3. **Feature Scaling:** `StandardScaler` was fitted **ONLY on X_train** and then transformed X_test to prevent data leakage. Scaled features were supplied to Logistic Regression; raw unscaled features were supplied to Random Forest Classifier.

---

## 🤖 Models Used

1. **Logistic Regression (Baseline):** Parametric linear classifier trained on scaled feature inputs.
2. **Random Forest Classifier (Primary):** Non-linear ensemble model with 100 decision trees capable of learning complex feature interactions without overfitting.

---

## 📊 Evaluation Metrics & Actual Results

All evaluation metrics were calculated from model execution on the **1,000-sample test set**.

### Actual Test Set Performance Metrics:
| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 91.10% | 91.64% | 93.36% | 92.49% | 0.9729 |
| **Random Forest Classifier** | **93.70%** | **94.11%** | **95.23%** | **94.67%** | **0.9850** |

---

## 🏆 Model Selection Decision

> **Model Selection Statement:**  
> **Random Forest Classifier** was selected for inference because it achieved the higher **ROC-AUC (0.9850 vs 0.9729)** on this held-out test evaluation, demonstrating superior ranking performance across classification thresholds.

---

## 🌐 Web Application & Vercel API Endpoint

The repository includes a minimal, Vercel-compatible web application interface (`index.html`) backed by a pure Python Serverless Function (`api/predict.py`).

### API Endpoint (`POST /api/predict`)
- **Request Body (JSON):**
```json
{
  "action": "predict",
  "model_type": "rf",
  "location_data": {
    "population": 180000,
    "average_monthly_income": 6500.0,
    "daily_foot_traffic": 14000,
    "nearby_competitors": 3,
    "monthly_rent": 6500.0,
    "distance_to_mall_km": 1.2,
    "nearby_retail_stores": 45,
    "estimated_monthly_customers": 1200,
    "average_purchase_value": 320.0,
    "monthly_operating_cost": 7500.0,
    "local_demand_score": 8.2,
    "target_age_group_score": 7.5
  }
}
```
- **Response Body (JSON):**
```json
{
  "prediction": "PROFITABLE",
  "label": 1,
  "probability_percent": 93.0,
  "probability_raw": 0.93,
  "model_used": "Random Forest Classifier",
  "financial_estimates": {
    "est_monthly_revenue": 38400.0,
    "total_monthly_cost": 14000.0,
    "est_monthly_net_profit": 24400.0,
    "initial_setup_cost": 150000,
    "breakeven_months": 6.1
  }
}
```

---

## 📁 Project Structure

```
Watch-Store-Location-Success-Prediction/
├── api/
│   └── predict.py                        # Vercel Serverless Python Handler
├── data/
│   └── watch_store_locations.csv         # 5,000-record clean CSV dataset
├── models/
│   └── trained_models.pkl                # Serialized model artifacts & scaler
├── notebooks/
│   └── watch_store_location_prediction.ipynb # Executable 22-section notebook
├── src/
│   ├── generate_dataset.py                # Dataset generator script
│   ├── preprocess.py                      # Quality check, scaler & stratified split
│   ├── train_eval.py                      # Classifier training & evaluation
│   └── predict.py                         # Probability inference & business tools
├── index.html                             # Vercel Static Web Interface
├── vercel.json                            # Vercel configuration routing
├── README.md                              # Technical project documentation
├── requirements.txt                       # Project dependencies
└── .gitignore                             # Git exclusion rules
```

---

## ⚡ Local Setup & Run Instructions

### 1. Environment Setup
```bash
git clone https://github.com/ansh1kaa/Watch-Store-Location-Success-Prediction.git
cd Watch-Store-Location-Success-Prediction
pip install -r requirements.txt
```

### 2. Running Scripts via Command Line
```bash
# Step 1: Generate dataset (5,000 records)
python src/generate_dataset.py

# Step 2: Preprocess, train models & display empirical test evaluation
python src/train_eval.py

# Step 3: Run inference demo & business decision output
python src/predict.py
```

### 3. Local Web Testing
Serve static files and Python API locally using Python's HTTP server:
```bash
python -m http.server 8000
```
Then navigate to `http://localhost:8000` in your web browser.

---

## 🚀 Vercel Deployment Instructions

1. **Push to GitHub:** Ensure your latest commits are pushed to `main`.
2. **Import to Vercel:**
   - Log in to your [Vercel Dashboard](https://vercel.com).
   - Click **Add New** > **Project** and import `ansh1kaa/Watch-Store-Location-Success-Prediction`.
   - Vercel automatically detects `vercel.json` and configures the static frontend and `@vercel/python` Serverless Function (`api/predict.py`).
3. **Deploy:** Click **Deploy**. Your app and API will be live instantly!

---

## ⚠️ Limitations & Academic Disclaimers
1. **Synthetic Dataset Notice:** The dataset is synthetic and created for academic machine-learning practice. It does not represent actual watch-store businesses or real-world locations.
2. **Academic Decision Support:** Predictions are academic decision-support estimates based on synthetic data and do NOT guarantee real-world commercial profitability.
3. **Static Scope:** Does not capture dynamic inflation rates, localized macroeconomic shifts, or multi-year competitor entry.

---

## 🔮 Future Improvements
1. Integrate real-world GIS traffic and foot-fall API data.
2. Add time-series forecasting for multi-year lease ROI analysis.
3. Implement hyperparameter optimization using `GridSearchCV`.
