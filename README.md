# 🏥 Healthcare Premium Prediction

A machine learning project to predict annual healthcare insurance premium amounts based on policyholder demographics, medical history, lifestyle, and financial attributes. The project includes thorough error analysis and model segmentation to handle age-based behavioral differences in the data.

---
[Demo App Link](https://cipherx-healthcare-cost-prediction.streamlit.app/)

## 📁 Project Structure

```
Healthcare_premium_pred/
│
├── Healthcare_premium_pred.ipynb           # Full dataset EDA, feature engineering & modeling
├── Healthcare_premium_pred2.ipynb          # Dataset splitting (young vs rest)
├── Healthcare_premium_pred_young.ipynb     # Modeling for age ≤ 25 segment
├── Healthcare_premium_pred_young_with_gr.ipynb  # Young segment + Genetical Risk feature
├── Healthcare_premium_pred_rest.ipynb      # Modeling for age > 25 segment
└── Healthcare_premium_pred_rest_with_gr.ipynb   # Rest segment + Genetical Risk feature
```

---

## 📊 Dataset Features

| Feature | Description |
|---|---|
| `age` | Age of the policyholder |
| `number_of_dependants` | Number of dependants |
| `income_lakhs` | Annual income in lakhs (INR) |
| `income_level` | Categorical income bracket |
| `gender` | Male / Female |
| `region` | Geographic region |
| `marital_status` | Married / Unmarried |
| `bmi_category` | BMI classification |
| `smoking_status` | Smoking / No Smoking / Occasional |
| `employment_status` | Employment type |
| `medical_history` | Diseases (e.g., Diabetes, Heart Disease, Hypertension, Thyroid) |
| `insurance_plan` | Bronze / Silver / Gold |
| `genetical_risk` | Genetic risk score (added in later iteration) |
| `annual_premium_amount` | **Target variable** — premium in INR |

---

## 🔄 Project Workflow

### 1. Data Cleaning & Preprocessing
- Removed null values and duplicate records
- Fixed negative values in `number_of_dependants` using `abs()`
- Capped age outliers at 100 and income outliers at the 99.9th percentile
- Standardized inconsistent labels in `smoking_status` (e.g., `"Smoking=0"`, `"Does Not Smoke"` → `"No Smoking"`)

### 2. Exploratory Data Analysis (EDA)
- Univariate distribution analysis on all numeric and categorical features
- Scatter plots for each numeric feature vs. `annual_premium_amount`
- Bivariate analysis: Income Level vs. Insurance Plan (crosstab + heatmap)
- Percentage distribution bar charts for all 9 categorical columns

### 3. Feature Engineering
- Split `medical_history` into `disease1` and `disease2`
- Assigned domain-based **risk scores** to each disease:
  - Heart Disease → 8, Diabetes → 6, High Blood Pressure → 6, Thyroid → 5, None → 0
- Created `total_risk_score` and `normalized_risk_score`
- Ordinal encoding for `insurance_plan` (Bronze=1, Silver=2, Gold=3) and `income_level`
- One-hot encoding for nominal categorical columns

### 4. Scaling & Multicollinearity Check
- MinMax scaling on: `age`, `number_of_dependants`, `income_level`, `income_lakhs`, `insurance_plan`
- VIF (Variance Inflation Factor) analysis — dropped `income_level` to reduce multicollinearity

### 5. Model Training
Models trained and evaluated (R² score):

| Model | Notes |
|---|---|
| Linear Regression | Baseline model |
| Ridge Regression | Tested across multiple alpha values |
| XGBoost Regressor | Best performer; tuned via RandomizedSearchCV |

Hyperparameter tuning for XGBoost:
- `n_estimators`: [20, 40, 50]
- `learning_rate`: [0.01, 0.1, 0.2]
- `max_depth`: [3, 4, 5]

### 6. Error Analysis & Model Segmentation
- Computed residuals and `diff_pct` (percentage error) for all predictions
- Identified **extreme errors** where `|diff_pct| > 10%`
- Plotted feature distributions for extreme-error samples vs. overall dataset
- **Key finding:** ~97% of extreme error cases belonged to policyholders aged ≤ 25
- **Resolution:** Split the dataset into two segments — `age ≤ 25` (young) and `age > 25` (rest) — and trained separate models for each

### 7. Genetical Risk Feature (Extended Analysis)
- Added `genetical_risk` as an additional feature for both segments
- Evaluated whether including genetic risk improved model performance

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, statsmodels
- **Environment:** Google Colab

---

## ▶️ How to Run

1. Clone the repository and open the notebooks in Google Colab or Jupyter.
2. Mount your Google Drive and update the dataset paths accordingly.
3. Run notebooks in this order:
   - `Healthcare_premium_pred.ipynb` → Full pipeline + error analysis
   - `Healthcare_premium_pred2.ipynb` → Split dataset by age group
   - `Healthcare_premium_pred_young.ipynb` / `Healthcare_premium_pred_rest.ipynb` → Segment-wise models
   - `*_with_gr.ipynb` variants → Models with Genetical Risk feature

---

## 💡 Key Insights

- **Age** is the most influential factor in premium prediction, especially for those under 25, where prediction behavior differs significantly from older policyholders.
- **Smoking status** and **medical history** (especially heart disease and diabetes) are strong drivers of premium amounts.
- **Model segmentation** by age group substantially reduced prediction error, demonstrating the importance of error analysis in real-world ML workflows.
- Including a **Genetical Risk** score improves model interpretability and can marginally improve accuracy for certain age segments.

---

## 📌 Future Improvements

- Deploy as a web app using Streamlit or Flask
- Add SHAP values for model explainability
- Experiment with ensemble approaches (stacking/blending) across age segments
- Include more granular genetic risk data if available
