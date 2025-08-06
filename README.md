# 🏥 Health Insurance Premium Prediction - Advanced ML Pipeline

This project focuses on **predicting annual health insurance premiums** using demographic, financial, and health-related features. The pipeline explores robust **EDA**, **Feature Engineering**, **ML Modeling**, and **Error Analysis** across **three different customer segments**:

- `premiums.xlsx` (All users)
- `premiums_rest.xlsx` (All except younger users)
- `premiums_young.xlsx` (Younger user segment)

---

## 📁 Dataset Overview

Each dataset contains attributes such as:

| Feature                | Description                                 |
|------------------------|---------------------------------------------|
| `age`                  | Age of policyholder                         |
| `gender`               | Gender                                      |
| `region`               | Geographic region                           |
| `income_lakhs`         | Income in Lakhs                             |
| `income_level`         | Binned income level                         |
| `employment_status`    | Employment type                             |
| `smoking_status`       | Smoker status                               |
| `medical_history`      | Health condition(s)                         |
| `insurance_plan`       | Bronze/Silver/Gold                          |
| `annual_premium_amount`| 💡 **Target Variable**                      |

---

## 🔄 Data Preprocessing

- Removed nulls and duplicates
- Replaced negative values in `number_of_dependants`
- Capped outliers using **IQR** and **quantile filtering**
- Fixed inconsistencies in values (e.g. `'Smoking=0'` → `'No Smoking'`)

---

## 📊 Exploratory Data Analysis (EDA)

- 📦 **Boxplots** for outlier detection  
- 📈 **Histograms** with KDE for distribution shape  
- 🟢 **Scatterplots** for feature vs premium correlation  
- 📉 **Bar plots & Heatmaps** for bivariate categorical insight

---

## 🧠 Feature Engineering

- Parsed `medical_history` into `disease1` & `disease2`
- Assigned **Risk Scores** to each disease:
  ```python
  risk_score = {
      "diabetes": 6,
      "heart disease": 8,
      "high blood pressure": 6,
      "thyroid": 5,
      "no disease": 0,
      "none": 0
  }
Created a normalized total risk score feature
Mapped ordinal features (e.g. plan tiers, income levels)
One-hot encoded nominal features
📉 Correlation & Multicollinearity
Generated correlation heatmap
Used VIF (Variance Inflation Factor) to detect and remove multicollinear features (income_level)
📏 Feature Scaling
Scaled using MinMaxScaler for:
age
number_of_dependants
income_level
income_lakhs
insurance_plan
🤖 Model Training
Linear Regression
Interpretable and stable
Feature importance visualized
Ridge Regression
Compared multiple alpha values
XGBoost Regressor
Outperformed linear models in accuracy
Tuned using RandomizedSearchCV
🧪 Evaluation Metrics
R² Score
MSE and RMSE
Feature importances from both linear & tree-based models
❌ Error Analysis
Identified high-error predictions using %diff
Segmented out extreme errors > 10%
Compared distributions (all data vs extreme error group)
🔍 Finding
Most extreme prediction errors occur in customers below age 25, even after scaling correction.
✅ Solution
Implemented segmented modeling:
premiums_rest.xlsx: General population
premiums_young.xlsx: Dedicated model for younger users
🧩 Next Steps
Apply age-based model segmentation in production
Test stacked models or ensemble approaches
Try log/power transform on target to reduce skewness
Explore SHAP/LIME for XGBoost interpretability
🛠 Tech Stack
Python
NumPy, Pandas
Seaborn, Matplotlib
Scikit-learn
XGBoost
Statsmodels
