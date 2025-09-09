# MuhammadAdnan-FYP

# File
**Edible vs. Poisonous Mushroom Classification from Categorical Traits** 


---

# Overview
This notebook develops and evaluates multiple machine learning models for classifying mushrooms as edible or poisonous using the UCI Mushroom dataset. The pipeline includes data loading, preprocessing (handling missing values, label encoding), exploratory data analysis, model training, and evaluation. The aim is to identify the most effective model for accurate classification.

---

# Dataset
- **Path/URL:** https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data  
- **Target column:** `class`  
- **Feature column(s):** All remaining 22 columns including `cap-shape`, `odor`, `gill-color`, `population`, etc.  
- **Feature count/types:** 22 categorical features, all label-encoded

---

# Features & Preprocessing
- Missing values replaced using mode
- Label Encoding applied to all categorical columns using `LabelEncoder()`

---

# Models
- `GaussianNB()` – Naive Bayes
- `SVC(kernel='rbf', probability=True)` – Support Vector Machine
- `XGBClassifier(eval_metric='logloss')` – Extreme Gradient Boosting (XGBoost)

---

# Evaluation
- **Metrics:** `accuracy_score`, `cohen_kappa`, `mcc`, `balanced_accuracy`, `brier_score`, `roc_auc_score`
- **Visualizations:** 
  - Countplot for class distribution  
  - Correlation heatmap of label-encoded features  
  - Pairplot of selected features (`cap-shape`, `odor`, `gill-color`, etc.)
- **Tuning:** No tuning (e.g., GridSearchCV) applied

---

# Statistical Comparison
- **Stats:** `10 fold cross validation`, `t-test`, `Wilcoxon Test`

---

# Environment & Requirements
- **Libraries:**  
  `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`, `xgboost`
- **Install example:**
  ```bash
  pip install pandas numpy seaborn matplotlib scikit-learn xgboost
  ```

