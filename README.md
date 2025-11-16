# Cafe Sales Prediction

This repository contains a data science project focused on cleaning, analyzing, and modeling a "dirty" dataset of cafe sales (`dirty_cafe_sales.csv`). The primary goal is to preprocess the data extensively and then build and evaluate several regression models to predict the `Total_Spent` by customers.

## ⚠️ Correction: Identifying and Fixing Target Leakage

An earlier version of this project (see `CafeML_old_version.ipynb`) contained a significant methodological error: **target leakage**.

* **The Flaw:** Missing values for `Quantity` and `Price_Per_Unit` were imputed using the target variable, `Total_Spent` (e.g., `Price = Total / Qty`). This "leaked" the answer to the model, resulting in an unrealistic R² score of 0.9999.
* **The Fix:** This version of the notebook (`CafeML.ipynb`) corrects this flaw. The leaky imputation has been removed and replaced with a valid, non-leaky strategy:
    1.  Rows with a missing target (`Total_Spent`) are dropped.
    2.  Missing `Price_Per_Unit` and `Quantity` are imputed using the **median** value for that specific `Item`.
    3.  This approach prevents any information from the target variable from leaking into the features.

The original, flawed notebook and its report are preserved (`CafeML_old_version.ipynb` and `CafeML_old_version.pdf`) for transparency and to demonstrate the process of identifying and correcting this common and critical error in machine learning.

---

## 📈 Project Workflow

The project follows a standard data science pipeline:

### 1. Data Cleaning and Imputation
The initial dataset contained significant inconsistencies, including string errors (`'ERROR'`, `'UNKNOWN'`) and missing values (`NaN`). The cleaning process involved:
* Standardizing error strings to `NaN`.
* Converting key columns (`Total_Spent`, `Quantity`, `Price_Per_Unit`) to numeric types.
* Dropping rows where the `Transaction_Date` was missing.
* **Valid Imputation (Corrected):**
    * Rows with a missing target variable (`Total_Spent`) were dropped.
    * Missing `Price_Per_Unit` and `Quantity` were imputed using the **median** value for that specific `Item`. This is a robust, non-leaky strategy.
    * Missing `Item` names were then filled by cross-referencing `Price_Per_Unit` where the price was unique to a specific item (e.g., 1.0 for Cookie, 1.5 for Tea).
* Dropping any final unrecoverable rows to ensure a 100% clean dataset.

### 2. Exploratory Data Analysis (EDA) & Feature Engineering
Before modeling, an EDA was performed to identify informative features.

* **Dropped Features:**
    * `Payment_Method`: Dropped as spending was almost evenly divided (≈33% each), providing little predictive value.
    * `Location`: Dropped as spending was nearly a 50/50 split between 'In-Store' and 'Takeaway'.
* **Retained Features:**
    * `Item`: Kept, as spending varied significantly across items.
    * `Quantity` & `Price_Per_Unit`: Kept, as these are the primary drivers of the target.
* **Feature Engineering:**
    * Extracted `Day` (of the week) and `Month` from the `Transaction_Date` column to capture temporal patterns.

### 3. Dataset Balance & Target Variable
The target variable, `Total_Spent`, was plotted. The resulting histogram showed a "nicely distributed" (slight positive-to-zero skew) range. This balance confirmed that standard regression metrics (MAE, MSE, R², etc.) are appropriate for evaluation.

### 4. Feature Selection & Pre-Modeling
* The `Transaction_ID` and original `Transaction_Date` columns were dropped.
* Categorical features (`Item`, `Day`) were **one-hot encoded** to prepare them for machine learning.
* The data was split into training (60%) and testing (40%) sets using `random_state=42` for reproducibility.
* Feature selection methods (`SelectKBest` and `RFE`) were tested, both identifying `Quantity` and `Price_Per_Unit` as top features.

### 5. Model Building & Evaluation
Four different regression models were trained and evaluated on the full feature set. The results below are from the **corrected, non-leaky** data.

| Model | MAE | MSE | RMSE | R² | PinballLoss (q=0.9) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 1.440064 | 4.246326 | 2.060661 | 0.882059 | 0.692111 |
| **MLP Regressor** | 0.317991 | 1.078842 | 1.038673 | 0.970035 | 0.134566 |
| **KNN Regressor** | 0.708742 | 1.847085 | 1.359075 | 0.948698 | 0.332805 |
| **Gradient Boosting** | **0.266794** | **1.050022** | **1.024706** | **0.970836** | **0.121238** |

The final **Gradient Boosting** model was the most accurate, achieving an **R² of 0.97** on the test set, demonstrating a very strong and realistic predictive model for cafe sales.

### 6. Hyperparameter Tuning
`GridSearchCV` and `RandomizedSearchCV` were used to find the optimal parameters for Lasso Regression and KNN, which improved their performance.

## Dependencies
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

You can run this notebook in any environment that supports Jupyter, such as Google Colab, VS Code, or a local Jupyter server.