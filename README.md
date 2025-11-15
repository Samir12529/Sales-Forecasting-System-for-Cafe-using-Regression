# Cafe Sales Prediction

This repository contains a data science project focused on cleaning, analyzing, and modeling a "dirty" dataset of cafe sales (`dirty_cafe_sales.csv`). The primary goal is to preprocess the data extensively and then build and evaluate several regression models to predict the `Total_Spent` by customers.

The entire analysis is contained within the `CafeML.ipynb` notebook.

## 📈 Project Workflow

The project follows a standard data science pipeline:

### 1. Data Cleaning and Imputation
The initial dataset contained significant inconsistencies, including string errors (`'ERROR'`, `'UNKNOWN'`) and missing values (`NaN`). The cleaning process involved:
* Standardizing error strings to `NaN`.
* Renaming columns to snake_case (e.g., `Total Spent` to `Total_Spent`).
* Dropping a small percentage (4.6%) of rows where the `Transaction_Date` was missing.
* **Strategic Imputation:**
    * Filled missing `Item` names by cross-referencing `Price_Per_Unit` where the price was unique to a specific item (e.g., 1.0 for Cookie, 1.5 for Tea).
    * Filled remaining ambiguous `Item` NaNs using a weighted random choice based on the distribution of known items at that price point.
    * Cross-imputed missing values for `Total_Spent`, `Quantity`, and `Price_Per_Unit` by leveraging the formula: `Total_Spent = Quantity * Price_Per_Unit`.
    * Filled remaining `Price_Per_Unit` NaNs by mapping them to the known price of the `Item` in that row.
* Dropped a final 55 unrecoverable rows where key information was still missing.

### 2. Exploratory Data Analysis (EDA) & Feature Engineering
Before modeling, an EDA was performed to identify informative features.

* **Dropped Features:**
    * `Payment_Method`: Dropped as spending was almost evenly divided (≈33% each), providing little predictive value.
    * `Location`: Dropped as spending was nearly a 50/50 split between 'In-Store' and 'Takeaway'.
* **Retained Features:**
    * `Item`: Kept, as spending varied significantly across items (from 4% to 21.7%).
    * `Quantity` & `Price_Per_Unit`: Kept, as both showed a strong positive correlation with `Total_Spent`.
* **Feature Engineering:**
    * Extracted `Day` (of the week) and `Month` from the `Transaction_Date` column to capture temporal patterns. These new features were retained as they showed a considerable (though not dominant) relationship with sales.

### 3. Dataset Balance & Target Variable
The target variable, `Total_Spent`, was plotted. The resulting histogram showed a "nicely distributed" (slight positive-to-zero skew) range from 1 to 25. This balance confirmed that standard regression metrics (MAE, MSE, R², etc.) are appropriate for evaluation.

### 4. Feature Selection & Pre-Modeling
* The `Transaction_ID` and original `Transaction_Date` columns were dropped.
* Categorical features (`Item`, `Day`) were **one-hot encoded** to prepare them for machine learning.
* The data was split into training (60%) and testing (40%) sets using `random_state=42` for reproducibility.
* Feature selection methods (`SelectKBest` and `RFE`) were tested, both identifying the top 5 most predictive features.

### 5. Model Building & Evaluation
Four different regression models were trained and evaluated on the full feature set.

| Model | MAE | MSE | RMSE | R² | PinballLoss (q=0.9) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 1.292841 | 3.260440 | 1.805669 | 0.909003 | 0.633414 |
| **Lasso Regression (Tuned)** | 1.287621 | 3.251464 | 1.803182 | 0.909254 | 0.630333 |
| **KNN Regressor (Base)** | 0.451104 | 0.497501 | 0.705338 | 0.986115 | 0.229514 |
| **KNN Regressor (Tuned)** | 0.089367 | 0.078033 | 0.279345 | 0.997822 | 0.044664 |
| **Gradient Boosting** | **0.031999** | **0.002167** | **0.046555** | **0.999940** | **0.016485** |

### 6. Hyperparameter Tuning
`GridSearchCV` and `RandomizedSearchCV` were used to find the optimal parameters for Lasso Regression and KNN, which improved their performance significantly.

## 🏁 Conclusion

The analysis demonstrates a complete data-cleaning and modeling workflow.
* **Gradient Boosting** was the top-performing model, achieving a near-perfect R² score of **0.9999**.
* The high accuracy across most models (especially tree-based ones like Gradient Boosting) strongly suggests that the models were able to successfully reverse-engineer the `Total_Spent = Quantity * Price_Per_Unit` relationship that was heavily implied in the cleaned data.
* Linear models (Linear, Lasso) performed worst, as they could not capture this non-linear multiplicative relationship.

## 🛠️ How to Run
This project is contained in a single Jupyter Notebook (`.ipynb`).

**Dependencies:**
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

You can run this notebook in any environment that supports Jupyter, such as Google Colab, VS Code, or a local Jupyter server.

1.  Ensure you have the required libraries installed:
    ```sh
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
2.  Place the `dirty_cafe_sales.csv` file in the correct path (the notebook assumes `/content/dirty_cafe_sales.csv`).
3.  Run all cells in the `CafeML.ipynb` notebook.