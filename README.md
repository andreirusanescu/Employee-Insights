# Homework 1 AI - Introduction to Machine Learning

**Author:** Andrei Rusanescu

This repository contains the solution for Homework 1 of the *Introduction to Machine Learning* course. The project follows a complete Machine Learning pipeline, including Exploratory Data Analysis (EDA), data preprocessing, as well as training and evaluating models for classification and regression tasks on an employee dataset.

## Project Structure
- **Task 1: Exploratory Data Analysis (EDA)**
- **Task 2: Data Preprocessing**
- **Task 3: Classification for `vacation` target**
- **Task 4: Regression for `salary` target**

---

## Task 1: Exploratory Data Analysis (EDA)
Numerical and categorical attributes were analyzed to understand data distributions and correlations:
- **Numerical attributes:** The analysis revealed a perfect linear relationship: `total_days_worked = 240 * experience_years`, indicating redundancy. Additionally, `aggregated_score` proved to be pure noise (Pearson correlation near zero with the rest of the data).
- **Categorical attributes:** The target variable for classification, `vacation`, shows a massive imbalance: ~54% *No Vacation*, ~20% *Small*, ~13% *Medium*, ~12% *Large*.
- **Major correlations:** `experience_years` is the strongest numerical predictor for `salary` (Pearson: 0.44), while `education_level` and `company_size` are the best categorical predictors for `vacation`.

## Task 2: Data Preprocessing
The preprocessing pipeline included the following steps applied to the training data and propagated to validation/test sets:
1. **Missing values imputation:** 
   - `SimpleImputer(strategy='median')` for skewed numerical attributes.
   - `SimpleImputer(strategy='mode')` for the categorical attribute `remote_work` ("Hybrid").
2. **Outlier Treatment:** Analyzed using the Interquartile Range (IQR) method. For salary prediction, capping proved counterproductive, as high salaries are valid data points that improve the model.
3. **Removing redundancies:** `total_days_worked` and `aggregated_score` were excluded.
4. **Encoding categorical variables:** `OrdinalEncoder` was used for ordinal features (`education_level`, `company_size`, `skill_bracket`), and `OneHotEncoder` for nominal ones.
5. **Standardization:** `StandardScaler` was applied to numerical variables.

## Task 3: Classification for the `vacation` target
**Goal:** Predict the length of an employee's vacation.
Evaluated models: *Decision Tree*, *Logistic Regression*, *Random Forest*, and *Gradient Boosting*.
- **Winning model:** **Gradient Boosting**. Due to its sequential nature, it better identified the minority classes (*Small* and *Medium*), which are notoriously hard to separate.
- **Performance:** The model achieved an F1-macro score of ~0.67 and an accuracy of ~76% on the test set.
- **Feature Engineering:** To improve performance, *target encodings* and new features (`edu_comp_score`, `senior_flag`, etc.) were introduced, capturing the distinct profile of highly experienced employees.

## Task 4: Regression for the `salary` target
**Goal:** Predict an employee's salary.
Analyzed models: *Linear Regression*, *Ridge (L2)*, *Lasso (L1)*, and *Gradient Boosting Regressor*.
- Linear models explained ~94.1% of the data variance (R² = 0.9412) but hit an error ceiling (MSE ~81.7M, RMSE ~$9040). L1 and L2 penalties did not bring major improvements.
- **Winning model:** **Gradient Boosting Regressor**. It successfully captured non-linear interactions, reducing the MSE by approximately 63%, reaching an **MSE of ~30.1M** and an **R² of 0.978**.
- Learning curves confirmed the stability and excellent generalization capability of the tree-based model.
