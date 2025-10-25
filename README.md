# 🧠 ML Project: Obesity Level Estimation  

## Project Overview  

This project is a **comprehensive, three-part data science pipeline** focused on predicting obesity levels based on eating habits and physical condition.  
It covers the **entire machine learning workflow** — from initial data analysis to final model optimization.  

### 🔹 Part 1: Exploratory Data Analysis (EDA)
- Load, clean, and analyze the dataset  
- Visualize key relationships and patterns using **pandas**, **matplotlib**, and **seaborn**

### 🔹 Part 2: Machine Learning Fundamentals
- Build **data preprocessing pipelines** using **scikit-learn**  
- Train baseline ML models  
- Implement **linear** and **logistic regression** from scratch using **NumPy**

### 🔹 Part 3: Model Optimization
- Improve performance using:
  - **Cross-validation**  
  - **Regularization**  
  - **Handling imbalanced data**  
  - **Hyperparameter tuning**

---

## Dataset  

**Source:** [UCI Machine Learning Repository – Estimation of Obesity Levels Based On Eating Habits and Physical Condition](https://archive.ics.uci.edu/)  

**Description:**  
The dataset contains **17 features** related to eating habits (e.g., `FAVC`, `FCVC`) and physical condition (e.g., `Age`, `Weight`, `Height`, `CALC`).  
The target variable is **`NObesity`**, which represents the obesity level category.  

# Core Features & Methodologies

## Part 1: Exploratory Data Analysis (EDA)

### Data Loading
- **Custom data loading class/method** implemented in  
  `scripts/load_data.py`

### Descriptive Statistics
- **Numerical features:**
  - Mean, median, min/max, standard deviation, percentiles, missing values
- **Categorical features:**
  - Unique classes, proportions, missing values  
  *(Implemented in `scripts/data_analysis_main.py`)*

### Visualization
*(Implemented in `scripts/generate_plots.py`)*  
- `seaborn.boxplot` & `seaborn.violinplot`  
- `seaborn.histplot` (with `hue` for conditioned histograms)  
- Error bar plots  
- `seaborn.heatmap` for correlation matrix  
- `seaborn.regplot` for linear regression analysis  

---

## Part 2: Machine Learning Fundamentals

### Preprocessing
- Implemented using:
  - `sklearn.pipeline.Pipeline`
  - `sklearn.compose.ColumnTransformer`
  - `SimpleImputer` and `OneHotEncoder`  
  *(Implemented in `scripts/pipeline.py`)*

### Baseline Models
- Training and evaluation of three scikit-learn models (e.g.):
  - Logistic Regression
  - Decision Tree
  - Support Vector Machine (SVM)  
  *(Implemented in `scripts/pipeline2.py`)*

### Custom NumPy Models
- **Linear Regression (Closed-Form):**  
  Implemented via the normal equation `(XᵀX)⁻¹Xᵀy`  
  *(scripts/closedFormula.py)*  

- **Gradient Descent:**  
  Iterative implementation for Linear/Logistic Regression with:
  - Batch processing
  - Custom cost functions (MSE / Cross-Entropy)  
  *(scripts/gradientDescent.py)*  

---

## Part 3: Model Optimization

### Cross-Validation
- 3-fold cross-validation (`KFold` / `StratifiedKFold`)  
  *(optimalisation/scikit1.py)*  

### Bias-Variance Analysis
- Learning curve plots to diagnose overfitting/underfitting  
- Tested with `PolynomialFeatures`  
  *(optimalisation/scikit2.py)*  

### Regularization
- Comparison of:
  - **L1 (Lasso)**
  - **L2 (Ridge)**  
  *(optimalisation/scikit3.py)*  

### Imbalanced Data
- Techniques:
  - **SMOTE (oversampling)**
  - **Undersampling**
- Evaluation metrics:
  - Precision, Recall, F1-score  
  *(optimalisation/scikit4.py)*  

### Hyperparameter Tuning
- **GridSearchCV** for optimal parameter selection  
  *(optimalisation/scikit5.py)*  

### Custom Optimization
- Integration of regularization and cross-validation with NumPy models  
  *(optimalisation/gradient\*.py)*  

## Technologies Used

- **Python 3.x**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn (sklearn)**
