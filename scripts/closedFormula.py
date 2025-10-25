import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# 1. Wczytanie danych
data = pd.read_csv('data/obesity_data.csv')

selected_cols = [
    'Gender', 'Age', 'Height', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
    'FAF', 'TUE', 'CALC', 'MTRANS'
]
target_col = 'Weight'
data = data[selected_cols + [target_col]]

X = data.drop(target_col, axis=1)
y = data[target_col].to_numpy()

# 2. Preprocessing
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# 3. Pomocnicze funkcje
def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def linear_regression_train(X, y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

def linear_regression_predict(X, W):
    return X @ W

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 4. K-FOLD cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

mse_scores = []
rmse_scores = []

fold = 1
for train_index, test_index in kf.split(X):
    print(f"\n--- Fold {fold} ---")
    X_train_raw, X_test_raw = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Preprocessing (fit na treningowym, transform na obu)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Bias
    X_train_b = add_bias(X_train)
    X_test_b = add_bias(X_test)

    # Trening
    W = linear_regression_train(X_train_b, y_train)

    # Predykcja
    y_pred = linear_regression_predict(X_test_b, W)

    # Ewaluacja
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f} kg")
    fold += 1

# 5. Średnie wyniki
print("\n--- Średnie wyniki K-Fold ---")
print(f"Średni MSE: {np.mean(mse_scores):.4f} ± {np.std(mse_scores):.4f}")
print(f"Średni RMSE: {np.mean(rmse_scores):.4f} kg ± {np.std(rmse_scores):.4f} kg")
