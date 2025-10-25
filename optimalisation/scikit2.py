import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Wczytanie danych ---
data = pd.read_csv('data/obesity_data.csv')

selected_cols = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
    'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'
]
data = data[selected_cols]

X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

# --- 2. Podział danych 80% trening / 20% test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- 3. Przetwarzanie danych ---
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

poly_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),     #tworzy nowe cechy wielomianowe
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

preprocessor_poly = ColumnTransformer([
    ('num', poly_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# --- 4. Modele ---
models = {
    "Regresja Logistyczna": Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ]),
    "Regresja logistyczna z cechami wielomianowymi": Pipeline([
        ('preprocessor', preprocessor_poly),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
}

# --- 5. Trening i ewaluacja ---
for model_name, model_pipeline in models.items():
    print(f"\n=============== {model_name.upper()} ===============")

    model_pipeline.fit(X_train, y_train)

    # Ewaluacja na zbiorze treningowym
    y_train_pred = model_pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Ewaluacja na zbiorze testowym
    y_test_pred = model_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))
