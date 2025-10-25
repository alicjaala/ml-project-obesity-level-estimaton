import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 3. Przetwarzanie danych ---
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

# --- Funkcja do trenowania i ewaluacji ---
def train_evaluate(X_tr, y_tr, X_te, y_te, description=""):
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    model.fit(X_tr, y_tr)

    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)

    print(f"\n--- {description} ---")
    print(f"Train Accuracy: {accuracy_score(y_tr, y_pred_train):.4f}")
    print(f"Test Accuracy:  {accuracy_score(y_te, y_pred_test):.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_te, y_pred_test, digits=4))


# --- 4. Model na oryginalnych danych ---
print(f"Początkowy rozkład danych: {y_train.value_counts().to_dict()}")
train_evaluate(X_train, y_train, X_test, y_test, description="Oryginalne dane")


# --- 5. Oversampling z SMOTE ---
#SMOTE działa tylko na danych numerycznych, więc trzeba wykonać preprocessing wcześniej

X_train_preprocessed = preprocessor.fit_transform(X_train)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)

print(f"\nRozkład klas po SMOTE: {np.bincount(y_train_smote.factorize()[0]) if hasattr(y_train_smote, 'factorize') else np.bincount(pd.factorize(y_train_smote)[0])}")

# Trening i ewaluacja na zbalansowanych danych (SMOTE)
model_smote = LogisticRegression(max_iter=1000, random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

# Ewaluacja
X_test_preprocessed = preprocessor.transform(X_test)
y_test_pred_smote = model_smote.predict(X_test_preprocessed)

print("\n--- Po oversamplingu SMOTE ---")
print(f"Test Accuracy:  {accuracy_score(y_test, y_test_pred_smote):.4f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred_smote, digits=4))


# --- 6. Undersampling ---
rus = RandomUnderSampler(random_state=42)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_train_rus, y_train_rus = rus.fit_resample(X_train_preprocessed, y_train)

print(f"\nRozkład klas po undersamplingu: {np.bincount(pd.factorize(y_train_rus)[0])}")

# Trening i ewaluacja na undersamplowanych danych
model_rus = LogisticRegression(max_iter=1000, random_state=42)
model_rus.fit(X_train_rus, y_train_rus)

y_test_pred_rus = model_rus.predict(X_test_preprocessed)

print("\n--- Po undersamplingu ---")
print(f"Test Accuracy:  {accuracy_score(y_test, y_test_pred_rus):.4f}")
print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred_rus, digits=4))
