import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Wczytanie danych
data = pd.read_csv('data/obesity_data.csv')

selected_cols = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
    'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'
]
data = data[selected_cols]
X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

# 2. Przetwarzanie danych
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numerical_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

# 3. Pipeline z regresją logistyczną
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 4. Walidacja krzyżowa
#obiekt dzielący dane na 3 części zachowując proporcje między klasami
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

train_accuracies = []
test_accuracies = []

#skf.split generuje indeksy dla podziału dla zbiorów train i test
#enumarate zaczyna numerowanie od 1
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):

    #tworzę foldy treningowe i testowe na podstawie indeksów
    #iloc wybiera odpowiednie wiersze z ramki danych według indeksów
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    print(f"\n--- Fold {fold} ---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred))

# 5. Podsumowanie
print("\n=========== PODSUMOWANIE ===========")
print(f"Średnia Train Accuracy: {np.mean(train_accuracies):.4f}")
print(f"Średnia Test Accuracy:  {np.mean(test_accuracies):.4f}")
print(f"Odchylenie std (Test):  {np.std(test_accuracies):.4f}")
