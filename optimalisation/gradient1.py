import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import time

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

# 2. Przetwarzanie
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numerical_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_cols)
])

encoder = OneHotEncoder(sparse_output=False)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def logistic_regression_train(X, Y, X_val, Y_val, lr=0.1, epochs=500, batch_size=64):
    n_samples, n_features = X.shape
    n_classes = Y.shape[1]

    X_b = np.hstack([np.ones((n_samples, 1)), X])
    X_val_b = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

    weights = np.random.randn(n_features + 1, n_classes)

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            batch_idx = indices[start_idx:end_idx]
            X_batch = X_b[batch_idx]
            Y_batch = Y[batch_idx]

            logits = X_batch @ weights
            probs = softmax(logits)
            grad = X_batch.T @ (probs - Y_batch) / batch_size
            weights -= lr * grad

    return weights

def predict(X, weights):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X])
    logits = X_b @ weights
    return softmax(logits)

def evaluate_model(X, y_true, weights, encoder, name="TEST"):
    probs = predict(X, weights)
    preds = np.argmax(probs, axis=1)
    true = np.argmax(y_true, axis=1)
    acc = accuracy_score(true, preds)
    loss = cross_entropy_loss(y_true, probs)
    print(f"\n[{name}] Accuracy: {acc:.4f}")
    print(f"[{name}] Loss: {loss:.4f}")
    print(f"[{name}] Raport klasyfikacji:\n", classification_report(true, preds, target_names=encoder.categories_[0]))

#tak samo jak wcześniej - tworzę 3 foldy z zachowaniem proporcji klas
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold = 1
accuracies = []

for train_idx, val_idx in skf.split(X, y):          #split dzieli zbiór
    print(f"\n=========== Fold {fold} ===========")
    X_train_raw, X_val_raw = X.iloc[train_idx], X.iloc[val_idx]     #iloc wybiera odpowiednie wiersze wg indeksów
    y_train_raw, y_val_raw = y.iloc[train_idx], y.iloc[val_idx]

    # Przetwarzanie danych
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)

    y_train_encoded = encoder.fit_transform(y_train_raw.to_numpy().reshape(-1, 1))
    y_val_encoded = encoder.transform(y_val_raw.to_numpy().reshape(-1, 1))

    # Trening
    start_time = time.time()
    weights = logistic_regression_train(X_train, y_train_encoded, X_val, y_val_encoded)
    end_time = time.time()
    print(f"Czas treningu: {end_time - start_time:.2f} sekund")

    # Ewaluacja
    evaluate_model(X_val, y_val_encoded, weights, encoder, name=f"FOLD {fold}")
    probs = predict(X_val, weights)
    preds = np.argmax(probs, axis=1)
    true = np.argmax(y_val_encoded, axis=1)
    accuracies.append(accuracy_score(true, preds))

    fold += 1

print("\n=========== PODSUMOWANIE ===========")
print(f"Średnia dokładność (accuracy): {np.mean(accuracies):.4f}")
print(f"Odchylenie standardowe:         {np.std(accuracies):.4f}")
