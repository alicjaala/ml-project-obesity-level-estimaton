import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


data = pd.read_csv('data/obesity_data.csv')

selected_cols = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
    'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'
]
data = data[selected_cols]

X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

X_temp, X_test_raw, y_temp, y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def logistic_regression_train(X, Y, lr=0.1, epochs=1000, batch_size=64):
    n_samples, n_features = X.shape
    n_classes = Y.shape[1]

    X_b = np.hstack([np.ones((n_samples, 1)), X])
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

def evaluate_model_detailed(X, y_true, weights, encoder, name="TEST"):
    probs = predict(X, weights)
    preds = np.argmax(probs, axis=1)
    true = np.argmax(y_true, axis=1)

    acc = accuracy_score(true, preds)
    loss = cross_entropy_loss(y_true, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(true, preds, average='weighted')

    print(f"\n[{name}] Accuracy: {acc:.4f}")
    print(f"[{name}] Loss: {loss:.4f}")
    print(f"\n[{name}] Raport klasyfikacji:\n", classification_report(true, preds, target_names=encoder.categories_[0]))

X_train_proc = preprocessor.fit_transform(X_train_raw)
X_val_proc = preprocessor.transform(X_val_raw)
X_test_proc = preprocessor.transform(X_test_raw)

encoder = OneHotEncoder(sparse_output=False)
y_train_enc = encoder.fit_transform(y_train_raw.to_numpy().reshape(-1, 1))
y_val_enc = encoder.transform(y_val_raw.to_numpy().reshape(-1, 1))
y_test_enc = encoder.transform(y_test_raw.to_numpy().reshape(-1, 1))

weights_orig = logistic_regression_train(X_train_proc, y_train_enc)

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_proc, np.argmax(y_train_enc, axis=1))
y_smote_enc = np.eye(y_train_enc.shape[1])[y_smote]
weights_smote = logistic_regression_train(X_smote, y_smote_enc)

rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train_proc, np.argmax(y_train_enc, axis=1))
y_rus_enc = np.eye(y_train_enc.shape[1])[y_rus]
weights_rus = logistic_regression_train(X_rus, y_rus_enc)

print("\nEvaluation on TEST set")
print("Original model:")
evaluate_model_detailed(X_test_proc, y_test_enc, weights_orig, encoder, name="TEST - Original")
print("Class distribution - original data:")
print(np.bincount(np.argmax(y_train_enc, axis=1)))

print("\nSMOTE model:")
evaluate_model_detailed(X_test_proc, y_test_enc, weights_smote, encoder, name="TEST - SMOTE")
print("\nClass distribution - after SMOTE:")
print(np.bincount(y_smote))


print("\nUndersampling model:")
evaluate_model_detailed(X_test_proc, y_test_enc, weights_rus, encoder, name="TEST - Undersampling")
print("\nClass distribution - after undersamplingu:")
print(np.bincount(y_rus))


