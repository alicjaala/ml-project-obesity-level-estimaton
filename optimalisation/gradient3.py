import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
import time


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

X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

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

X_train = preprocessor.fit_transform(X_train_raw)

X_val = preprocessor.transform(X_val_raw)
X_test = preprocessor.transform(X_test_raw) 

encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train_raw.to_numpy().reshape(-1, 1))
y_val_encoded = encoder.transform(y_val_raw.to_numpy().reshape(-1, 1))
y_test_encoded = encoder.transform(y_test_raw.to_numpy().reshape(-1, 1))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    eps = 1e-15 
    y_pred = np.clip(y_pred, eps, 1 - eps) 
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

def logistic_regression_train(X, Y, X_val, Y_val, lr=0.1, epochs=1000, batch_size=64, reg_lambda=0.0, reg_type=None):
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

            if reg_type == 'l2':
                grad[1:] += reg_lambda * weights[1:]  
            elif reg_type == 'l1':
                grad[1:] += reg_lambda * np.sign(weights[1:]) 

            weights -= lr * grad     

        if epoch % 100 == 0 or epoch == epochs - 1:
            train_probs = softmax(X_b @ weights)
            val_probs = softmax(X_val_b @ weights)
            train_loss = cross_entropy_loss(Y, train_probs)
            val_loss = cross_entropy_loss(Y_val, val_probs)
            train_acc = accuracy_score(np.argmax(Y, axis=1), np.argmax(train_probs, axis=1))
            val_acc = accuracy_score(np.argmax(Y_val, axis=1), np.argmax(val_probs, axis=1))
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

    return weights


def predict(X, weights):
    X_b = np.hstack([np.ones((X.shape[0], 1)), X]) #bias
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

start_time = time.time()

weights_reg = logistic_regression_train(X_train, y_train_encoded, X_val, y_val_encoded,
                                       lr=0.1, epochs=1000, batch_size=64, reg_lambda=0.1, reg_type='l2')

end_time = time.time()
print(f"\nCzas treningu z regularyzacją: {end_time - start_time:.2f} sekund")

evaluate_model(X_train, y_train_encoded, weights_reg, encoder, name="TRAIN")
evaluate_model(X_val, y_val_encoded, weights_reg, encoder, name="VAL")
evaluate_model(X_test, y_test_encoded, weights_reg, encoder, name="TEST")


