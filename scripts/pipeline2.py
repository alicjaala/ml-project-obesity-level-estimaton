import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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

# --- 2. Podział na zbiór treningowy i testowy (80/20) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

# --- 4. Funkcja do trenowania i ewaluacji ---
def train_and_evaluate(model, model_name):
    print(f"\n--------------- {model_name.upper()} -----------------")
    model.fit(X_train, y_train)

    for name, X_data, y_data in [('TRAIN', X_train, y_train), 
                                 ('TEST', X_test, y_test)]:
        y_pred = model.predict(X_data)
        acc = accuracy_score(y_data, y_pred)
        report = classification_report(y_data, y_pred)
        print(f"\n[{name}] Accuracy: {acc:.4f}")
        print(f"[{name}] Classification Report:\n{report}")

# --- 5. Modele w pipeline ---
pipe_tree = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


pipe_svc = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# --- 6. Trening i ewaluacja ---
train_and_evaluate(pipe_tree, "Random Forest")
train_and_evaluate(pipe_svc, "SVC")
