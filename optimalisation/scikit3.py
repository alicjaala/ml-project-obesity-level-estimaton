import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

# --- 4. Modele do porównania ---
models = {
    "Regresja logistyczna BEZ regularyzacji": LogisticRegression(
        penalty=None, solver='saga', max_iter=5000, random_state=42
    ),
    "Regresja logistyczna z Ridge (L2)": LogisticRegression(
        penalty='l2', C=0.05, solver='saga', max_iter=5000, random_state=42
    ),
    "Regresja logistyczna z Lasso (L1)": LogisticRegression(
        penalty='l1', C=0.05, solver='saga', max_iter=5000, random_state=42
    ),
}

# --- 5. Trening, ewaluacja i analiza wag ---
for name, clf in models.items():
    print(f"\n=============== {name.upper()} ===============")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    
    # Ewaluacja
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))
    
    # Wagi modelu - po przetworzeniu cech
    clf_fitted = pipeline.named_steps['classifier']
    preproc = pipeline.named_steps['preprocessor']
    
    #Liczba cech po transformacji
    feature_names_num = numerical_cols      #lista nazw cech
    onehot = preproc.named_transformers_['cat'].named_steps['onehot']   #zakodowane nazwy
    cat_features = onehot.get_feature_names_out(categorical_cols)      #łączy słowne i zakodowane nazwy w odpowiedniej kolejności
    
    feature_names = np.concatenate([feature_names_num, cat_features])
    
    #waga - liczba określająca wpływ cechy wejściowej na przewidywaną klasę
    #dodatnia waga zwiększa prawdopodobieństwo, ujemna zmniejsza, im większa wartość bezwzględna tym bardziej wpływowa
    #shape = (n_classes, n_features))
    coefs = clf_fitted.coef_    #pobiera macierz wag modelu
    
    #wypisuje współczynniki pierwszej klasy lub średnią bezwzględną wartość wag z każdej klasy, aby określić wpływ cechy
    if coefs.shape[0] == 1:
        coefs_to_show = coefs[0]
    else:
        # średnia bezwzględna po klasach
        coefs_to_show = np.mean(np.abs(coefs), axis=0)
    
    print("\nPrzykładowe wagi cech (absolutne wartości):")
    for fname, coef in zip(feature_names, coefs_to_show):
        print(f"{fname}: {coef:.4f}")

