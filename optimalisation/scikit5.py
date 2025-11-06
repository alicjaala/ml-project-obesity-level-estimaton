import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/obesity_data.csv')

selected_cols = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC',
    'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad'
]
data = data[selected_cols]

X = data.drop("NObeyesdad", axis=1)
y = data["NObeyesdad"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

pipe_rf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipe_svc = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

param_grid_rf = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

param_grid_svc = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto']
}

print("\nRandom Forest: hiperparameters i accuracy:\n")
grid_rf = RandomizedSearchCV(pipe_rf, param_grid_rf, cv=3, scoring='accuracy', verbose=0)
grid_rf.fit(X_train, y_train)

rf_results = pd.DataFrame(grid_rf.cv_results_)
for idx, row in rf_results.iterrows():
    print(f"Params: {row['params']}")
    print(f"Mean Accuracy: {row['mean_test_score']:.4f}")
    print("---------------------------------------------------------")

print("\nSVC: hiperparameters i accuracy:\n")
grid_svc = RandomizedSearchCV(pipe_svc, param_grid_svc, cv=5, scoring='accuracy', verbose=0)
grid_svc.fit(X_train, y_train)

svc_results = pd.DataFrame(grid_svc.cv_results_)
for idx, row in svc_results.iterrows():
    print(f"Params: {row['params']}")
    print(f"Mean Accuracy: {row['mean_test_score']:.4f}")
    print("----------------------------------------------------------------")

