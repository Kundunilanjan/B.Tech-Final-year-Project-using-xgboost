import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("cleaned_raw_dataset.csv")

# =========================
# ENCODE CATEGORICAL COLUMNS
# =========================

label_encoders = {}

categorical_cols = [
    'Soil_Type',
    'Retaining_Wall_Type',
    'Support_System'
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =========================
# FEATURES & TARGET
# =========================

X = df.drop('Risk_Level', axis=1)
y = df['Risk_Level']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# XGBOOST MODEL
# =========================

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    random_state=42
)

# =========================
# HYPERPARAMETER TUNING
# =========================

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

# =========================
# TRAIN MODEL
# =========================

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# =========================
# PREDICTIONS
# =========================

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(best_model, "xgboost_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("\nModel Saved Successfully")