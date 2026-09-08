import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# 1. File paths
# ==============================

DATA_PATH = "data/Electric_Vehicle_Population_Data_Preprocessed.csv"
MODEL_PATH = "models/ev_range_model.pkl"


# ==============================
# 2. Load dataset
# ==============================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==============================
# 3. Remove rows with missing target
# ==============================

df = df.dropna(subset=["Electric Range"])


# ==============================
# 4. Select useful features
# ==============================

features = [
    "Model Year",
    "Make",
    "Model",
    "Electric Vehicle Type",
    "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
    "Base MSRP"
]

target = "Electric Range"

X = df[features]
y = df[target]


# ==============================
# 5. Separate numerical/categorical
# ==============================

numerical_features = [
    "Model Year",
    "Base MSRP"
]

categorical_features = [
    "Make",
    "Model",
    "Electric Vehicle Type",
    "Clean Alternative Fuel Vehicle (CAFV) Eligibility"
]


# ==============================
# 6. Preprocessing
# ==============================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=5
    ))
])


preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# ==============================
# 7. ML Model
# ==============================

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# ==============================
# 8. Create Pipeline
# ==============================

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ==============================
# 9. Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==============================
# 10. Train model
# ==============================

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")


# ==============================
# 11. Prediction
# ==============================

y_pred = pipeline.predict(X_test)


# ==============================
# 12. Evaluation
# ==============================

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"MAE  : {mae:.2f} km")
print(f"RMSE : {rmse:.2f} km")
print(f"R2   : {r2:.4f}")


# ==============================
# 13. Save model
# ==============================

os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)