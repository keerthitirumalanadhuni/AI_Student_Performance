import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ============================================================
# 1. LOAD CLEAN DATASET
# ============================================================

df = pd.read_csv("dataset/student_performance_clean.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["final_score", "student_id"])

y = df["final_score"]


# ============================================================
# 3. IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = [
    "gender",
    "department"
]

numerical_columns = [
    column
    for column in X.columns
    if column not in categorical_columns
]


# ============================================================
# 4. PREPROCESSING
# ============================================================

numerical_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_preprocessor,
            numerical_columns
        ),
        (
            "categorical",
            categorical_preprocessor,
            categorical_columns
        )
    ]
)


# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ============================================================
# 7. TRAIN AND COMPARE MODELS
# ============================================================

results = []

trained_models = {}


for model_name, model in models.items():

    print("\nTraining:", model_name)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append(
        {
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        }
    )

    trained_models[model_name] = pipeline


# ============================================================
# 8. DISPLAY MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n========================================")
print("          MODEL COMPARISON")
print("========================================")

print(
    results_df.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "R2": "{:.4f}".format
        }
    )
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

# Higher R2 is better.
# If R2 is similar, lower RMSE is preferred.

best_model_name = (
    results_df
    .sort_values(
        by=["R2", "RMSE"],
        ascending=[False, True]
    )
    .iloc[0]["Model"]
)


best_model = trained_models[
    best_model_name
]


# ============================================================
# 10. DISPLAY BEST MODEL
# ============================================================

best_row = results_df[
    results_df["Model"] == best_model_name
].iloc[0]


print("\n========================================")
print("           BEST MODEL")
print("========================================")

print(
    "Selected Model:",
    best_model_name
)

print(
    "MAE:",
    round(best_row["MAE"], 2)
)

print(
    "RMSE:",
    round(best_row["RMSE"], 2)
)

print(
    "R²:",
    round(best_row["R2"], 4)
)


# ============================================================
# 11. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/student_performance_model.pkl"
)

print("\nBest model saved successfully.")

print(
    "Output file:",
    "models/student_performance_model.pkl"
)
