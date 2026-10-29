"""Preprocessing pipelines and model zoo."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42


def preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    parts = [("num", num, numeric)]
    if categorical:
        parts.append(("cat", cat, categorical))
    return ColumnTransformer(parts)


def candidates(kind: str) -> dict:
    """Models plus small hyper-parameter grids tuned with cross-validation."""
    if kind == "classification":
        return {
            "Logistic Regression": (LogisticRegression(max_iter=5000),
                                    {"model__C": [0.01, 0.1, 1, 10]}),
            "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                              {"model__max_depth": [3, 5, 8, None],
                               "model__min_samples_leaf": [1, 5, 10]}),
            "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
                              {"model__max_depth": [None, 8],
                               "model__min_samples_leaf": [1, 3]}),
        }
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED),
                          {"model__max_depth": [6, 10, 15],
                           "model__min_samples_leaf": [1, 5, 10]}),
        "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
                          {"model__max_depth": [None, 15],
                           "model__min_samples_leaf": [1, 3]}),
    }


def build(model, numeric, categorical) -> Pipeline:
    return Pipeline([("prep", preprocessor(numeric, categorical)), ("model", model)])
