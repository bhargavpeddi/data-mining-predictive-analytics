"""Dataset loaders and per-dataset feature engineering.

Every loader returns a ``Task`` describing the target, the problem type and
which columns are numeric / categorical so the modelling code stays generic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@dataclass
class Task:
    name: str
    kind: str  # "classification" | "regression"
    X: pd.DataFrame
    y: pd.Series
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    description: str = ""


def titanic() -> Task:
    df = pd.read_csv(RAW / "titanic.csv")
    # Feature engineering
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["is_alone"] = (df["family_size"] == 1).astype(int)
    df["fare_per_person"] = df["fare"] / df["family_size"]
    df["age_missing"] = df["age"].isna().astype(int)
    df["child"] = (df["age"] < 16).astype(float).where(df["age"].notna())
    numeric = ["age", "fare", "sibsp", "parch", "family_size", "is_alone",
               "fare_per_person", "age_missing", "child"]
    categorical = ["pclass", "sex", "embarked"]
    df["pclass"] = df["pclass"].astype(str)
    return Task("Titanic survival", "classification", df[numeric + categorical],
                df["survived"], numeric, categorical,
                "891 passengers; predict whether a passenger survived.")


def breast_cancer() -> Task:
    data = load_breast_cancer(as_frame=True)
    X = data.data
    # 0 = malignant, 1 = benign in sklearn; flip so 1 = malignant (positive class)
    y = (1 - data.target).rename("malignant")
    return Task("Breast cancer diagnosis", "classification", X, y, list(X.columns), [],
                "569 tumour biopsies (Wisconsin); predict malignant vs benign.")


def penguins() -> Task:
    df = pd.read_csv(RAW / "penguins.csv").dropna(subset=["species"])
    df["bill_ratio"] = df["bill_length_mm"] / df["bill_depth_mm"]
    df["mass_per_flipper"] = df["body_mass_g"] / df["flipper_length_mm"]
    numeric = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
               "bill_ratio", "mass_per_flipper"]
    categorical = ["island", "sex"]
    return Task("Penguin species", "classification", df[numeric + categorical],
                df["species"], numeric, categorical,
                "344 Palmer penguins; multi-class species prediction.")


def diamonds(sample: int | None = 20000, seed: int = 42) -> Task:
    df = pd.read_csv(RAW / "diamonds.csv")
    df = df[(df[["x", "y", "z"]] > 0).all(axis=1)]  # drop impossible zero dimensions
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=seed)
    df["volume"] = df["x"] * df["y"] * df["z"]
    df["log_carat"] = np.log(df["carat"])
    numeric = ["carat", "log_carat", "depth", "table", "x", "y", "z", "volume"]
    categorical = ["cut", "color", "clarity"]
    return Task("Diamond price", "regression", df[numeric + categorical],
                np.log(df["price"]).rename("log_price"), numeric, categorical,
                "53,940 diamonds (20k sampled); regress log price.")


ALL = {"titanic": titanic, "breast_cancer": breast_cancer,
       "penguins": penguins, "diamonds": diamonds}
