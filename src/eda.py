"""Exploratory data analysis figures for each dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datasets import ALL

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
sns.set_theme(style="whitegrid", palette="deep")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)


def titanic_eda(task):
    df = task.X.assign(survived=task.y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.barplot(data=df, x="sex", y="survived", hue="pclass", hue_order=["1", "2", "3"], ax=axes[0], errorbar=None)
    axes[0].set_title("Survival rate by sex and class")
    sns.histplot(data=df, x="age", hue="survived", bins=30, multiple="stack", ax=axes[1])
    axes[1].set_title("Age distribution by outcome")
    sns.barplot(data=df, x="family_size", y="survived", ax=axes[2], errorbar=None, color="C2")
    axes[2].set_title("Survival rate by family size")
    _save(fig, "eda_titanic.png")


def breast_cancer_eda(task):
    df = task.X.copy()
    top = df.corrwith(task.y).abs().sort_values(ascending=False).head(10).index
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sns.heatmap(df[top].corr(), cmap="vlag", center=0, ax=axes[0], annot=False)
    axes[0].set_title("Correlation of the 10 most predictive features")
    sns.kdeplot(data=df.assign(diagnosis=task.y.map({1: "malignant", 0: "benign"})),
                x="worst concave points", hue="diagnosis", fill=True, ax=axes[1])
    axes[1].set_title("Worst concave points separates the classes")
    _save(fig, "eda_breast_cancer.png")


def penguins_eda(task):
    df = task.X.assign(species=task.y)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.scatterplot(data=df, x="bill_length_mm", y="bill_depth_mm", hue="species", ax=axes[0])
    axes[0].set_title("Bill length vs depth")
    sns.boxplot(data=df, x="species", y="flipper_length_mm", ax=axes[1])
    axes[1].set_title("Flipper length by species")
    _save(fig, "eda_penguins.png")


def diamonds_eda(task):
    df = task.X.assign(price=np.exp(task.y))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    sns.scatterplot(data=df.sample(4000, random_state=0), x="carat", y="price",
                    hue="clarity", s=10, ax=axes[0], legend=False)
    axes[0].set_title("Price vs carat (coloured by clarity)")
    axes[1].hist(df["price"], bins=60, color="C0")
    axes[1].set_title("Price is right-skewed → model log(price)")
    order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
    sns.boxplot(data=df, x="cut", y="price", order=order, ax=axes[2])
    axes[2].set_yscale("log")
    axes[2].set_title("Price by cut (log scale)")
    _save(fig, "eda_diamonds.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    funcs = {"titanic": titanic_eda, "breast_cancer": breast_cancer_eda,
             "penguins": penguins_eda, "diamonds": diamonds_eda}
    for key, loader in ALL.items():
        task = loader()
        print(f"[EDA] {task.name}: {task.X.shape[0]} rows, {task.X.shape[1]} features, "
              f"missing cells = {int(task.X.isna().sum().sum())}")
        funcs[key](task)


if __name__ == "__main__":
    main()
