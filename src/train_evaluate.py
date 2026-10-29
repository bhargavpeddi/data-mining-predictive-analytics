"""Train, tune and evaluate every model on every dataset.

Outputs
-------
results/metrics.csv              hold-out metrics for every (dataset, model)
reports/figures/model_comparison.png
reports/figures/<dataset>_*.png  confusion matrices / feature importance / residuals
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     train_test_split)

from datasets import ALL
from models import SEED, build, candidates

ROOT = Path(__file__).resolve().parents[1]
FIG, RES = ROOT / "reports" / "figures", ROOT / "results"
sns.set_theme(style="whitegrid", palette="deep")


def evaluate_classifier(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)
    if proba.shape[1] == 2:
        auc = roc_auc_score(y_te, proba[:, 1])
    else:
        auc = roc_auc_score(y_te, proba, multi_class="ovr")
    return {"accuracy": accuracy_score(y_te, pred),
            "f1_macro": f1_score(y_te, pred, average="macro"),
            "roc_auc": auc}


def evaluate_regressor(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    price, price_hat = np.exp(y_te), np.exp(pred)
    return {"r2": r2_score(y_te, pred),
            "rmse_log": float(np.sqrt(mean_squared_error(y_te, pred))),
            "mae_usd": mean_absolute_error(price, price_hat),
            "mape_pct": float(np.mean(np.abs(price - price_hat) / price) * 100)}


def feature_importance_plot(pipe, key, title):
    model = pipe.named_steps["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    imp = pd.Series(model.feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp.plot.barh(ax=ax, color="C0")
    ax.set_title(f"{title}: Random Forest feature importance")
    fig.tight_layout()
    fig.savefig(FIG / f"{key}_feature_importance.png", dpi=130)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    rows, best_params = [], {}

    for key, loader in ALL.items():
        task = loader()
        strat = task.y if task.kind == "classification" else None
        X_tr, X_te, y_tr, y_te = train_test_split(task.X, task.y, test_size=0.2,
                                                  random_state=SEED, stratify=strat)
        cv = (StratifiedKFold(5, shuffle=True, random_state=SEED)
              if task.kind == "classification" else KFold(5, shuffle=True, random_state=SEED))
        scoring = "f1_macro" if task.kind == "classification" else "r2"
        fitted = {}

        for name, (model, grid) in candidates(task.kind).items():
            search = GridSearchCV(build(model, task.numeric, task.categorical),
                                  grid or {"model__fit_intercept": [True]},
                                  cv=cv, scoring=scoring, n_jobs=-1)
            search.fit(X_tr, y_tr)
            pipe = search.best_estimator_
            fitted[name] = pipe
            metrics = (evaluate_classifier if task.kind == "classification"
                       else evaluate_regressor)(pipe, X_te, y_te)
            rows.append({"dataset": task.name, "task": task.kind, "model": name,
                         f"cv_{scoring}": search.best_score_, **metrics})
            best_params[f"{key}/{name}"] = {k.replace("model__", ""): v
                                            for k, v in search.best_params_.items()}
            print(f"{task.name:<26}{name:<20}" +
                  "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

        feature_importance_plot(fitted["Random Forest"], key, task.name)

        if task.kind == "classification":
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
            for ax, (name, pipe) in zip(axes, fitted.items()):
                ConfusionMatrixDisplay.from_estimator(pipe, X_te, y_te, ax=ax,
                                                      colorbar=False, cmap="Blues")
                ax.set_title(name)
                ax.grid(False)
            fig.suptitle(f"{task.name}: hold-out confusion matrices")
            fig.tight_layout()
            fig.savefig(FIG / f"{key}_confusion_matrices.png", dpi=130)
            plt.close(fig)
        else:
            pred = fitted["Random Forest"].predict(X_te)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].scatter(np.exp(y_te), np.exp(pred), s=6, alpha=0.4)
            lim = [0, np.exp(y_te).max()]
            axes[0].plot(lim, lim, "r--", lw=1)
            axes[0].set(xlabel="Actual price ($)", ylabel="Predicted price ($)",
                        title="Random Forest: predicted vs actual")
            sns.histplot(y_te - pred, bins=60, ax=axes[1])
            axes[1].set(title="Residuals (log price)", xlabel="actual − predicted")
            fig.tight_layout()
            fig.savefig(FIG / f"{key}_residuals.png", dpi=130)
            plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "metrics.csv", index=False, float_format="%.4f")
    (RES / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

    # Model comparison chart (the "non-technical stakeholder" view)
    clf = df[df.task == "classification"]
    reg = df[df.task == "regression"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [3, 1.2]})
    sns.barplot(data=clf, x="dataset", y="accuracy", hue="model", ax=axes[0])
    axes[0].set(ylim=(0.6, 1.0), title="Classification: hold-out accuracy", xlabel="")
    for c in axes[0].containers:
        axes[0].bar_label(c, fmt="%.2f", fontsize=8)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    sns.barplot(data=reg, x="model", y="r2", ax=axes[1], hue="model", legend=False)
    axes[1].set(ylim=(0.8, 1.0), title="Diamond price: hold-out R²", xlabel="")
    axes[1].tick_params(axis="x", rotation=20)
    for c in axes[1].containers:
        axes[1].bar_label(c, fmt="%.3f", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=130)
    plt.close(fig)
    print("\nSaved results/metrics.csv and figures to reports/figures/")


if __name__ == "__main__":
    main()
