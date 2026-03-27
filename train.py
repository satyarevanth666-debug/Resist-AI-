from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from preprocess import TARGET_MAPPING, load_and_prepare_data

try:
    from xgboost import XGBClassifier

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _evaluate_model(
    model, x_test: pd.DataFrame, y_test: pd.Series, num_classes: int
) -> Dict[str, float]:
    preds = model.predict(x_test)
    proba = model.predict_proba(x_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_weighted": float(
            precision_score(y_test, preds, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, preds, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
    }

    if proba is not None and num_classes > 2:
        y_bin = label_binarize(y_test, classes=list(range(num_classes)))
        metrics["roc_auc_ovr"] = float(
            roc_auc_score(y_bin, proba, multi_class="ovr", average="weighted")
        )
    elif proba is not None:
        metrics["roc_auc_ovr"] = float(roc_auc_score(y_test, proba[:, 1]))
    else:
        metrics["roc_auc_ovr"] = 0.0

    return metrics


def train_and_save_model() -> Tuple[dict, pd.DataFrame]:
    merged_df, x, encoders = load_and_prepare_data()
    y = merged_df["ResultCode"]

    print("\nClass distribution:")
    print(merged_df["Result"].value_counts())
    print("\nTop bacteria trends:")
    print(merged_df["Bacteria"].value_counts().head(10))
    print("\nTop antibiotic trends:")
    print(merged_df["Antibiotic"].value_counts().head(10))

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    num_classes = len(np.unique(y))

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1200, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            random_state=42,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
        ),
    }

    if XGB_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            random_state=42,
            n_estimators=350,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss",
        )

    results: Dict[str, dict] = {}
    trained_models: Dict[str, object] = {}
    confusion_matrices: Dict[str, list] = {}

    for model_name, model in models.items():
        if model_name == "xgboost":
            model.fit(x_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(x_train, y_train)
        trained_models[model_name] = model

        model_metrics = _evaluate_model(model, x_test, y_test, num_classes)
        y_pred = model.predict(x_test)
        confusion_matrices[model_name] = confusion_matrix(y_test, y_pred).tolist()

        print(f"\n=== {model_name} ===")
        print(model_metrics)
        print(classification_report(y_test, y_pred, zero_division=0))
        results[model_name] = model_metrics

    best_name = max(
        results.keys(),
        key=lambda k: (results[k]["f1_weighted"], results[k]["roc_auc_ovr"]),
    )
    best_model = trained_models[best_name]

    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = {
            "BacteriaEncoded": float(best_model.feature_importances_[0]),
            "AntibioticEncoded": float(best_model.feature_importances_[1]),
        }
    elif hasattr(best_model, "coef_"):
        avg_coef = np.abs(best_model.coef_).mean(axis=0)
        feature_importance = {
            "BacteriaEncoded": float(avg_coef[0]),
            "AntibioticEncoded": float(avg_coef[1]),
        }

    artifact = {
        "best_model_name": best_name,
        "best_model": best_model,
        "all_metrics": results,
        "target_mapping": TARGET_MAPPING,
        "inverse_target_mapping": {v: k for k, v in TARGET_MAPPING.items()},
        "encoders": encoders,
        "feature_columns": ["BacteriaEncoded", "AntibioticEncoded"],
        "feature_importance": feature_importance,
        "confusion_matrices": confusion_matrices,
    }

    with open(MODELS_DIR / "model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    merged_df.to_csv(MODELS_DIR / "cleaned_data.csv", index=False)
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nBest model: {best_name}")
    print(f"Saved model artifact to: {MODELS_DIR / 'model.pkl'}")
    return artifact, merged_df


if __name__ == "__main__":
    train_and_save_model()
