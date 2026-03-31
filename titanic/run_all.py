"""
Full Titanic ML pipeline: EDA → Feature Engineering → Training → Inference.
Usage:  python run_all.py
        ENABLE_OPTUNA=1 python run_all.py   (with hyperparameter tuning ~5 min)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

# Kaggle API token (set via env or hardcoded for convenience)
os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_ab12eec1dd2ed1e13c07d7d4e060fb24")

import numpy as np
import pandas as pd

from src.data_loader import load_raw
from src.features import build_features
from src.models import (get_base_models, cross_validate_all,
                         train_stacking_ensemble, train_and_save_all,
                         tune_with_optuna)
from src.evaluate import (plot_cv_comparison, plot_confusion_matrix, plot_roc_curves,
                          plot_feature_importance, plot_shap_summary, plot_learning_curve)
from src.inference import generate_submission
from src.config import SUBMISSIONS, FIGURES_DIR, MODELS_DIR, EDA_REPORT_DIR
import src.config as config

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ENABLE_OPTUNA = os.environ.get("ENABLE_OPTUNA", "0") == "1"


def save_fig(fig, name: str):
    path = FIGURES_DIR / f"{name}.html"
    fig.write_html(str(path), config={"responsive": True})
    print(f"  Figure saved → {path.name}")


# ── Step 1: EDA HTML ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: EDA → Generating eda_report/index.html")
print("="*60)
from generate_eda_html import main as gen_eda
gen_eda()


# ── Step 2: Feature Engineering ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: Feature Engineering")
print("="*60)
train, test = load_raw()
X_train, y_train, X_test, test_ids, preprocessor, feature_names = build_features(train, test)
print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Features ({len(feature_names)}): {feature_names[:8]}...")


# ── Step 2.5 (Optional): Optuna Hyperparameter Tuning ───────────────────────
if ENABLE_OPTUNA:
    print("\n" + "="*60)
    print("STEP 2.5: Optuna Hyperparameter Tuning (~5 min)")
    print("="*60)
    best_params = tune_with_optuna(X_train, y_train, n_trials=60)

    # Apply tuned XGBoost params
    config.XGB_PARAMS.update(best_params["xgb"])
    config.XGB_PARAMS.update({"eval_metric": "logloss",
                               "random_state": config.RANDOM_SEED, "n_jobs": -1})
    # Apply tuned LightGBM params
    config.LGBM_PARAMS.update(best_params["lgbm"])
    config.LGBM_PARAMS.update({"random_state": config.RANDOM_SEED,
                                "n_jobs": -1, "verbose": -1})
    print("  Config updated with tuned params.")


# ── Step 3: Cross-Validation ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: 5-Fold Cross-Validation")
print("="*60)
models = get_base_models()
cv_results = cross_validate_all(models, X_train, y_train)
print("\nCV Results:")
print(cv_results.to_string(index=False))
cv_fig = plot_cv_comparison(cv_results)
save_fig(cv_fig, "cv_comparison")


# ── Step 4: ROC Curves ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: ROC Curves")
print("="*60)
roc_fig = plot_roc_curves(models, X_train, y_train)
save_fig(roc_fig, "roc_curves")


# ── Step 5: Train all models ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: Training all models on full training data")
print("="*60)
fitted_models = train_and_save_all(models, X_train, y_train)


# ── Step 6: Best model analysis ─────────────────────────────────────────────
best_name = cv_results.iloc[0]["Model"]
best_model = fitted_models[best_name]
print(f"\nBest model: {best_name} (CV={cv_results.iloc[0]['CV_Mean']:.4f})")

print(f"\nGenerating confusion matrix for {best_name}...")
cm_fig = plot_confusion_matrix(best_model, X_train, y_train, best_name)
save_fig(cm_fig, "confusion_matrix")

fi_fig = plot_feature_importance(best_model, feature_names, best_name)
save_fig(fi_fig, "feature_importance")

if best_name in ("XGBoost", "LightGBM", "Random Forest", "Gradient Boosting"):
    print(f"Computing SHAP values for {best_name}...")
    shap_fig = plot_shap_summary(best_model, X_train, feature_names, best_name)
    save_fig(shap_fig, "shap_summary")

print(f"Computing learning curve for {best_name}...")
lc_fig = plot_learning_curve(best_model, X_train, y_train, best_name)
save_fig(lc_fig, "learning_curve")


# ── Step 7: Stacking Ensemble ─────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: Stacking Ensemble")
print("="*60)
stacking_preds, meta_model, model_list = train_stacking_ensemble(
    get_base_models(), X_train, y_train, X_test
)
generate_submission(stacking_preds, test_ids, "submission_stacking_v2.csv")


# ── Step 8: Individual submissions ───────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8: Individual Model Submissions")
print("="*60)
for name, model in fitted_models.items():
    preds = model.predict(X_test)
    safe = name.lower().replace(" ", "_")
    generate_submission(preds, test_ids, f"submission_{safe}_v2.csv")


# ── Step 9: Auto-submit best to Kaggle ───────────────────────────────────────
print("\n" + "="*60)
print("STEP 9: Submitting to Kaggle")
print("="*60)
import subprocess
best_file = SUBMISSIONS / "submission_stacking_v2.csv"
result = subprocess.run([
    "kaggle", "competitions", "submit",
    "-c", "titanic",
    "-f", str(best_file),
    "-m", f"v2: fixed TicketFreq, new features (IsChild/IsMother/FarePerPerson/SurnameFreq), stronger regularization. Best model: {best_name}"
], capture_output=True, text=True)
print(result.stdout or result.stderr)

# Check score
import time
time.sleep(20)
result2 = subprocess.run(
    ["kaggle", "competitions", "submissions", "-c", "titanic"],
    capture_output=True, text=True
)
print(result2.stdout)


# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"\n📊 EDA Report:    {EDA_REPORT_DIR}/index.html")
print(f"📈 Figures:       {FIGURES_DIR}/")
print(f"🤖 Models:        {MODELS_DIR}/")
print(f"📤 Submissions:   {SUBMISSIONS}/")
print(f"\nBest submission: submission_stacking_v2.csv")
