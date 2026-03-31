"""
v8 Experiment: OOF Survival Encoding + LGBM Low-LR + Multi-Seed Ensemble
Run: python run_v8.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("KAGGLE_API_TOKEN", "KGAT_ab12eec1dd2ed1e13c07d7d4e060fb24")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import xgboost as xgb
import lightgbm as lgb

from src.data_loader import load_raw
from src.features import build_features, add_oof_survival_encoding
from src.models import get_base_models, train_stacking_ensemble
from src.inference import generate_submission
from src.config import SUBMISSIONS, RANDOM_SEED, CV_FOLDS, XGB_PARAMS, LGBM_PARAMS

SEP = "=" * 60

# ── Load & Build base features ───────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 1: Load + Feature Engineering (v4 base)\n{SEP}")
train, test = load_raw()
X_train, y_train, X_test, test_ids, preprocessor, feature_names, train_eng, test_eng = \
    build_features(train, test)
print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


# ── OOF Survival Encoding ────────────────────────────────────────────────────
print(f"\n{SEP}\nSTEP 2: OOF Survival Encoding (Surname + Ticket)\n{SEP}")

surname_oof, surname_test = add_oof_survival_encoding(
    train_eng, y_train, test_eng, "Surname", skf
)
ticket_oof, ticket_test = add_oof_survival_encoding(
    train_eng, y_train, test_eng, "Ticket", skf
)

X_train_v8 = np.hstack([X_train, surname_oof.reshape(-1, 1), ticket_oof.reshape(-1, 1)])
X_test_v8  = np.hstack([X_test,  surname_test.reshape(-1, 1), ticket_test.reshape(-1, 1)])
feature_names_v8 = feature_names + ["Surname_OOF_Survival", "Ticket_OOF_Survival"]
print(f"  X_train_v8: {X_train_v8.shape} (+2 OOF features)")

# Quick sanity: LightGBM CV on v8 features
lgbm_v8 = lgb.LGBMClassifier(**LGBM_PARAMS)
scores_v8 = cross_val_score(lgbm_v8, X_train_v8, y_train, cv=skf, scoring="accuracy")
print(f"  LGBM v8 CV: {scores_v8.mean():.4f} ± {scores_v8.std():.4f}")

xgb_v8 = xgb.XGBClassifier(**XGB_PARAMS)
scores_xgb_v8 = cross_val_score(xgb_v8, X_train_v8, y_train, cv=skf, scoring="accuracy")
print(f"  XGB  v8 CV: {scores_xgb_v8.mean():.4f} ± {scores_xgb_v8.std():.4f}")


# ── Experiment A: Stacking on v8 features ────────────────────────────────────
print(f"\n{SEP}\nEXP A: Stacking Ensemble on v8 (OOF-encoded) features\n{SEP}")
models_v8 = get_base_models()
stacking_preds_v8, meta_v8, _ = train_stacking_ensemble(
    models_v8, X_train_v8, y_train, X_test_v8
)
generate_submission(stacking_preds_v8, test_ids, "submission_v8_oof_stacking.csv")


# ── Experiment B: LGBM Low-LR Solo ───────────────────────────────────────────
print(f"\n{SEP}\nEXP B: LGBM Low-LR Solo (lr=0.010, leaves=33, n=569)\n{SEP}")
LGBM_LOWLR = dict(
    n_estimators=569, num_leaves=33, learning_rate=0.010,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
)
lgbm_lowlr = lgb.LGBMClassifier(**LGBM_LOWLR)
scores_lowlr = cross_val_score(lgbm_lowlr, X_train_v8, y_train, cv=skf, scoring="accuracy")
print(f"  LGBM Low-LR (v8 feats) CV: {scores_lowlr.mean():.4f} ± {scores_lowlr.std():.4f}")

# Also try on base v4 features
scores_lowlr_v4 = cross_val_score(
    lgb.LGBMClassifier(**LGBM_LOWLR), X_train, y_train, cv=skf, scoring="accuracy"
)
print(f"  LGBM Low-LR (v4 feats) CV: {scores_lowlr_v4.mean():.4f} ± {scores_lowlr_v4.std():.4f}")

# Submit with better features
best_X = X_train_v8 if scores_lowlr.mean() >= scores_lowlr_v4.mean() else X_train
best_Xtest = X_test_v8 if scores_lowlr.mean() >= scores_lowlr_v4.mean() else X_test
lgbm_lowlr.fit(best_X, y_train)
preds_lowlr = lgbm_lowlr.predict(best_Xtest)
generate_submission(preds_lowlr, test_ids, "submission_v8_lgbm_lowlr.csv")


# ── Experiment C: Multi-Seed Ensemble ────────────────────────────────────────
print(f"\n{SEP}\nEXP C: Multi-Seed Ensemble (5 seeds × 2 models on v8 feats)\n{SEP}")
SEEDS = [42, 7, 21, 99, 123]
all_proba = []

for seed in SEEDS:
    xp = dict(**XGB_PARAMS)
    xp["random_state"] = seed
    m_xgb = xgb.XGBClassifier(**xp)
    m_xgb.fit(X_train_v8, y_train)
    all_proba.append(m_xgb.predict_proba(X_test_v8)[:, 1])

    lp = dict(**LGBM_PARAMS)
    lp["random_state"] = seed
    m_lgbm = lgb.LGBMClassifier(**lp)
    m_lgbm.fit(X_train_v8, y_train)
    all_proba.append(m_lgbm.predict_proba(X_test_v8)[:, 1])

mean_proba = np.mean(all_proba, axis=0)
preds_multiseed = (mean_proba >= 0.5).astype(int)
generate_submission(preds_multiseed, test_ids, "submission_v8_multiseed.csv")
print(f"  Ensembled {len(all_proba)} models (5 seeds × XGB+LGBM)")

# CV estimate for multi-seed (using OOF)
multiseed_oof_proba = np.zeros((len(y_train), len(SEEDS) * 2))
col = 0
for seed in SEEDS:
    xp = dict(**XGB_PARAMS); xp["random_state"] = seed
    oof_x = cross_val_predict(xgb.XGBClassifier(**xp), X_train_v8, y_train,
                               cv=skf, method="predict_proba")[:, 1]
    multiseed_oof_proba[:, col] = oof_x; col += 1

    lp = dict(**LGBM_PARAMS); lp["random_state"] = seed
    oof_l = cross_val_predict(lgb.LGBMClassifier(**lp), X_train_v8, y_train,
                               cv=skf, method="predict_proba")[:, 1]
    multiseed_oof_proba[:, col] = oof_l; col += 1

oof_mean = multiseed_oof_proba.mean(axis=1)
oof_acc = ((oof_mean >= 0.5).astype(int) == y_train).mean()
print(f"  Multi-seed OOF accuracy: {oof_acc:.4f}")


# ── Experiment D: Combined (Stacking + Low-LR + Multi-seed) ─────────────────
print(f"\n{SEP}\nEXP D: Combined Ensemble (Stacking + Low-LR + Multi-seed)\n{SEP}")
# Get stacking probabilities
oof_preds_stack = np.zeros((len(y_train), len(models_v8)))
for j, (name, model) in enumerate(list(models_v8.items())):
    oof_preds_stack[:, j] = cross_val_predict(
        model, X_train_v8, y_train, cv=skf, method="predict_proba", n_jobs=-1
    )[:, 1]
    model.fit(X_train_v8, y_train)
test_stack_proba = meta_v8.predict_proba(
    np.column_stack([m.predict_proba(X_test_v8)[:, 1] for m in models_v8.values()])
)[:, 1]

# Combine: 40% stacking + 30% low-LR + 30% multi-seed
combined_proba = (
    0.40 * test_stack_proba +
    0.30 * lgbm_lowlr.predict_proba(best_Xtest)[:, 1] +
    0.30 * mean_proba
)
preds_combined = (combined_proba >= 0.5).astype(int)
generate_submission(preds_combined, test_ids, "submission_v8_combined.csv")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{SEP}\nSUMMARY\n{SEP}")
print(f"  LGBM v4 base CV:           {cross_val_score(lgb.LGBMClassifier(**LGBM_PARAMS), X_train, y_train, cv=skf, scoring='accuracy').mean():.4f}")
print(f"  LGBM v8 OOF feats CV:      {scores_v8.mean():.4f} ± {scores_v8.std():.4f}")
print(f"  XGB  v8 OOF feats CV:      {scores_xgb_v8.mean():.4f} ± {scores_xgb_v8.std():.4f}")
print(f"  LGBM Low-LR (best) CV:     {max(scores_lowlr.mean(), scores_lowlr_v4.mean()):.4f}")
print(f"  Multi-seed OOF accuracy:   {oof_acc:.4f}")
print()
print("  Submissions generated:")
print("    submission_v8_oof_stacking.csv  — Stacking on OOF-encoded features")
print("    submission_v8_lgbm_lowlr.csv    — LGBM low-LR (lr=0.010, n=569)")
print("    submission_v8_multiseed.csv     — 10-model multi-seed ensemble")
print("    submission_v8_combined.csv      — Weighted blend of all 3")
print(f"\n  Previous best: 0.77751 (v7 OOF stacking)")
print(f"  Submit these 4 files to Kaggle to compare!")
