"""
Model training: individual models + stacking ensemble.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
import xgboost as xgb
import lightgbm as lgb
import joblib

from src.config import RANDOM_SEED, CV_FOLDS, RF_PARAMS, XGB_PARAMS, LGBM_PARAMS, MODELS_DIR


def get_base_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED
        ),
        "Random Forest": RandomForestClassifier(**RF_PARAMS),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_SEED
        ),
        "XGBoost": xgb.XGBClassifier(**XGB_PARAMS),
        "LightGBM": lgb.LGBMClassifier(**LGBM_PARAMS),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_SEED),
    }


def cross_validate_all(models: dict, X, y) -> pd.DataFrame:
    """Run 5-fold CV on all models, return results DataFrame."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    records = []
    for name, model in models.items():
        print(f"  CV: {name}...", end=" ", flush=True)
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        records.append(dict(
            Model=name,
            CV_Mean=scores.mean(),
            CV_Std=scores.std(),
            CV_Min=scores.min(),
            CV_Max=scores.max(),
        ))
        print(f"{scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(records).sort_values("CV_Mean", ascending=False).reset_index(drop=True)


def train_stacking_ensemble(models: dict, X_train, y_train, X_test):
    """
    Build a stacking ensemble:
      Level 0: base models (OOF predictions on train, regular predictions on test)
      Level 1: LogisticRegression meta-learner
    Returns (final_train_meta, final_test_preds, meta_model).
    """
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    n_models = len(models)

    oof_preds = np.zeros((n_train, n_models))
    test_preds = np.zeros((n_test, n_models))

    model_list = list(models.items())
    print("Building stacking ensemble...")
    for j, (name, model) in enumerate(model_list):
        print(f"  OOF: {name}...")
        oof_preds[:, j] = cross_val_predict(
            model, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1
        )[:, 1]
        model.fit(X_train, y_train)
        test_preds[:, j] = model.predict_proba(X_test)[:, 1]

    # Meta-learner
    meta_model = LogisticRegression(C=1.0, max_iter=500, random_state=RANDOM_SEED)
    meta_model.fit(oof_preds, y_train)
    final_preds = (meta_model.predict_proba(test_preds)[:, 1] >= 0.5).astype(int)

    meta_scores = cross_val_score(meta_model, oof_preds, y_train,
                                  cv=skf, scoring="accuracy")
    print(f"  Stacking meta-CV: {meta_scores.mean():.4f} ± {meta_scores.std():.4f}")

    return final_preds, meta_model, model_list


def train_voting_ensemble(models: dict, X_train, y_train, X_test):
    """Hard voting ensemble of all models."""
    estimators = [(name, model) for name, model in models.items()]
    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    voting.fit(X_train, y_train)
    return voting.predict(X_test), voting


def tune_with_optuna(X, y, n_trials: int = 60) -> dict:
    """
    Optuna hyperparameter search for XGBoost and LightGBM.
    Returns {'xgb': best_xgb_params, 'lgbm': best_lgbm_params}.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    def xgb_objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 2, 5),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.5, 3.0),
            eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1,
        )
        scores = cross_val_score(xgb.XGBClassifier(**params), X, y,
                                 cv=skf, scoring="accuracy")
        return scores.mean()

    def lgbm_objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800),
            num_leaves=trial.suggest_int("num_leaves", 8, 40),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            feature_fraction=trial.suggest_float("feature_fraction", 0.6, 1.0),
            bagging_fraction=trial.suggest_float("bagging_fraction", 0.6, 1.0),
            bagging_freq=5,
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )
        scores = cross_val_score(lgb.LGBMClassifier(**params), X, y,
                                 cv=skf, scoring="accuracy")
        return scores.mean()

    print("  Tuning XGBoost...")
    xgb_study = optuna.create_study(direction="maximize")
    xgb_study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  XGBoost best CV: {xgb_study.best_value:.4f} | params: {xgb_study.best_params}")

    print("  Tuning LightGBM...")
    lgbm_study = optuna.create_study(direction="maximize")
    lgbm_study.optimize(lgbm_objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  LightGBM best CV: {lgbm_study.best_value:.4f} | params: {lgbm_study.best_params}")

    return {"xgb": xgb_study.best_params, "lgbm": lgbm_study.best_params}


def train_and_save_all(models: dict, X_train, y_train):
    """Fit all models on full training data and save to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fitted = {}
    for name, model in models.items():
        print(f"  Fitting: {name}...")
        model.fit(X_train, y_train)
        safe_name = name.lower().replace(" ", "_")
        path = MODELS_DIR / f"{safe_name}.pkl"
        joblib.dump(model, path)
        fitted[name] = model
        print(f"    Saved → {path}")
    return fitted
