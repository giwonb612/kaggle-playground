"""
v19b: D_minimal 기반 추가 실험
1. +Cabin_Deck 10-seed 검증
2. +AgeGroup 10-seed 검증
3. D_minimal + SVM base model
4. D_minimal × v10 블렌딩 (소프트 보팅)
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_raw
from src.features import build_features
from src.config import RF_PARAMS, XGB_PARAMS, SUBMISSIONS

BASE_5 = {
    'LR':  LogisticRegression(C=1.0, max_iter=1000),
    'RF':  RandomForestClassifier(**RF_PARAMS),
    'GB':  GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1),
    'XGB': xgb.XGBClassifier(**XGB_PARAMS),
    'LGBM_LL': lgb.LGBMClassifier(
        n_estimators=569, num_leaves=31, learning_rate=0.010,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        n_jobs=-1, verbose=-1
    ),
}

BASE_6 = {
    **BASE_5,
    'SVM': SVC(C=1.0, kernel='rbf', probability=True),
}

# v10 full feature set (19개) + SVM
V10_NUM = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketFreq", "LogFare"]
V10_CAT = ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "Has_Cabin",
           "Cabin_Deck", "FamilySizeGroup", "AgeGroup", "Pclass_Sex"]

D_NUM = ["Age", "Fare", "FamilySize"]
D_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]

SEEDS_FULL = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]
BASELINE = 0.8438


def bayesian_smooth(counts, means, global_mean, k):
    return (counts * means + k * global_mean) / (counts + k)


def add_oof_bayesian(train_eng, y, test_eng, group_col, skf, k):
    global_mean = y.mean()
    oof_enc = np.full(len(train_eng), global_mean)
    groups_train = train_eng[group_col].values
    groups_test  = test_eng[group_col].values
    for tr_idx, val_idx in skf.split(train_eng, y):
        y_tr = y[tr_idx]
        g_tr = groups_train[tr_idx]
        counts   = pd.Series(g_tr).value_counts()
        means    = pd.Series(y_tr).groupby(g_tr).mean()
        smoothed = bayesian_smooth(counts, means, global_mean, k)
        oof_enc[val_idx] = pd.Series(groups_train[val_idx]).map(smoothed).fillna(global_mean).values
    counts_full   = pd.Series(groups_train).value_counts()
    means_full    = pd.Series(y).groupby(groups_train).mean()
    smoothed_full = bayesian_smooth(counts_full, means_full, global_mean, k)
    test_enc = pd.Series(groups_test).map(smoothed_full).fillna(global_mean).values
    return oof_enc, test_enc


def make_features(train_eng, y_train, test_eng, skf, num_feats, cat_feats):
    train_eng = train_eng.copy()
    test_eng  = test_eng.copy()
    train_eng["LogFare"] = np.log1p(train_eng["Fare"].fillna(train_eng["Fare"].median()))
    test_eng["LogFare"]  = np.log1p(test_eng["Fare"].fillna(test_eng["Fare"].median()))
    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"]  = test_eng["Sex"]  + "_" + test_eng["Surname"]

    ss_tr,   ss_te   = add_oof_bayesian(train_eng, y_train, test_eng, "SexSurname", skf, k=3)
    tick_tr, tick_te = add_oof_bayesian(train_eng, y_train, test_eng, "Ticket",     skf, k=5)

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scaler",  StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                                  unknown_value=-1))])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,     num_feats),
        ("cat", categorical_pipeline, cat_feats),
    ], remainder="drop")

    X_tr_base = preprocessor.fit_transform(train_eng[num_feats + cat_feats])
    X_te_base = preprocessor.transform(test_eng[num_feats + cat_feats])
    X_train = np.hstack([X_tr_base, ss_tr.reshape(-1, 1), tick_tr.reshape(-1, 1)])
    X_test  = np.hstack([X_te_base, ss_te.reshape(-1, 1), tick_te.reshape(-1, 1)])
    return X_train, X_test


def stacking_cv(X_train, y_train, X_test, seed, base_models):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_base = len(base_models)
    oof_meta  = np.zeros((len(y_train), n_base))
    test_meta = np.zeros((len(X_test),  n_base))

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]
        fold_preds = []
        for i, (name, model) in enumerate(base_models.items()):
            m = clone(model)
            try:
                m.set_params(random_state=seed)
            except ValueError:
                pass
            m.fit(X_tr, y_tr)
            oof_meta[val_idx, i] = m.predict_proba(X_val)[:, 1]
            fold_preds.append(m.predict_proba(X_test)[:, 1])
        test_meta += np.column_stack(fold_preds) / 5

    cv_scores = []
    for tr_idx, val_idx in skf.split(oof_meta, y_train):
        meta = LogisticRegression(C=0.05, max_iter=500)
        meta.fit(oof_meta[tr_idx], y_train[tr_idx])
        cv_scores.append((meta.predict(oof_meta[val_idx]) == y_train[val_idx]).mean())

    meta_final = LogisticRegression(C=0.05, max_iter=500)
    meta_final.fit(oof_meta, y_train)
    return np.array(cv_scores), meta_final.predict_proba(test_meta)[:, 1]


def run_exp(label, num_feats, cat_feats, train_eng, y_train, test_eng, skf_base,
            seeds, base_models=None):
    if base_models is None:
        base_models = BASE_5
    total = len(num_feats) + len(cat_feats) + 2
    n_base = len(base_models)
    print(f"\n--- {label}  ({total}피처, {n_base}base) ---")
    sys.stdout.flush()
    X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base, num_feats, cat_feats)
    cv_means, probas = [], []
    for seed in seeds:
        scores, tp = stacking_cv(X_train, y_train, X_test, seed, base_models)
        cv_means.append(scores.mean())
        probas.append(tp)
        print(f"  seed={seed:3d} | CV={scores.mean():.4f}")
        sys.stdout.flush()
    avg  = np.mean(cv_means)
    std  = np.std(cv_means)
    diff = avg - BASELINE
    n_surv = (np.mean(probas, axis=0) >= 0.5).sum()
    star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
    print(f"  → {len(seeds)}-seed: {avg:.4f} ±{std:.4f}  생존:{n_surv}  diff={diff:+.4f}{star}")
    sys.stdout.flush()
    return avg, std, n_surv, np.mean(probas, axis=0)


def main():
    train, test = load_raw()
    _, y_train, _, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 65)
    print("v19b: D_minimal 기반 추가 실험 (10-seed)")
    print(f"기준: D_minimal 10-seed {BASELINE:.4f}  Public 0.79186")
    print("=" * 65)

    experiments = {
        "Dmin_base":      (D_NUM,                D_CAT,                             BASE_5),
        "Dmin+CabinDeck": (D_NUM,                D_CAT + ["Cabin_Deck"],             BASE_5),
        "Dmin+AgeGroup":  (D_NUM,                D_CAT + ["AgeGroup"],               BASE_5),
        "Dmin+FamGrp":    (D_NUM,                D_CAT + ["FamilySizeGroup"],        BASE_5),
        "Dmin+SVM":       (D_NUM,                D_CAT,                             BASE_6),
        "v10_full":       (V10_NUM,              V10_CAT,                           BASE_6),
    }

    results = {}
    for label, (num, cat, base_m) in experiments.items():
        avg, std, ns, proba = run_exp(
            label, num, cat, train_eng, y_train, test_eng, skf_base, SEEDS_FULL, base_m
        )
        results[label] = (avg, std, ns, proba)

    print(f"\n{'='*65}")
    print("10-seed 결과 요약")
    print(f"{'실험':<20} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
    print("-" * 65)
    for label, (cv, std, ns, _) in results.items():
        diff = cv - BASELINE
        star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
        print(f"{label:<20} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
    sys.stdout.flush()

    # 모든 실험 제출 파일 저장 (CV 무관 — Public으로 판단)
    print(f"\n{'='*65}")
    print("제출 파일 저장 (모든 실험)")
    dmin_proba = results["Dmin_base"][3]
    v10_proba  = results["v10_full"][3]

    for label, (cv, std, ns, proba) in results.items():
        preds = (proba >= 0.5).astype(int)
        fname = f"submission_v19b_{label.lower().replace('+','_').replace(' ','')}_10s.csv"
        pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
            SUBMISSIONS / fname, index=False)
        print(f"  저장: {fname}  생존:{ns}")
        sys.stdout.flush()

    # 블렌딩: D_minimal × v10 소프트 보팅
    for w_dmin in [0.5, 0.6, 0.7, 0.75]:
        w_v10 = 1 - w_dmin
        blended = w_dmin * dmin_proba + w_v10 * v10_proba
        preds = (blended >= 0.5).astype(int)
        ns = preds.sum()
        fname = f"submission_v19b_blend_dmin{int(w_dmin*100)}_v10{int(w_v10*100)}.csv"
        pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
            SUBMISSIONS / fname, index=False)
        print(f"  저장: {fname}  생존:{ns}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
