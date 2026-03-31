"""
v13 실험: 피처 삭제 실험 (Feature Pruning)
- 소규모 데이터에서 노이즈 피처 제거 = 일반화 향상 가능
- 현재 피처 수: 19개 (NUM 7 + CAT 10 + OOF 2)
- SHAP 중요도 낮은 피처부터 제거하며 CV 변화 추적

실험 계획:
- A: LogFare, FarePerPerson 계열 제거 (Fare 중복 정보)
- B: IsChild, IsMother, SurnameFreq 제거 (v2에서 실패한 피처들)
- C: CAT 피처 일부 제거 (Embarked, IsAlone 등 중요도 낮은 것)
- D: 전체 축소 버전 (핵심 10개만)
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
from src.features import build_features, add_oof_survival_encoding
from src.config import RF_PARAMS, XGB_PARAMS, SUBMISSIONS, NUM_FEATURES, CAT_FEATURES

BASE_MODELS = {
    'LR':  LogisticRegression(C=1.0, max_iter=1000),
    'RF':  RandomForestClassifier(**RF_PARAMS),
    'GB':  GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1),
    'XGB': xgb.XGBClassifier(**XGB_PARAMS),
    'LGBM_LL': lgb.LGBMClassifier(
        n_estimators=569, num_leaves=33, learning_rate=0.010,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        n_jobs=-1, verbose=-1
    ),
}

SEEDS_QUICK = [42, 7, 21, 99, 123]
SEEDS_FULL = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]

# 피처 셋 정의
FEAT_SETS = {
    "v10_full": {
        "num": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketFreq", "LogFare"],
        "cat": ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "Has_Cabin",
                "Cabin_Deck", "FamilySizeGroup", "AgeGroup", "Pclass_Sex"],
    },
    "A_noLogFare": {
        "num": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketFreq"],
        "cat": ["Pclass", "Sex", "Embarked", "Title", "IsAlone", "Has_Cabin",
                "Cabin_Deck", "FamilySizeGroup", "AgeGroup", "Pclass_Sex"],
    },
    "B_noIslands": {
        # IsChild, IsMother, SurnameFreq는 engineer_features에서 생성되지만 base 피처셋에 없음
        # → 실제로는 Embarked, IsAlone 제거 실험
        "num": ["Age", "Fare", "SibSp", "Parch", "FamilySize", "TicketFreq", "LogFare"],
        "cat": ["Pclass", "Sex", "Title", "Has_Cabin",
                "Cabin_Deck", "FamilySizeGroup", "AgeGroup", "Pclass_Sex"],
    },
    "C_core12": {
        # 핵심 12개만
        "num": ["Age", "Fare", "SibSp", "FamilySize", "TicketFreq"],
        "cat": ["Pclass", "Sex", "Title", "Has_Cabin",
                "Cabin_Deck", "AgeGroup", "Pclass_Sex"],
    },
    "D_minimal": {
        # 최소 핵심 피처
        "num": ["Age", "Fare", "FamilySize"],
        "cat": ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"],
    },
}


def bayesian_smooth(counts, means, global_mean, k):
    return (counts * means + k * global_mean) / (counts + k)


def add_oof_bayesian(train_eng, y, test_eng, group_col, skf, k):
    global_mean = y.mean()
    oof_enc = np.full(len(train_eng), global_mean)
    groups_train = train_eng[group_col].values
    groups_test = test_eng[group_col].values
    for tr_idx, val_idx in skf.split(train_eng, y):
        y_tr = y[tr_idx]
        g_tr = groups_train[tr_idx]
        counts = pd.Series(g_tr).value_counts()
        means = pd.Series(y_tr).groupby(g_tr).mean()
        smoothed = bayesian_smooth(counts, means, global_mean, k)
        oof_enc[val_idx] = pd.Series(groups_train[val_idx]).map(smoothed).fillna(global_mean).values
    counts_full = pd.Series(groups_train).value_counts()
    means_full = pd.Series(y).groupby(groups_train).mean()
    smoothed_full = bayesian_smooth(counts_full, means_full, global_mean, k)
    test_enc = pd.Series(groups_test).map(smoothed_full).fillna(global_mean).values
    return oof_enc, test_enc


def make_features(train_eng, y_train, test_eng, skf, feat_set):
    num_feats = feat_set["num"]
    cat_feats = feat_set["cat"]

    train_eng = train_eng.copy()
    test_eng = test_eng.copy()
    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"] = test_eng["Sex"] + "_" + test_eng["Surname"]

    ss_tr, ss_te = add_oof_bayesian(train_eng, y_train, test_eng, "SexSurname", skf, k=3)
    tick_tr, tick_te = add_oof_bayesian(train_eng, y_train, test_eng, "Ticket", skf, k=5)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_feats),
        ("cat", categorical_pipeline, cat_feats),
    ], remainder="drop")

    X_tr_base = preprocessor.fit_transform(train_eng[num_feats + cat_feats])
    X_te_base = preprocessor.transform(test_eng[num_feats + cat_feats])

    X_train = np.hstack([X_tr_base, ss_tr.reshape(-1, 1), tick_tr.reshape(-1, 1)])
    X_test = np.hstack([X_te_base, ss_te.reshape(-1, 1), tick_te.reshape(-1, 1)])
    return X_train, X_test


def stacking_cv(X_train, y_train, X_test, seed):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_base = len(BASE_MODELS)
    oof_meta = np.zeros((len(y_train), n_base))
    test_meta = np.zeros((len(X_test), n_base))

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr = y_train[tr_idx]
        fold_test_preds = []
        for i, (name, model) in enumerate(BASE_MODELS.items()):
            m = clone(model)
            try:
                m.set_params(random_state=seed)
            except ValueError:
                pass
            m.fit(X_tr, y_tr)
            oof_meta[val_idx, i] = m.predict_proba(X_val)[:, 1]
            fold_test_preds.append(m.predict_proba(X_test)[:, 1])
        test_meta += np.column_stack(fold_test_preds) / 5

    cv_scores = []
    for tr_idx, val_idx in skf.split(oof_meta, y_train):
        meta = LogisticRegression(C=0.05, max_iter=500)
        meta.fit(oof_meta[tr_idx], y_train[tr_idx])
        cv_scores.append((meta.predict(oof_meta[val_idx]) == y_train[val_idx]).mean())

    meta_final = LogisticRegression(C=0.05, max_iter=500)
    meta_final.fit(oof_meta, y_train)
    test_preds = meta_final.predict_proba(test_meta)[:, 1]
    return np.array(cv_scores), test_preds


def run_experiment(name, feat_set, train_eng, y_train, test_eng, skf_base, test_ids,
                   seeds=SEEDS_QUICK, baseline_cv=None):
    total_feats = len(feat_set["num"]) + len(feat_set["cat"]) + 2  # +2 OOF
    print(f"\n{'='*55}")
    print(f"실험: {name}  ({len(seeds)}-seed, {total_feats}개 피처)")
    print(f"{'='*55}")
    sys.stdout.flush()

    X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base, feat_set)

    cv_means, test_probas = [], []
    for seed in seeds:
        cv_scores, tp = stacking_cv(X_train, y_train, X_test, seed)
        cv_means.append(cv_scores.mean())
        test_probas.append(tp)
        print(f"  seed={seed:3d} | CV={cv_scores.mean():.4f} ±{cv_scores.std():.4f}")
        sys.stdout.flush()

    avg_cv = np.mean(cv_means)
    std_cv = np.std(cv_means)
    avg_preds = (np.mean(test_probas, axis=0) >= 0.5).astype(int)
    n_survived = avg_preds.sum()

    ref = baseline_cv or 0.8412
    diff = avg_cv - ref
    print(f"\n  {len(seeds)}-seed 평균: {avg_cv:.4f} ±{std_cv:.4f}  생존:{n_survived}명  diff={diff:+.4f}")
    sys.stdout.flush()

    pd.DataFrame({"PassengerId": test_ids, "Survived": avg_preds}).to_csv(
        SUBMISSIONS / f"submission_v13_{name.lower()}.csv", index=False)
    return avg_cv, std_cv, n_survived


def main():
    train, test = load_raw()
    _, y_train, _, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 55)
    print("v13 피처 삭제 실험 — 5-seed 빠른 탐색")
    print("기준: v10 10-seed 84.12%")
    print("=" * 55)
    sys.stdout.flush()

    results = {}
    baseline_cv = None

    for name, feat_set in FEAT_SETS.items():
        cv, std, ns = run_experiment(
            name, feat_set, train_eng, y_train, test_eng, skf_base, test_ids,
            seeds=SEEDS_QUICK, baseline_cv=baseline_cv
        )
        results[name] = (cv, std, ns)
        if name == "v10_full":
            baseline_cv = cv

    # 요약
    print(f"\n{'='*55}")
    print("v13 결과 요약 (5-seed)")
    print(f"{'실험':<15} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
    print("-" * 55)
    base = results["v10_full"][0]
    for name, (cv, std, ns) in results.items():
        diff = cv - base
        star = " ★" if diff > 0.002 else (" △" if diff > 0 else "")
        print(f"{name:<15} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
