"""
v16 실험: 피처 변환 + 메타러너 변경
D_noFamily 기준 (9개 피처) + 다양한 변형

A: LogFare 단독 (Fare 대신)
B: Age×Pclass 상호작용 추가
C: RF 메타러너
D: GB 메타러너
E: D_noFamily + LogFare 추가 (Fare 유지, LogFare도)
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
from src.features import build_features
from src.config import RF_PARAMS, XGB_PARAMS, SUBMISSIONS

BASE_MODELS = {
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

SEEDS_QUICK = [42, 7, 21, 99, 123]
SEEDS_FULL  = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]

# D_noFamily 기준 피처
NF_NUM = ["Age", "Fare"]
NF_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]

META_LEARNERS = {
    "LR_C005": LogisticRegression(C=0.05, max_iter=500),
    "LR_C001": LogisticRegression(C=0.01, max_iter=500),
    "LR_C01":  LogisticRegression(C=0.1,  max_iter=500),
    "RF_meta": RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42),
    "GB_meta": GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1),
}


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


def make_features(train_eng, y_train, test_eng, skf, num_feats, cat_feats, extra_num=None):
    """extra_num: list of extra numeric column names to compute before preprocessing"""
    train_eng = train_eng.copy()
    test_eng  = test_eng.copy()

    # Age×Pclass interaction
    train_eng["Age_Pclass"] = train_eng["Age"].fillna(train_eng["Age"].median()) * train_eng["Pclass"]
    test_eng["Age_Pclass"]  = test_eng["Age"].fillna(test_eng["Age"].median())   * test_eng["Pclass"]

    # LogFare
    train_eng["LogFare"] = np.log1p(train_eng["Fare"].fillna(train_eng["Fare"].median()))
    test_eng["LogFare"]  = np.log1p(test_eng["Fare"].fillna(test_eng["Fare"].median()))

    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"]  = test_eng["Sex"]  + "_" + test_eng["Surname"]

    ss_tr,   ss_te   = add_oof_bayesian(train_eng, y_train, test_eng, "SexSurname", skf, k=3)
    tick_tr, tick_te = add_oof_bayesian(train_eng, y_train, test_eng, "Ticket",     skf, k=5)

    all_num = num_feats + (extra_num or [])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,     all_num),
        ("cat", categorical_pipeline, cat_feats),
    ], remainder="drop")

    X_tr_base = preprocessor.fit_transform(train_eng[all_num + cat_feats])
    X_te_base = preprocessor.transform(test_eng[all_num + cat_feats])

    X_train = np.hstack([X_tr_base, ss_tr.reshape(-1, 1), tick_tr.reshape(-1, 1)])
    X_test  = np.hstack([X_te_base, ss_te.reshape(-1, 1), tick_te.reshape(-1, 1)])
    return X_train, X_test


def stacking_cv(X_train, y_train, X_test, seed, meta_learner_cls=None):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_base = len(BASE_MODELS)
    oof_meta  = np.zeros((len(y_train), n_base))
    test_meta = np.zeros((len(X_test),  n_base))

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

    if meta_learner_cls is None:
        meta_learner_cls = LogisticRegression(C=0.05, max_iter=500)

    cv_scores = []
    for tr_idx, val_idx in skf.split(oof_meta, y_train):
        meta = clone(meta_learner_cls)
        meta.fit(oof_meta[tr_idx], y_train[tr_idx])
        cv_scores.append((meta.predict(oof_meta[val_idx]) == y_train[val_idx]).mean())

    meta_final = clone(meta_learner_cls)
    meta_final.fit(oof_meta, y_train)
    test_preds = meta_final.predict_proba(test_meta)[:, 1]
    return np.array(cv_scores), test_preds


def run_experiment(label, num_feats, cat_feats, extra_num, meta_key,
                   train_eng, y_train, test_eng, skf_base, seeds, baseline):
    total = len(num_feats) + len(extra_num or []) + len(cat_feats) + 2
    meta_cls = META_LEARNERS[meta_key]
    print(f"\n--- {label}  ({total}개 피처, meta={meta_key}) ---")
    sys.stdout.flush()

    X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base, num_feats, cat_feats, extra_num)
    cv_means, probas = [], []
    for seed in seeds:
        scores, tp = stacking_cv(X_train, y_train, X_test, seed, meta_cls)
        cv_means.append(scores.mean())
        probas.append(tp)
        print(f"  seed={seed:3d} | CV={scores.mean():.4f}")
        sys.stdout.flush()

    avg  = np.mean(cv_means)
    std  = np.std(cv_means)
    diff = avg - baseline
    n_surv = (np.mean(probas, axis=0) >= 0.5).sum()
    star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
    print(f"  → {len(seeds)}-seed: {avg:.4f} ±{std:.4f}  생존:{n_surv}  diff={diff:+.4f}{star}")
    sys.stdout.flush()
    return avg, std, n_surv, np.mean(probas, axis=0)


EXPERIMENTS = [
    # label, num, cat, extra_num, meta_key
    ("NF_base",          NF_NUM, NF_CAT, None,            "LR_C005"),
    ("A_LogFare",        ["Age", "LogFare"], NF_CAT, None, "LR_C005"),
    ("B_AgePclass",      NF_NUM, NF_CAT, ["Age_Pclass"],  "LR_C005"),
    ("C_meta_LR_C001",   NF_NUM, NF_CAT, None,            "LR_C001"),
    ("D_meta_LR_C01",    NF_NUM, NF_CAT, None,            "LR_C01"),
    ("E_meta_RF",        NF_NUM, NF_CAT, None,            "RF_meta"),
    ("F_meta_GB",        NF_NUM, NF_CAT, None,            "GB_meta"),
]


def main():
    train, test = load_raw()
    _, y_train, _, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    BASELINE = 0.8465  # v14 D_noFamily 10-seed

    print("=" * 65)
    print("v16: 피처 변환 + 메타러너 변경 실험 (5-seed)")
    print(f"기준: D_noFamily 10-seed {BASELINE:.4f}")
    print("=" * 65)

    results = {}
    for args in EXPERIMENTS:
        label, num, cat, extra, meta_key = args
        avg, std, ns, proba = run_experiment(
            label, num, cat, extra, meta_key,
            train_eng, y_train, test_eng, skf_base, SEEDS_QUICK, BASELINE
        )
        results[label] = (avg, std, ns, proba)

    print(f"\n{'='*65}")
    print("5-seed 요약")
    print(f"{'실험':<22} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
    print("-" * 65)
    base_cv = results["NF_base"][0]
    for label, (cv, std, ns, _) in results.items():
        diff = cv - base_cv
        star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
        print(f"{label:<22} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
    sys.stdout.flush()

    # 10-seed 검증
    candidates_10s = [(label, *EXPERIMENTS[[e[0] for e in EXPERIMENTS].index(label)])
                      for label, (cv, *_) in results.items()
                      if label != "NF_base" and cv - base_cv > 0.001]

    if candidates_10s:
        print(f"\n{'='*65}")
        print(f"10-seed 검증 ({len(candidates_10s)}개 후보)")
        print("=" * 65)
        results_10s = {}
        for (label, _, num, cat, extra, meta_key) in candidates_10s:
            avg, std, ns, proba = run_experiment(
                label, num, cat, extra, meta_key,
                train_eng, y_train, test_eng, skf_base, SEEDS_FULL, BASELINE
            )
            results_10s[label] = (avg, std, ns, proba)
            if avg > BASELINE:
                preds = (proba >= 0.5).astype(int)
                pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
                    SUBMISSIONS / f"submission_v16_{label.lower()}_10s.csv", index=False)
                print(f"  → 저장: submission_v16_{label.lower()}_10s.csv")
                sys.stdout.flush()

        print(f"\n{'='*65}")
        print("10-seed 결과")
        print(f"{'실험':<22} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
        print("-" * 65)
        for label, (cv, std, ns, _) in results_10s.items():
            diff = cv - BASELINE
            star = " ★" if diff > 0 else ""
            print(f"{label:<22} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
        sys.stdout.flush()
    else:
        print("\n10-seed 스킵: 유의미한 개선 없음.")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
