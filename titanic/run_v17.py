"""
v17 실험: 유망 조합 + Pseudo-labeling

A: A_LogFare + noCabin (Age, LogFare | Pclass, Sex, Title, Pclass_Sex | OOF×2)
B: Pseudo-labeling (고확신 테스트 예측 → 학습 데이터에 추가)
   - 현재 최적 설정(D_noFamily)으로 학습 후 test prob > 0.85 또는 < 0.15 케이스 추가
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


def preprocess(train_eng, test_eng, num_feats, cat_feats):
    """Add computed columns to both frames (in-place copies)."""
    train_eng = train_eng.copy()
    test_eng  = test_eng.copy()

    # LogFare
    train_eng["LogFare"] = np.log1p(train_eng["Fare"].fillna(train_eng["Fare"].median()))
    test_eng["LogFare"]  = np.log1p(test_eng["Fare"].fillna(test_eng["Fare"].median()))

    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"]  = test_eng["Sex"]  + "_" + test_eng["Surname"]
    return train_eng, test_eng


def make_features(train_eng, y_train, test_eng, skf, num_feats, cat_feats):
    train_eng, test_eng = preprocess(train_eng, test_eng, num_feats, cat_feats)

    ss_tr,   ss_te   = add_oof_bayesian(train_eng, y_train, test_eng, "SexSurname", skf, k=3)
    tick_tr, tick_te = add_oof_bayesian(train_eng, y_train, test_eng, "Ticket",     skf, k=5)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,     num_feats),
        ("cat", categorical_pipeline, cat_feats),
    ], remainder="drop")

    X_tr_base = preprocessor.fit_transform(train_eng[num_feats + cat_feats])
    X_te_base = preprocessor.transform(test_eng[num_feats + cat_feats])

    X_train = np.hstack([X_tr_base, ss_tr.reshape(-1, 1), tick_tr.reshape(-1, 1)])
    X_test  = np.hstack([X_te_base, ss_te.reshape(-1, 1), tick_te.reshape(-1, 1)])
    return X_train, X_test


def stacking_cv(X_train, y_train, X_test, seed):
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

    cv_scores = []
    for tr_idx, val_idx in skf.split(oof_meta, y_train):
        meta = LogisticRegression(C=0.05, max_iter=500)
        meta.fit(oof_meta[tr_idx], y_train[tr_idx])
        cv_scores.append((meta.predict(oof_meta[val_idx]) == y_train[val_idx]).mean())

    meta_final = LogisticRegression(C=0.05, max_iter=500)
    meta_final.fit(oof_meta, y_train)
    test_preds = meta_final.predict_proba(test_meta)[:, 1]
    return np.array(cv_scores), test_preds


def run_seeds(label, X_train, X_test, y_train, seeds, baseline):
    cv_means, probas = [], []
    for seed in seeds:
        scores, tp = stacking_cv(X_train, y_train, X_test, seed)
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


def pseudo_label_round(X_train, y_train, X_test, test_proba, threshold=0.85, seed=42):
    """Add high-confidence test pseudo-labels to training data."""
    hi_idx = np.where(test_proba >= threshold)[0]
    lo_idx = np.where(test_proba <= (1 - threshold))[0]
    pseudo_idx = np.concatenate([hi_idx, lo_idx])
    pseudo_y   = np.where(test_proba[pseudo_idx] >= threshold, 1, 0)
    n_pseudo = len(pseudo_idx)
    print(f"  pseudo-labels: {n_pseudo}개 (threshold={threshold})")
    sys.stdout.flush()
    if n_pseudo == 0:
        return X_train, y_train

    X_aug = np.vstack([X_train, X_test[pseudo_idx]])
    y_aug = np.concatenate([y_train, pseudo_y])
    return X_aug, y_aug


def main():
    train, test = load_raw()
    _, y_train, _, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    BASELINE = 0.8465  # v14 D_noFamily 10-seed

    # ── Experiment A: LogFare + noCabin ──────────────────────────────────
    A_NUM = ["Age", "LogFare"]
    A_CAT = ["Pclass", "Sex", "Title", "Pclass_Sex"]  # noCabin

    print("=" * 65)
    print("v17 Experiment A: LogFare + noCabin (5→10-seed)")
    print(f"기준: {BASELINE:.4f}")
    print("=" * 65)
    X_tr_A, X_te_A = make_features(train_eng, y_train, test_eng, skf_base, A_NUM, A_CAT)
    print(f"\n--- A_LogFare_noCabin  ({len(A_NUM)+len(A_CAT)+2}개 피처) ---")
    avg_a5, std_a5, ns_a5, pr_a5 = run_seeds("A", X_tr_A, X_te_A, y_train, SEEDS_QUICK, BASELINE)

    if avg_a5 - 0.8498 > 0.001:  # 0.8498 = NF_base 5-seed CV
        print("\n→ 10-seed 검증 진행")
        avg_a10, std_a10, ns_a10, pr_a10 = run_seeds("A_10s", X_tr_A, X_te_A, y_train, SEEDS_FULL, BASELINE)
        if avg_a10 > BASELINE:
            preds = (pr_a10 >= 0.5).astype(int)
            pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
                SUBMISSIONS / "submission_v17_a_logfare_nocabin_10s.csv", index=False)
            print("  → 저장: submission_v17_a_logfare_nocabin_10s.csv")
    else:
        print(f"  → 5-seed에서 NF_base 대비 개선 없음. 10-seed 스킵.")
    sys.stdout.flush()

    # ── Experiment B: Pseudo-labeling ────────────────────────────────────
    NF_NUM = ["Age", "Fare"]
    NF_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]

    print(f"\n{'='*65}")
    print("v17 Experiment B: Pseudo-labeling (D_noFamily 기준)")
    print("=" * 65)

    X_tr_nf, X_te_nf = make_features(train_eng, y_train, test_eng, skf_base, NF_NUM, NF_CAT)

    # 1단계: 초기 모델로 테스트 예측값 획득
    print("\n[1단계] 초기 10-seed 예측...")
    sys.stdout.flush()
    init_probas = []
    for seed in SEEDS_FULL:
        _, tp = stacking_cv(X_tr_nf, y_train, X_te_nf, seed)
        init_probas.append(tp)
        print(f"  seed={seed:3d} | done")
        sys.stdout.flush()
    avg_init_proba = np.mean(init_probas, axis=0)

    # 2단계: 여러 threshold로 pseudo-labeling 실험
    for threshold in [0.90, 0.85, 0.80]:
        print(f"\n[2단계] pseudo-labeling threshold={threshold}")
        sys.stdout.flush()
        X_aug, y_aug = pseudo_label_round(X_tr_nf, y_train, X_te_nf, avg_init_proba, threshold=threshold)
        if len(y_aug) == len(y_train):
            print("  → 추가 pseudo-label 없음. 스킵.")
            continue

        cv_means, probas = [], []
        for seed in SEEDS_QUICK:
            scores, tp = stacking_cv(X_aug, y_aug, X_te_nf, seed)
            cv_means.append(scores.mean())
            probas.append(tp)
            print(f"  seed={seed:3d} | CV={scores.mean():.4f}")
            sys.stdout.flush()

        avg  = np.mean(cv_means)
        std  = np.std(cv_means)
        diff = avg - BASELINE
        n_surv = (np.mean(probas, axis=0) >= 0.5).sum()
        star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
        print(f"  → threshold={threshold}: {avg:.4f} ±{std:.4f}  생존:{n_surv}  diff={diff:+.4f}{star}")
        sys.stdout.flush()

    print(f"\n{'='*65}")
    print("v17 완료")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
