"""
v17 Pseudo-labeling 제출 파일 생성
주의: CV는 누수로 인해 신뢰 불가. Public score로만 평가.

CV 누수 이유:
  - pseudo-label은 전체 891 샘플로 훈련된 모델에서 유도
  - 해당 테스트 샘플이 CV val fold에도 포함 → 인위적 CV 상승
  - 실제 테스트 예측값은 더 많은 데이터로 훈련한 모델에서 나오므로 유효할 수 있음
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

SEEDS_FULL = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]
NF_NUM = ["Age", "Fare"]
NF_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]


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


def make_base_features(train_eng, y_train, test_eng, skf):
    train_eng = train_eng.copy()
    test_eng  = test_eng.copy()
    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"]  = test_eng["Sex"]  + "_" + test_eng["Surname"]

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
        ("num", numeric_pipeline,     NF_NUM),
        ("cat", categorical_pipeline, NF_CAT),
    ], remainder="drop")

    X_tr_base = preprocessor.fit_transform(train_eng[NF_NUM + NF_CAT])
    X_te_base = preprocessor.transform(test_eng[NF_NUM + NF_CAT])

    X_train = np.hstack([X_tr_base, ss_tr.reshape(-1, 1), tick_tr.reshape(-1, 1)])
    X_test  = np.hstack([X_te_base, ss_te.reshape(-1, 1), tick_te.reshape(-1, 1)])
    return X_train, X_test


def stacking_predict(X_train, y_train, X_test, seed):
    """Train on full data and predict test (no CV eval)."""
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

    meta_final = LogisticRegression(C=0.05, max_iter=500)
    meta_final.fit(oof_meta, y_train)
    return meta_final.predict_proba(test_meta)[:, 1]


def main():
    train, test = load_raw()
    _, y_train, _, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("=" * 60)
    print("v17 Pseudo-labeling 제출 파일 생성")
    print("(CV 누수 경고: Public score로만 평가)")
    print("=" * 60)

    # Step 1: 초기 10-seed 예측 (D_noFamily 기준)
    print("\n[Step 1] 초기 10-seed 예측...")
    sys.stdout.flush()
    X_tr, X_te = make_base_features(train_eng, y_train, test_eng, skf_base)
    init_probas = []
    for seed in SEEDS_FULL:
        tp = stacking_predict(X_tr, y_train, X_te, seed)
        init_probas.append(tp)
        print(f"  seed={seed:3d} | done  (prob range: {tp.min():.3f}~{tp.max():.3f})")
        sys.stdout.flush()
    avg_proba = np.mean(init_probas, axis=0)
    print(f"\n  전체 예측 확률 분포: min={avg_proba.min():.3f}, max={avg_proba.max():.3f}, "
          f"mean={avg_proba.mean():.3f}")
    print(f"  >0.85: {(avg_proba > 0.85).sum()}개, >0.80: {(avg_proba > 0.80).sum()}개, "
          f"<0.15: {(avg_proba < 0.15).sum()}개, <0.20: {(avg_proba < 0.20).sum()}개")
    sys.stdout.flush()

    # Step 2: 각 threshold로 pseudo-labeling 후 10-seed 평균 예측
    for threshold in [0.85, 0.80]:
        hi_idx = np.where(avg_proba >= threshold)[0]
        lo_idx = np.where(avg_proba <= (1 - threshold))[0]
        pseudo_idx = np.concatenate([hi_idx, lo_idx])
        pseudo_y   = np.where(avg_proba[pseudo_idx] >= threshold, 1, 0)
        n_pseudo   = len(pseudo_idx)

        print(f"\n[Step 2] threshold={threshold}: {n_pseudo}개 pseudo-label 추가")
        print(f"  생존(1): {pseudo_y.sum()}개, 사망(0): {(pseudo_y==0).sum()}개")
        sys.stdout.flush()

        if n_pseudo == 0:
            print("  → 스킵")
            continue

        X_aug = np.vstack([X_tr, X_te[pseudo_idx]])
        y_aug = np.concatenate([y_train, pseudo_y])

        print(f"  학습 샘플: {len(y_aug)}개 (원본 {len(y_train)} + pseudo {n_pseudo})")
        sys.stdout.flush()

        aug_probas = []
        for seed in SEEDS_FULL:
            tp = stacking_predict(X_aug, y_aug, X_te, seed)
            aug_probas.append(tp)
            print(f"  seed={seed:3d} | done")
            sys.stdout.flush()

        avg_aug_proba = np.mean(aug_probas, axis=0)
        preds = (avg_aug_proba >= 0.5).astype(int)
        n_surv = preds.sum()
        fname = f"submission_v17_pseudo{int(threshold*100)}_10s.csv"
        pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
            SUBMISSIONS / fname, index=False)
        print(f"  → 저장: {fname}  생존예측: {n_surv}명")
        sys.stdout.flush()

    print("\n완료.")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
