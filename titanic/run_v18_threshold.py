"""
v18: D_minimal threshold 최적화
- D_minimal 10-seed 예측 확률 분포 분석
- threshold 0.40~0.55 탐색 → 각 생존 예측 수 확인
- 제출 후보: 137(현재) 근처 + 기대값(~150-160) 근처 파일 생성

참고: Titanic 훈련셋 생존율 38.4% (342/891)
      418명 기준 기대 생존 예측: 160명 ± 15명
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

D_NUM = ["Age", "Fare", "FamilySize"]
D_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]


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


def make_features(train_eng, y_train, test_eng, skf):
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
        ("num", numeric_pipeline,     D_NUM),
        ("cat", categorical_pipeline, D_CAT),
    ], remainder="drop")

    X_tr_base = preprocessor.fit_transform(train_eng[D_NUM + D_CAT])
    X_te_base = preprocessor.transform(test_eng[D_NUM + D_CAT])

    X_train = np.hstack([X_tr_base, ss_tr.reshape(-1, 1), tick_tr.reshape(-1, 1)])
    X_test  = np.hstack([X_te_base, ss_te.reshape(-1, 1), tick_te.reshape(-1, 1)])
    return X_train, X_test


def stacking_predict(X_train, y_train, X_test, seed):
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
    print("v18: D_minimal threshold 최적화")
    print("=" * 60)

    # 10-seed 확률 생성
    print("\n[1] D_minimal 10-seed 예측 확률 생성...")
    sys.stdout.flush()
    X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base)
    probas = []
    for seed in SEEDS_FULL:
        tp = stacking_predict(X_train, y_train, X_test, seed)
        probas.append(tp)
        print(f"  seed={seed:3d} done")
        sys.stdout.flush()
    avg_proba = np.mean(probas, axis=0)

    print(f"\n  확률 분포:")
    print(f"  min={avg_proba.min():.4f}, max={avg_proba.max():.4f}, mean={avg_proba.mean():.4f}")
    print(f"  std={avg_proba.std():.4f}")

    # 분위수
    for q in [0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90]:
        print(f"  {int(q*100)}%tile = {np.quantile(avg_proba, q):.4f}")
    sys.stdout.flush()

    # Threshold 탐색
    print(f"\n[2] Threshold 탐색")
    print(f"{'threshold':>12} | {'생존 예측':>8} | {'사망 예측':>8}")
    print("-" * 40)
    for t in np.arange(0.35, 0.60, 0.01):
        preds = (avg_proba >= t).astype(int)
        n_surv = preds.sum()
        n_dead = 418 - n_surv
        marker = " ← 현재(0.50)" if abs(t - 0.50) < 0.005 else ""
        marker += " ← 기대(160)" if abs(n_surv - 160) <= 2 else ""
        marker += " ← 기대(152)" if abs(n_surv - 152) <= 2 else ""
        print(f"  t={t:.2f}     | {n_surv:>8} | {n_dead:>8}{marker}")
    sys.stdout.flush()

    # 특정 threshold로 제출 파일 생성
    # 현재 0.50 = 137명 → 140~165 사이를 커버
    saved = []
    for t in np.arange(0.30, 0.55, 0.01):
        preds = (avg_proba >= t).astype(int)
        n_surv = preds.sum()
        # 관심 구간만 저장: 130, 140, 150, 155, 160, 165
        if n_surv in [130, 135, 140, 145, 150, 152, 155, 160, 165, 170]:
            t_str = f"{int(t*100):02d}"
            fname = f"submission_v18_thresh{t_str}_surv{n_surv}.csv"
            pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
                SUBMISSIONS / fname, index=False)
            saved.append((t, n_surv, fname))
            print(f"  저장: {fname}")
            sys.stdout.flush()

    print(f"\n[3] 저장 파일 목록")
    for t, n, fname in saved:
        print(f"  t={t:.2f} | 생존={n:3d} | {fname}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
