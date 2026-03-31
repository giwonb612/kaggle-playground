"""
v12 실험: Bayesian Smoothing k값 그리드 서치
- SexSurname k: [1, 2, 3, 5, 7, 10]
- Ticket k: [1, 3, 5, 7, 10]
- 총 30개 조합, 5-seed 빠른 탐색 → 상위 3개 10-seed 검증
- v10 설정 (LGBM_LowLR + C=0.05, SVM 제외)
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
        n_estimators=569, num_leaves=33, learning_rate=0.010,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        n_jobs=-1, verbose=-1
    ),
}

SEEDS_QUICK = [42, 7, 21, 99, 123]
SEEDS_FULL = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]

# 그리드
K_SEXSURNAME = [1, 2, 3, 5, 7, 10]
K_TICKET = [1, 3, 5, 7, 10]


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


def make_features(train_eng, y_train, test_eng, skf, k_ss, k_tick):
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from src.config import NUM_FEATURES, CAT_FEATURES

    train_eng = train_eng.copy()
    test_eng = test_eng.copy()

    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"] = test_eng["Sex"] + "_" + test_eng["Surname"]

    ss_tr, ss_te = add_oof_bayesian(train_eng, y_train, test_eng, "SexSurname", skf, k_ss)
    tick_tr, tick_te = add_oof_bayesian(train_eng, y_train, test_eng, "Ticket", skf, k_tick)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUM_FEATURES),
        ("cat", categorical_pipeline, CAT_FEATURES),
    ], remainder="drop")

    X_tr_base = preprocessor.fit_transform(train_eng[NUM_FEATURES + CAT_FEATURES])
    X_te_base = preprocessor.transform(test_eng[NUM_FEATURES + CAT_FEATURES])

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


def run_grid(train_eng, y_train, test_eng, skf_base, test_ids, seeds):
    results = {}
    total = len(K_SEXSURNAME) * len(K_TICKET)
    done = 0

    print(f"\n총 {total}개 조합, {len(seeds)}-seed CV")
    print(f"{'k_ss':>4} {'k_tick':>6} | {'CV':>7} {'std':>7} | diff")
    print("-" * 40)
    sys.stdout.flush()

    for k_ss in K_SEXSURNAME:
        for k_tick in K_TICKET:
            X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base, k_ss, k_tick)
            cv_means = []
            test_probas = []
            for seed in seeds:
                cv_scores, tp = stacking_cv(X_train, y_train, X_test, seed)
                cv_means.append(cv_scores.mean())
                test_probas.append(tp)
            avg = np.mean(cv_means)
            std = np.std(cv_means)
            results[(k_ss, k_tick)] = {
                'cv': avg, 'std': std,
                'preds': (np.mean(test_probas, axis=0) >= 0.5).astype(int)
            }
            diff = avg - 0.8412  # v10 10-seed baseline
            star = " ★" if avg > 0.8412 else ""
            done += 1
            print(f"  k_ss={k_ss:2d} k_tick={k_tick:2d} | {avg:.4f} ±{std:.4f} | {diff:+.4f}{star}  [{done}/{total}]")
            sys.stdout.flush()

    return results


def main():
    train, test = load_raw()
    _, y_train, _, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Phase 1: 5-seed 그리드 탐색 ──────────────────────────────────────
    print("=" * 60)
    print("v12 k값 그리드 서치 — Phase 1 (5-seed)")
    print("현재 v10: k_ss=3, k_tick=5, CV 84.12%")
    print("=" * 60)
    sys.stdout.flush()

    results = run_grid(train_eng, y_train, test_eng, skf_base, test_ids, SEEDS_QUICK)

    # 상위 3개 출력
    top3 = sorted(results.items(), key=lambda x: -x[1]['cv'])[:3]
    print(f"\n{'='*60}")
    print("Phase 1 상위 3개:")
    for (k_ss, k_tick), r in top3:
        diff = r['cv'] - 0.8412
        print(f"  k_ss={k_ss}, k_tick={k_tick}: {r['cv']:.4f} ±{r['std']:.4f}  diff={diff:+.4f}")
    sys.stdout.flush()

    # ── Phase 2: 상위 후보 10-seed 검증 ──────────────────────────────────
    candidates = [(k, r) for k, r in top3 if r['cv'] > 0.8412]
    if candidates:
        print(f"\n{'='*60}")
        print(f"Phase 2: 10-seed 검증 ({len(candidates)}개 후보)")
        print(f"{'='*60}")
        sys.stdout.flush()

        best_10s = None
        best_10s_cv = 0
        best_10s_key = None

        for (k_ss, k_tick), _ in candidates:
            X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base, k_ss, k_tick)
            cv_means, test_probas = [], []
            for seed in SEEDS_FULL:
                cv_scores, tp = stacking_cv(X_train, y_train, X_test, seed)
                cv_means.append(cv_scores.mean())
                test_probas.append(tp)
                print(f"  k_ss={k_ss} k_tick={k_tick} seed={seed:3d} | CV={cv_scores.mean():.4f}")
                sys.stdout.flush()
            avg10 = np.mean(cv_means)
            std10 = np.std(cv_means)
            diff10 = avg10 - 0.8412
            print(f"  → 10-seed: {avg10:.4f} ±{std10:.4f}  diff={diff10:+.4f}")
            sys.stdout.flush()

            if avg10 > best_10s_cv:
                best_10s_cv = avg10
                best_10s = np.mean(test_probas, axis=0)
                best_10s_key = (k_ss, k_tick)

        if best_10s_cv > 0.8412:
            k_ss, k_tick = best_10s_key
            preds = (best_10s >= 0.5).astype(int)
            fname = f"submission_v12_ks{k_ss}_kt{k_tick}.csv"
            pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
                SUBMISSIONS / fname, index=False)
            print(f"\n제출 파일 저장: {fname}  (생존 {preds.sum()}명)")
            sys.stdout.flush()
        else:
            print("\nPhase 2: v10 대비 개선 없음 → 제출 보류")
            sys.stdout.flush()
    else:
        print("\nPhase 1: 유망 후보 없음 → Phase 2 생략")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
