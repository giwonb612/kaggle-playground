"""
v14b: D_noFamily 기준 추가 ablation
- D_noFamily 기준 피처: Age, Fare | Pclass, Sex, Title, Has_Cabin, Pclass_Sex | OOF×2 = 9개
- Has_Cabin 제거, Title 제거, Pclass_Sex 제거, 둘 다 제거 조합 실험
- 5-seed → 유망하면 10-seed
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
        n_estimators=569, num_leaves=33, learning_rate=0.010,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
        n_jobs=-1, verbose=-1
    ),
}

SEEDS_QUICK = [42, 7, 21, 99, 123]
SEEDS_FULL  = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]

# D_noFamily 기준
NF_NUM = ["Age", "Fare"]
NF_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]

CANDIDATES = {
    "NF_base":        {"num": NF_NUM, "cat": NF_CAT,                                       "k_ss": 3, "k_tick": 5},
    "NF_noCabin":     {"num": NF_NUM, "cat": ["Pclass", "Sex", "Title", "Pclass_Sex"],      "k_ss": 3, "k_tick": 5},
    "NF_noTitle":     {"num": NF_NUM, "cat": ["Pclass", "Sex", "Has_Cabin", "Pclass_Sex"],  "k_ss": 3, "k_tick": 5},
    "NF_noPcSex":     {"num": NF_NUM, "cat": ["Pclass", "Sex", "Title", "Has_Cabin"],       "k_ss": 3, "k_tick": 5},
    "NF_coreOnly":    {"num": NF_NUM, "cat": ["Pclass", "Sex", "Title"],                    "k_ss": 3, "k_tick": 5},
    "NF_noTitleCabin":{"num": NF_NUM, "cat": ["Pclass", "Sex", "Pclass_Sex"],               "k_ss": 3, "k_tick": 5},
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


def make_features(train_eng, y_train, test_eng, skf, cfg):
    num_feats = cfg["num"]
    cat_feats = cfg["cat"]
    k_ss      = cfg["k_ss"]
    k_tick    = cfg["k_tick"]

    train_eng = train_eng.copy()
    test_eng  = test_eng.copy()
    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"]  = test_eng["Sex"]  + "_" + test_eng["Surname"]

    ss_tr,   ss_te   = add_oof_bayesian(train_eng, y_train, test_eng, "SexSurname", skf, k=k_ss)
    tick_tr, tick_te = add_oof_bayesian(train_eng, y_train, test_eng, "Ticket",     skf, k=k_tick)

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


def run_seeds(name, cfg, train_eng, y_train, test_eng, skf_base, seeds, baseline):
    total = len(cfg["num"]) + len(cfg["cat"]) + 2
    print(f"\n--- {name}  ({total}개 피처) ---")
    sys.stdout.flush()

    X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base, cfg)
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


def main():
    train, test = load_raw()
    _, y_train, _, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    BASELINE = 0.8465  # v14 D_noFamily 10-seed

    print("=" * 60)
    print("v14b: D_noFamily 기준 추가 ablation (5-seed)")
    print(f"기준: D_noFamily 10-seed {BASELINE:.4f}")
    print("=" * 60)

    results_5s = {}
    for name, cfg in CANDIDATES.items():
        avg, std, ns, proba = run_seeds(name, cfg, train_eng, y_train, test_eng, skf_base,
                                        SEEDS_QUICK, BASELINE)
        results_5s[name] = (avg, std, ns, proba)

    print(f"\n{'='*60}")
    print("5-seed 요약")
    print(f"{'실험':<20} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
    print("-" * 60)
    base_cv = results_5s["NF_base"][0]
    for name, (cv, std, ns, _) in results_5s.items():
        diff = cv - base_cv
        star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
        print(f"{name:<20} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
    sys.stdout.flush()

    # 10-seed 검증 (5-seed에서 +0.001 이상)
    candidates_10s = {name: cfg for name, cfg in CANDIDATES.items()
                      if name != "NF_base" and results_5s[name][0] - base_cv > 0.001}

    if candidates_10s:
        print(f"\n{'='*60}")
        print(f"10-seed 검증 ({len(candidates_10s)}개 후보)")
        print("=" * 60)

        results_10s = {}
        for name, cfg in candidates_10s.items():
            avg, std, ns, proba = run_seeds(name, cfg, train_eng, y_train, test_eng, skf_base,
                                            SEEDS_FULL, BASELINE)
            results_10s[name] = (avg, std, ns, proba)
            if avg > BASELINE:
                preds = (proba >= 0.5).astype(int)
                pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
                    SUBMISSIONS / f"submission_v14b_{name.lower()}_10s.csv", index=False)
                print(f"  → 저장: submission_v14b_{name.lower()}_10s.csv")
                sys.stdout.flush()

        print(f"\n{'='*60}")
        print("10-seed 결과")
        print(f"{'실험':<20} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
        print("-" * 60)
        for name, (cv, std, ns, _) in results_10s.items():
            diff = cv - BASELINE
            star = " ★" if diff > 0 else ""
            print(f"{name:<20} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
        sys.stdout.flush()
    else:
        print("\n10-seed 스킵: 5-seed에서 유의미한 개선 없음.")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
