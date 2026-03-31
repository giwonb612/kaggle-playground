"""
v14 실험: D_minimal 기반 추가 최적화
1. 개별 피처 제거 실험 (ablation)
2. D_minimal 기준 k값 재탐색

D_minimal 기준 피처 (10개):
  NUM: Age, Fare, FamilySize
  CAT: Pclass, Sex, Title, Has_Cabin, Pclass_Sex
  OOF: SexSurname(k=3), Ticket(k=5)
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

# D_minimal 기준
D_NUM = ["Age", "Fare", "FamilySize"]
D_CAT = ["Pclass", "Sex", "Title", "Has_Cabin", "Pclass_Sex"]

# Phase 1: 개별 피처 ablation 실험
ABLATION_SETS = {
    "D_base":     {"num": D_NUM,                              "cat": D_CAT,                                       "k_ss": 3, "k_tick": 5},
    "D_noAge":    {"num": ["Fare", "FamilySize"],             "cat": D_CAT,                                       "k_ss": 3, "k_tick": 5},
    "D_noFare":   {"num": ["Age", "FamilySize"],              "cat": D_CAT,                                       "k_ss": 3, "k_tick": 5},
    "D_noFamily": {"num": ["Age", "Fare"],                    "cat": D_CAT,                                       "k_ss": 3, "k_tick": 5},
    "D_noTitle":  {"num": D_NUM,                              "cat": ["Pclass", "Sex", "Has_Cabin", "Pclass_Sex"], "k_ss": 3, "k_tick": 5},
    "D_noCabin":  {"num": D_NUM,                              "cat": ["Pclass", "Sex", "Title", "Pclass_Sex"],    "k_ss": 3, "k_tick": 5},
    "D_noPcSex":  {"num": D_NUM,                              "cat": ["Pclass", "Sex", "Title", "Has_Cabin"],     "k_ss": 3, "k_tick": 5},
}

# Phase 2: k값 재탐색 (D_minimal 기준)
K_SETS = {
    "k_ss1_tick3":  {"num": D_NUM, "cat": D_CAT, "k_ss": 1, "k_tick": 3},
    "k_ss2_tick3":  {"num": D_NUM, "cat": D_CAT, "k_ss": 2, "k_tick": 3},
    "k_ss3_tick3":  {"num": D_NUM, "cat": D_CAT, "k_ss": 3, "k_tick": 3},
    "k_ss3_tick5":  {"num": D_NUM, "cat": D_CAT, "k_ss": 3, "k_tick": 5},  # 기준 (D_base)
    "k_ss3_tick7":  {"num": D_NUM, "cat": D_CAT, "k_ss": 3, "k_tick": 7},
    "k_ss5_tick5":  {"num": D_NUM, "cat": D_CAT, "k_ss": 5, "k_tick": 5},
    "k_ss5_tick7":  {"num": D_NUM, "cat": D_CAT, "k_ss": 5, "k_tick": 7},
    "k_ss7_tick5":  {"num": D_NUM, "cat": D_CAT, "k_ss": 7, "k_tick": 5},
    "k_ss7_tick7":  {"num": D_NUM, "cat": D_CAT, "k_ss": 7, "k_tick": 7},
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
        counts  = pd.Series(g_tr).value_counts()
        means   = pd.Series(y_tr).groupby(g_tr).mean()
        smoothed = bayesian_smooth(counts, means, global_mean, k)
        oof_enc[val_idx] = pd.Series(groups_train[val_idx]).map(smoothed).fillna(global_mean).values
    counts_full  = pd.Series(groups_train).value_counts()
    means_full   = pd.Series(y).groupby(groups_train).mean()
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
        ("num", numeric_pipeline,    num_feats),
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
    print(f"\n--- {name}  ({total}개 피처, k_ss={cfg['k_ss']}, k_tick={cfg['k_tick']}) ---")
    sys.stdout.flush()

    X_train, X_test = make_features(train_eng, y_train, test_eng, skf_base, cfg)
    cv_means, probas = [], []
    for seed in seeds:
        scores, tp = stacking_cv(X_train, y_train, X_test, seed)
        cv_means.append(scores.mean())
        probas.append(tp)
        print(f"  seed={seed:3d} | CV={scores.mean():.4f}")
        sys.stdout.flush()

    avg = np.mean(cv_means)
    std = np.std(cv_means)
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

    BASELINE = 0.8438  # v13 D_minimal 10-seed

    # ── Phase 1: Ablation (5-seed 빠른 탐색) ──────────────────────────────
    print("=" * 60)
    print("v14 Phase 1: D_minimal 피처 ablation (5-seed)")
    print(f"기준: D_minimal 10-seed {BASELINE:.4f}")
    print("=" * 60)

    p1_results = {}
    for name, cfg in ABLATION_SETS.items():
        avg, std, ns, proba = run_seeds(name, cfg, train_eng, y_train, test_eng, skf_base,
                                        SEEDS_QUICK, BASELINE)
        p1_results[name] = (avg, std, ns, proba)

    print(f"\n{'='*60}")
    print("Phase 1 요약 (5-seed ablation)")
    print(f"{'실험':<15} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
    print("-" * 60)
    base_cv = p1_results["D_base"][0]
    for name, (cv, std, ns, _) in p1_results.items():
        diff = cv - base_cv
        star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
        print(f"{name:<15} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
    sys.stdout.flush()

    # ── Phase 2: k값 재탐색 (5-seed) ─────────────────────────────────────
    print(f"\n{'='*60}")
    print("v14 Phase 2: D_minimal k값 재탐색 (5-seed)")
    print("=" * 60)

    p2_results = {}
    for name, cfg in K_SETS.items():
        avg, std, ns, proba = run_seeds(name, cfg, train_eng, y_train, test_eng, skf_base,
                                        SEEDS_QUICK, BASELINE)
        p2_results[name] = (avg, std, ns, proba)

    print(f"\n{'='*60}")
    print("Phase 2 요약 (5-seed k값 탐색)")
    print(f"{'실험':<16} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
    print("-" * 60)
    ref_cv = p2_results["k_ss3_tick5"][0]
    for name, (cv, std, ns, _) in p2_results.items():
        diff = cv - ref_cv
        star = " ★" if diff > 0.001 else (" △" if diff > 0 else "")
        print(f"{name:<16} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
    sys.stdout.flush()

    # ── Phase 3: 10-seed 검증 (5-seed 개선 후보만) ───────────────────────
    # 5-seed에서 +0.001 이상인 후보 자동 선택
    candidates_10s = {}
    for name, (cv, std, ns, proba) in {**p1_results, **p2_results}.items():
        if name in ("D_base", "k_ss3_tick5"):
            continue
        if cv - base_cv > 0.001:
            candidates_10s[name] = {
                **ABLATION_SETS.get(name, K_SETS.get(name)),
            }

    if candidates_10s:
        print(f"\n{'='*60}")
        print(f"Phase 3: 10-seed 검증 ({len(candidates_10s)}개 후보)")
        print("=" * 60)

        p3_results = {}
        for name, cfg in candidates_10s.items():
            cfg_full = ABLATION_SETS.get(name) or K_SETS.get(name)
            avg, std, ns, proba = run_seeds(name, cfg_full, train_eng, y_train, test_eng, skf_base,
                                            SEEDS_FULL, BASELINE)
            p3_results[name] = (avg, std, ns, proba)
            if avg > BASELINE:
                preds = (proba >= 0.5).astype(int)
                pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(
                    SUBMISSIONS / f"submission_v14_{name.lower()}_10s.csv", index=False)
                print(f"  → 저장: submission_v14_{name.lower()}_10s.csv")
                sys.stdout.flush()

        print(f"\n{'='*60}")
        print("Phase 3 요약 (10-seed)")
        print(f"{'실험':<16} | {'CV':>7} | {'std':>7} | {'생존':>5} | {'diff':>7}")
        print("-" * 60)
        for name, (cv, std, ns, _) in p3_results.items():
            diff = cv - BASELINE
            star = " ★" if diff > 0 else ""
            print(f"{name:<16} | {cv:.4f} | ±{std:.4f} | {ns:>5} | {diff:+.4f}{star}")
        sys.stdout.flush()
    else:
        print("\nPhase 3: 5-seed에서 유의미한 개선 없음. 10-seed 스킵.")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
