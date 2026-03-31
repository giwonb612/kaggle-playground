"""
v11 실험: OOF 피처 다양화
- v10 최고 설정 (LGBM_LowLR 베이스 + meta C=0.05) 위에 추가 OOF 피처 실험
- 실험 A: TicketPrefix OOF (PC, C, SOTON 등 사회계층 반영)
- 실험 B: AgeGroup×Sex OOF (Baby_female, Adult_male 등)
- 실험 C: A + B 동시 추가
- 먼저 5-seed CV로 빠른 방향성 확인 후 10-seed로 검증

속도 최적화:
- SVM 제거 (v10에서 -SVM 변형도 비슷했음)
- cross_val_score 대신 직접 fold 계산으로 재사용 효율화
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

# ── v10 최고 설정 (SVM 제거로 속도 개선, -SVM CV diff는 v10에서 -0.0003으로 무시 가능) ──
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
META = LogisticRegression(C=0.05, max_iter=500)

# 빠른 방향 확인: 5-seed, 상세 검증: 10-seed
SEEDS_QUICK = [42, 7, 21, 99, 123]
SEEDS_FULL = [42, 7, 21, 99, 123, 0, 1, 2, 3, 4]


def bayesian_smooth(series_n: pd.Series, series_mean: pd.Series,
                    global_mean: float, k: float) -> pd.Series:
    """Bayesian smoothing: (n*mean + k*global) / (n+k)"""
    return (series_n * series_mean + k * global_mean) / (series_n + k)


def add_oof_bayesian(train_eng, y, test_eng, group_col, skf, k=3):
    """OOF mean encoding with Bayesian smoothing."""
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

    # Test: full train
    counts_full = pd.Series(groups_train).value_counts()
    means_full = pd.Series(y).groupby(groups_train).mean()
    smoothed_full = bayesian_smooth(counts_full, means_full, global_mean, k)
    test_enc = pd.Series(groups_test).map(smoothed_full).fillna(global_mean).values
    return oof_enc, test_enc


def extract_ticket_prefix(ticket: str) -> str:
    """티켓 번호에서 prefix 추출. 숫자만이면 'NUM'."""
    parts = ticket.split()
    if len(parts) == 1:
        return "NUM"
    prefix = parts[0].replace(".", "").replace("/", "").upper()
    # 주요 prefix만 유지, 나머지는 OTHER
    major = {"PC", "CA", "SC", "SOTON", "A", "W", "PP", "LINE", "STON"}
    return prefix if prefix in major else "OTHER"


def make_features_v11(train_eng, y_train, test_eng, skf, extra_oofs):
    """
    v10 기본 OOF (SexSurname k=3, Ticket k=5) + extra_oofs 추가.
    extra_oofs: list of (group_col, k)
    Returns (X_train_extra, X_test_extra) — 기본 OOF 포함한 전체 행렬
    """
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from src.config import NUM_FEATURES, CAT_FEATURES

    # 기본 OOF (v9/v10과 동일)
    train_eng = train_eng.copy()
    test_eng = test_eng.copy()

    train_eng["SexSurname"] = train_eng["Sex"] + "_" + train_eng["Surname"]
    test_eng["SexSurname"] = test_eng["Sex"] + "_" + test_eng["Surname"]
    ss_tr, ss_te = add_oof_bayesian(train_eng, y_train, test_eng, "SexSurname", skf, k=3)
    tick_tr, tick_te = add_oof_bayesian(train_eng, y_train, test_eng, "Ticket", skf, k=5)

    extra_tr_cols = [ss_tr, tick_tr]
    extra_te_cols = [ss_te, tick_te]

    # 추가 OOF
    for gcol, k in extra_oofs:
        tr_enc, te_enc = add_oof_bayesian(train_eng, y_train, test_eng, gcol, skf, k=k)
        extra_tr_cols.append(tr_enc)
        extra_te_cols.append(te_enc)

    # 기본 피처 전처리
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

    X_train = np.hstack([X_tr_base] + [c.reshape(-1, 1) for c in extra_tr_cols])
    X_test = np.hstack([X_te_base] + [c.reshape(-1, 1) for c in extra_te_cols])
    return X_train, X_test


def stacking_cv(X_train, y_train, X_test, seed):
    """단일 seed stacking 실행 (올바른 CV 계산).
    1) 전체 5-fold OOF로 base model meta-features 수집
    2) 수집된 OOF meta-features로 또다른 5-fold CV → meta-learner CV 계산
    Returns (cv_scores array, test_preds_proba)
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    n_base = len(BASE_MODELS)
    oof_meta = np.zeros((len(y_train), n_base))
    test_meta = np.zeros((len(X_test), n_base))

    # Step 1: 모든 fold에서 base model OOF 수집
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

    # Step 2: meta-learner를 OOF meta-features에서 CV 평가
    # v10과 동일한 방식: meta OOF의 5-fold CV
    cv_scores = []
    for tr_idx, val_idx in skf.split(oof_meta, y_train):
        meta = LogisticRegression(C=0.05, max_iter=500)
        meta.fit(oof_meta[tr_idx], y_train[tr_idx])
        cv_scores.append((meta.predict(oof_meta[val_idx]) == y_train[val_idx]).mean())

    # Step 3: final test predictions
    meta_final = LogisticRegression(C=0.05, max_iter=500)
    meta_final.fit(oof_meta, y_train)
    test_preds = meta_final.predict_proba(test_meta)[:, 1]
    return np.array(cv_scores), test_preds


def run_experiment(name, extra_oofs, train_eng_base, y_train, test_eng_base,
                   skf_base, test_ids, seeds=None, baseline_cv=None):
    """N-seed 반복 실험."""
    if seeds is None:
        seeds = SEEDS_QUICK
    print(f"\n{'='*60}")
    print(f"실험: {name}  ({len(seeds)}-seed)")
    print(f"추가 OOF: {[g for g, _ in extra_oofs] if extra_oofs else '없음 (v10-nosvm baseline)'}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # 피처 빌드
    X_train, X_test = make_features_v11(train_eng_base, y_train, test_eng_base,
                                         skf_base, extra_oofs)

    all_cv_means = []
    all_test_proba = []

    for seed in seeds:
        cv_scores, test_proba = stacking_cv(X_train, y_train, X_test, seed)
        all_cv_means.append(cv_scores.mean())
        all_test_proba.append(test_proba)
        print(f"  seed={seed:3d} | CV={cv_scores.mean():.4f} ±{cv_scores.std():.4f}")
        sys.stdout.flush()

    avg_cv = np.mean(all_cv_means)
    std_cv = np.std(all_cv_means)
    avg_test_proba = np.mean(all_test_proba, axis=0)
    preds = (avg_test_proba >= 0.5).astype(int)
    n_survived = preds.sum()

    ref = baseline_cv if baseline_cv else 0.8409  # v10 -SVM baseline
    print(f"\n  {len(seeds)}-seed 평균 CV: {avg_cv:.4f} ±{std_cv:.4f}")
    print(f"  생존 예측 수: {n_survived}명")
    print(f"  기준({ref:.4f}) 대비: {avg_cv - ref:+.4f}")
    sys.stdout.flush()

    # 제출 파일 저장
    filename = f"submission_v11_{name.lower().replace(' ', '_').replace('+', '_')}.csv"
    out_path = SUBMISSIONS / filename
    pd.DataFrame({"PassengerId": test_ids, "Survived": preds}).to_csv(out_path, index=False)
    print(f"  저장: {filename}")
    sys.stdout.flush()

    return avg_cv, std_cv, n_survived


def main():
    train, test = load_raw()
    X_base, y_train, X_test_base, test_ids, _, _, train_eng, test_eng = build_features(train, test)
    skf_base = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # TicketPrefix 피처 추가
    def add_ticket_prefix(eng):
        eng = eng.copy()
        eng["TicketPrefix"] = eng["Ticket"].apply(extract_ticket_prefix)
        return eng

    train_eng_p = add_ticket_prefix(train_eng)
    test_eng_p = add_ticket_prefix(test_eng)

    # AgeGroup×Sex 피처 추가 (이미 AgeGroup, Sex 있음)
    train_eng_p["AgeSex"] = train_eng_p["AgeGroup"] + "_" + train_eng_p["Sex"]
    test_eng_p["AgeSex"] = test_eng_p["AgeGroup"] + "_" + test_eng_p["Sex"]

    results = {}

    # ── 5-seed 빠른 탐색 ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1: 5-seed 빠른 방향 탐색")
    print("="*60)
    sys.stdout.flush()

    # 기준: v10 -SVM (5-seed)
    cv, std, ns = run_experiment("v10_nosvm", [], train_eng_p, y_train, test_eng_p,
                                  skf_base, test_ids, seeds=SEEDS_QUICK)
    results["v10_nosvm"] = (cv, std, ns)
    baseline_cv5 = cv

    cv, std, ns = run_experiment("A_TicketPrefix", [("TicketPrefix", 3)],
                                  train_eng_p, y_train, test_eng_p, skf_base, test_ids,
                                  seeds=SEEDS_QUICK, baseline_cv=baseline_cv5)
    results["A_TicketPrefix"] = (cv, std, ns)

    cv, std, ns = run_experiment("B_AgeSex", [("AgeSex", 3)],
                                  train_eng_p, y_train, test_eng_p, skf_base, test_ids,
                                  seeds=SEEDS_QUICK, baseline_cv=baseline_cv5)
    results["B_AgeSex"] = (cv, std, ns)

    cv, std, ns = run_experiment("C_Both", [("TicketPrefix", 3), ("AgeSex", 3)],
                                  train_eng_p, y_train, test_eng_p, skf_base, test_ids,
                                  seeds=SEEDS_QUICK, baseline_cv=baseline_cv5)
    results["C_Both"] = (cv, std, ns)

    # ── 결과 요약 ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("v11 Phase1 결과 요약 (5-seed)")
    print(f"{'='*60}")
    print(f"{'실험':<20} | {'5-seed CV':>10} | {'분산':>8} | {'생존수':>6} | {'diff':>8}")
    print("-" * 60)
    for name_k, (cv_k, std_k, ns_k) in results.items():
        diff = cv_k - baseline_cv5 if name_k != "v10_nosvm" else 0.0
        star = " ★" if diff > 0.002 else (" △" if diff > 0 else "")
        print(f"{name_k:<20} | {cv_k:.4f}     | ±{std_k:.4f} | {ns_k:>6} | {diff:+.4f}{star}")
    sys.stdout.flush()

    # ── 유망 후보 10-seed 재검증 ───────────────────────────────────────────
    best = max(results.items(), key=lambda x: x[1][0])
    best_name, (best_cv, _, _) = best
    if best_name != "v10_nosvm" and best_cv - baseline_cv5 > 0.001:
        print(f"\n{'='*60}")
        print(f"PHASE 2: 10-seed 검증 — {best_name}")
        print(f"{'='*60}")
        sys.stdout.flush()
        extra = {"A_TicketPrefix": [("TicketPrefix", 3)],
                 "B_AgeSex": [("AgeSex", 3)],
                 "C_Both": [("TicketPrefix", 3), ("AgeSex", 3)]}
        cv10, std10, ns10 = run_experiment(
            f"{best_name}_10s",
            extra.get(best_name, []),
            train_eng_p, y_train, test_eng_p, skf_base, test_ids,
            seeds=SEEDS_FULL, baseline_cv=baseline_cv5
        )
        results[f"{best_name}_10s"] = (cv10, std10, ns10)
    else:
        print(f"\n유망 후보 없음 (5-seed diff ≤ 0.001). Phase 2 생략.")
        sys.stdout.flush()

    print(f"\n기준 (v10 Public 0.78708) 제출 가치: 10-seed CV +0.5%p 이상")


if __name__ == "__main__":
    main()
