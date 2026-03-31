# Titanic ML — 내일 TODO (2026-03-20)

현재 최고 점수: **0.77511** (v4, submission_stacking_v4.csv)

---

## 🚀 제출 큐 (한도 리셋 후 즉시)

오늘 생성 완료, 내일 바로 제출 가능한 파일들:

- [ ] `submission_stacking_v5_optuna.csv` — v4 피처 + Optuna 60trials 스태킹 (meta-CV 83.72%)
- [ ] `submission_top7c_stacking.csv` — TOP7 피처 스태킹 (meta-CV 82.38% ± 0.85%, 생존율 39.5% ≈ 실제 38.4%) ⭐ 분산 낮음
- [ ] `submission_thresh_0p5700000000000002.csv` — v4 평균 앙상블, threshold=0.57 (훈련 정확도 83.84%)
- [ ] `submission_thresh_0p48.csv` — v4 평균 앙상블, threshold=0.48

```bash
export KAGGLE_API_TOKEN=KGAT_ab12eec1dd2ed1e13c07d7d4e060fb24

kaggle competitions submit -c titanic \
  -f data/submissions/submission_stacking_v5_optuna.csv \
  -m "v5_optuna: v4 features + Optuna(60 trials) stacking"

kaggle competitions submit -c titanic \
  -f "data/submissions/submission_thresh_0p5700000000000002.csv" \
  -m "v4 avg ensemble threshold=0.57"
```

---

## 🛠️ 코딩 작업

### 1순위 — OOF 가족 생존율 인코딩 (가장 높은 기대 효과)

- [ ] `src/features.py`에 `add_oof_survival_encoding()` 함수 추가
  - 성씨(Surname) + 성별 그룹별 OOF 생존율 → 새 피처
  - Ticket 그룹별 OOF 생존율 → 새 피처
  - **중요**: K-Fold OOF로 누수 방지, 테스트는 전체 훈련 데이터 기준
  - 참고: 아래 의사코드

```python
def add_oof_survival_encoding(X_raw, y, X_test_raw, group_col, skf):
    """
    훈련 데이터: K-Fold OOF 평균 인코딩 (누수 방지)
    테스트 데이터: 전체 훈련 데이터 기준 인코딩
    """
    global_mean = y.mean()
    oof_enc = np.full(len(X_raw), global_mean)
    for tr_idx, val_idx in skf.split(X_raw, y):
        group_map = pd.Series(y[tr_idx]).groupby(
            X_raw[group_col].iloc[tr_idx]
        ).mean()
        oof_enc[val_idx] = X_raw[group_col].iloc[val_idx].map(group_map).fillna(global_mean)
    # 테스트
    full_map = pd.Series(y).groupby(X_raw[group_col]).mean()
    test_enc = X_test_raw[group_col].map(full_map).fillna(global_mean)
    return oof_enc, test_enc
```

### 2순위 — LGBM 단독 (낮은 LR)

- [ ] LightGBM 단독 제출: `n_estimators=569, num_leaves=33, learning_rate=0.010`
  - Optuna 2회 공통: 낮은 LR + 많은 트리 = 안정적 성능
  - 스태킹 없이 LGBM 단독으로 v4 넘을 수 있는지 확인

### 3순위 — 다중 랜덤 시드 앙상블

- [ ] 동일 모델을 random_seed=42,7,21,99,123으로 5회 학습 후 평균
  - 분산 감소(Variance Reduction) 효과
  - v4 스태킹 위에 추가 레이어로 적용

### 4순위 (선택) — 신경망 MLP

- [ ] sklearn `MLPClassifier` or `torch` 간단한 MLP
  - hidden_layers=(128,64,32), dropout 추가
  - 앙상블에 포함

---

## 📊 분석 작업

- [ ] v4 vs gender_submission 예측 비교
  - 어떤 승객에서 우리 모델이 틀리는지 분석
  - 특히 3등석 여성 (많은 모델이 생존 예측하지만 실제로는 사망)
- [ ] 틀린 예측 케이스 수동 검토 (PassengerId 기준)

---

## 🔧 설정 작업

- [ ] `src/config.py`에 현재 best 파라미터 정리 (v4 기준으로 고정)
- [ ] `run_all.py` LGBM low-LR 실험 플래그 추가

---

## 📝 메모

### 실험에서 배운 것
| 변경 | 효과 | 이유 |
|---|---|---|
| TicketFreq 버그 수정 | +0.005 ✅ | train/test 분포 일치 |
| 새 피처 (IsChild 등) | -0.007 ❌ | 891개 소규모 → 노이즈 |
| 강한 정규화 | -0.004 ❌ | 원본 파라미터가 이미 최적 |
| Optuna 60 trials | -0.013 ❌ | CV 자체에 과적합 |

### 파라미터 현황 (v4 기준, 검증됨)
```python
XGB_PARAMS = dict(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42)
LGBM_PARAMS = dict(n_estimators=500, num_leaves=31, learning_rate=0.05,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5, random_state=42)
```

### Optuna 2회 공통 발견 (참고용)
- LGBM 최적 LR: 0.010~0.015 (현재 0.05보다 훨씬 낮음)
- LGBM 최적 leaves: 16~33 (현재 31과 겹침)
- XGB 최적 depth: 5 (현재 4보다 약간 깊음)
- → 하지만 Public Score에서는 원본이 더 좋았음. 소규모 데이터 CV 한계

### Titanic 이론적 점수 상한
- 단순 ML: ~0.79~0.81
- 이름 기반 실제 생존 기록 활용 (편법): ~0.85~0.87
- 목표: 0.79 돌파

---

> 마지막 업데이트: 2026-03-19
> 현재 최고: **0.77511** (v4)
> 다음 목표: **0.78+**
