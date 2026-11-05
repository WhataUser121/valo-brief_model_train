import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss
)

# 데이터 불러오기
df = pd.read_csv("../data/team_features.csv")

# 입력 Feature
X = df.drop(columns=[
    "match_id",
    "started_at",
    "target"
])

# 정답(Label)
y = df["target"]

# Train/Test 분리
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# XGBoost 모델 생성
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# 예측
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# 평가
acc = accuracy_score(y_test, pred)
auc = roc_auc_score(y_test, prob)
ll = log_loss(y_test, prob)
brier = brier_score_loss(y_test, prob)

print(f"Accuracy : {acc:.3f}")
print(f"ROC AUC  : {auc:.3f}")
print(f"LogLoss  : {ll:.3f}")
print(f"Brier    : {brier:.3f}")

# Feature Importance
importance = (
    pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    })
    .sort_values("importance", ascending=False)
)

print(importance.head(10))

# 실제 예측 결과 확인
result = pd.DataFrame({
    "actual": y_test.values,
    "blue_win_probability": prob,
    "prediction": pred
})

print(result.head(10))

# 모델 저장
joblib.dump(model, "../models/xgboost_valorant.pkl")

print("모델 저장 완료 : ../model/xgboost_valorant.pkl")

# Feature 이름 저장
# Feature 순서 꼬이는거 방지
joblib.dump(
    list(X.columns),
    "../models/feature_columns.pkl"
)

print("Feature 컬럼 저장 완료")

# # 저장된 모델 불러오기
# import joblib
# model = joblib.load("../model/xgboost_valorant.pkl")
#
# # 새로운 경기(Feature) 사용
# prob = model.predict_proba(new_match_df)[:, 1]
# print(f"Blue 승리 확률 : {prob[0]*100:.1f}%")