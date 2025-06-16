import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime

st.set_page_config(page_title="환율 예측 AI (이중 기반)", layout="wide")
st.title("💱 환율 예측 AI 시스템 - 과거 vs 최신")

# 데이터 선택 UI
mode = st.radio("예측에 사용할 데이터", ["2024년 과거 기반 예측", "2025년 최신 기반 예측"])
days = st.slider("예측할 일 수", 1, 30, 7)

# 데이터 불러오기
if mode == "2024년 과거 기반 예측":
    df = pd.read_csv("data/exchange_rate_2024.csv")
else:
    df = pd.read_csv("data/exchange_rate_2025.csv")

# 날짜 형식 통일
df["ds"] = pd.to_datetime(df["ds"])
df["y"] = df["y"].astype(float)

# 예측 시작일 결정 (마지막 날짜 기준 미래로 예측)
start_forecast = df["ds"].max() + pd.Timedelta(days=1)

# Prophet 모델 학습 및 예측
try:
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    future = future[future["ds"] >= start_forecast]  # 미래 날짜만 예측에 사용
    forecast = model.predict(future)
    result = forecast[["ds", "yhat"]].rename(columns={"ds": "날짜", "yhat": "예측 환율 (KRW/USD)"})

    st.subheader(f"📊 {mode} 결과")
    st.line_chart(result.set_index("날짜"))
    st.dataframe(result)
except Exception as e:
    st.error(f"예측 실패: {e}")
