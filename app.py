import streamlit as st
import pandas as pd
from prophet import Prophet

st.set_page_config(page_title="환율 예측 AI (공공데이터 기반)", layout="wide")
st.title("💱 환율 예측 AI (샘플 CSV 기반)")

CSV_URL = "https://raw.githubusercontent.com/kimieun/forex-api-app/main/data/sample_exchange_rate.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = df["y"].astype(float)
    return df

df = load_data(CSV_URL)

st.write(f"📅 데이터 범위: {df.ds.min().date()} ~ {df.ds.max().date()}")
days = st.slider("예측할 일 수", 1, 30, 7)

model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)
result = forecast[["ds", "yhat"]].tail(days).rename(columns={"ds": "날짜", "yhat": "예측 환율"})

st.line_chart(result.set_index("날짜"))
st.dataframe(result)
