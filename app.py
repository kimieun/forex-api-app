import streamlit as st
import pandas as pd
from prophet import Prophet

st.set_page_config(page_title="환율 예측 AI (공공데이터)", layout="wide")
st.title("💱 환율 예측 AI (공공데이터포털 기반)")

# ✅ CSV 직접 URL 또는 다운로드 후 프로젝트 내부 경로
CSV_URL = "https://www.data.go.kr/dataset/15105540/fileData.do"  # 실제 CSV 주소 필요

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df = df.rename(columns={"Date":"ds", "Dollar":"y"})
    df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m-%d")
    df["y"] = df["y"].astype(float)
    return df

df = load_data(CSV_URL)

# ✅ 사용자 인터페이스
st.write(f"📅 데이터 범위: {df.ds.min().date()} ~ {df.ds.max().date()}")
days = st.slider("예측할 일 수", 1, 30, 7)

# ✅ Prophet 예측
try:
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    result = forecast[['ds','yhat']].tail(days).rename(columns={'ds':'날짜','yhat':'예측 환율'})
    st.line_chart(result.set_index("날짜"))
    st.dataframe(result)
except Exception as e:
    st.error(f"예측 실패: {e}")
