
import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="환율 예측 AI", layout="wide")
st.title("💱 환율 예측 AI 시스템")

# 사용자 입력
mode = st.radio("예측 방식", ["Prophet 기반 예측", "시연용(한국은행 API 데이터)"])
start_date = st.date_input("예측 시작 날짜", datetime.today())
days = st.slider("예측 일 수", min_value=1, max_value=30, value=7)

API_KEY = "99BO6UEVOS1ZHTSHK79J"

@st.cache_data
def fetch_api_exchange():
    start = "20240101"
    end = datetime.today().strftime("%Y%m%d")
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/036Y001/DD/{start}/{end}/0002"
    try:
        res = requests.get(url)
        data = res.json()
        rows = data['StatisticSearch']['row']
        df = pd.DataFrame(rows)
        df = df[['TIME', 'DATA_VALUE']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        return df
    except Exception as e:
        st.error(f"API 데이터 불러오기 실패: {e}")
        return None

if mode == "Prophet 기반 예측":
    try:
        df = pd.read_csv("data/exchange_rate.csv")
        df.columns = ['ds', 'y']
    except Exception as e:
        st.error(f"CSV 로딩 실패: {e}")
        st.stop()
else:
    df = fetch_api_exchange()
    if df is None:
        st.stop()

# 모델 학습 및 예측
try:
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].tail(days)
    result.columns = ['날짜', '예측 환율 (KRW/USD)']
    st.line_chart(result.set_index("날짜"))
    st.dataframe(result)
except Exception as e:
    st.error(f"예측 실패: {e}")
