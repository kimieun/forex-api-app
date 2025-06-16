import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
import requests

st.set_page_config(page_title="환율 예측 AI", layout="wide")
st.title("💱 환율 예측 AI 시스템")

# 종료일 고정 (한국은행 API 마지막 날짜)
API_END_DATE = "20250613"
end_dt = datetime.strptime(API_END_DATE, "%Y%m%d")

# 시작일 기본값: 오늘과 종료일 중 더 이른 날짜
default_start = min(datetime.today(), end_dt).date()

# 사용자 입력 - 예측 시작일 (종료일보다 이후로 못 넘김)
start_date = st.date_input(
    "예측 시작 날짜",
    default_start,
    max_value=end_dt.date()
)

# 예측 일수 슬라이더
days = st.slider("예측 일 수", min_value=1, max_value=30, value=7)

# 예측 방식 선택
mode = st.radio("예측 방식", ["Prophet 기반 예측", "시연용(한국은행 API 데이터)"])

# 시작일 유효성 검사 (이론상 막혀 있지만 추가 안전망)
if start_date > end_dt.date():
    st.error(f"예측 시작일은 종료일({API_END_DATE[:4]}-{API_END_DATE[4:6]}-{API_END_DATE[6:]})보다 이전이어야 합니다.")
    st.stop()

API_KEY = "99BO6UEVOS1ZHTSHK79J"

@st.cache_data
def fetch_api_exchange(user_start_date):
    start = user_start_date.strftime("%Y%m%d")
    end = API_END_DATE
    url = f"http://ecos.bok.or.kr/api/StatisticSearch/{API_KEY}/json/kr/1/1000/036Y001/DD/{start}/{end}/0002"

    st.write("📡 요청 URL:", url)

    try:
        res = requests.get(url)

        try:
            data = res.json()
            st.write("📥 API 응답 (JSON):", data)
        except ValueError:
            st.write("📥 API 응답 (텍스트):", res.text)
            st.error("❌ JSON 형식 아님. API 응답 파싱 실패.")
            return None

        if 'StatisticSearch' not in data:
            msg = data.get("RESULT", {}).get("MESSAGE", "알 수 없는 오류")
            code = data.get("RESULT", {}).get("CODE", "N/A")
            st.error(f"❌ API 오류 (코드: {code}) → {msg}")
            return None

        rows = data['StatisticSearch']['row']
        df = pd.DataFrame(rows)
        df = df[['TIME', 'DATA_VALUE']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = df['y'].astype(float)
        return df

    except Exception as e:
        st.error(f"API 요청 실패: {e}")
        return None

# 데이터 불러오기
if mode == "Prophet 기반 예측":
    try:
        df = pd.read_csv("data/exchange_rate.csv")
        df.columns = ['ds', 'y']
    except Exception as e:
        st.error(f"CSV 로딩 실패: {e}")
        st.stop()
else:
    df = fetch_api_exchange(start_date)
    if df is None:
        st.stop()

# 예측 및 시각화
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
