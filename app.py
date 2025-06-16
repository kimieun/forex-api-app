import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
import requests

st.set_page_config(page_title="환율 예측 AI", layout="wide")
st.title("💱 환율 예측 AI 시스템")

# 종료일 고정 (한국은행 데이터 제공 범위 마지막 날짜)
API_END_DATE = "20250613"
end_dt = datetime.strptime(API_END_DATE, "%Y%m%d")

# 기본 시작일: 오늘과 종료일 중 더 이른 날짜 선택
default_start = min(datetime.today(), end_dt).date()

# 사용자 입력 - 시작일 제한
start_date = st.date_input(
    "예측 시작 날짜",
    default_start,
    max_value=end_dt.date()  # 날짜 제한 적용
)

days = st.slider("예측 일 수", min_value=1, max_value=30, value=7)
mode = st.radio("예측 방식", ["Prophet 기반 예측", "시연용(한국은행 API 데이터)"])

# 종료일 이후 날짜 선택 시 중단
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
            msg = data.get("RESULT", {}
