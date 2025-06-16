import streamlit as st
import pandas as pd
from prophet import Prophet
from datetime import datetime
import requests

st.set_page_config(page_title="환율 예측 AI", layout="wide")
st.title("💱 환율 예측 AI (ECOS API 연동)")

# --- 설정 파트 ---
API_KEY = "99BO6UEVOS1ZHTSHK79J"
LANG = "kr"
START_IDX = 1
END_IDX = 1000
TABLE = "731Y001"        # 외환시장 매매기준율
FREQ = "DD"
ITEM = "0000001"         # USD 항목 코드

# 조회 가능 날짜 예시
DEFAULT_START = "20240101"
DEFAULT_END = "20240105"

# 사용자 시작일/종료일 입력
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("시작일", datetime.strptime(DEFAULT_START, "%Y%m%d").date())
with col2:
    end_date = st.date_input("종료일", datetime.strptime(DEFAULT_END, "%Y%m%d").date())

# 유효 범위 검사
if start_date > end_date:
    st.error("시작일이 종료일보다 앞서야 합니다.") 
    st.stop()
if (end_date - start_date).days > 365:
    st.warning("1년 이하 기간만 권장됩니다.")

@st.cache_data
def fetch_exchange(sdt, edt):
    s = sdt.strftime("%Y%m%d")
    e = edt.strftime("%Y%m%d")
    url = (f"http://ecos.bok.or.kr/api/StatisticSearch/"
           f"{API_KEY}/json/{LANG}/{START_IDX}/{END_IDX}/"
           f"{TABLE}/{FREQ}/{s}/{e}/{ITEM}")
    st.write("🔗 요청 URL:", url)
    resp = requests.get(url)
    try:
        data = resp.json()
    except ValueError:
        return None, "JSON 파싱 오류", resp.text

    if "StatisticSearch" not in data:
        # 에러 메시지 포함 응답
        errmsg = data.get("RESULT", {}).get("MESSAGE", "알 수 없는 오류")
        return None, data.get("RESULT", {}).get("CODE", "ERROR"), errmsg

    rows = data["StatisticSearch"].get("row")
    if not rows:
        return None, "EMPTY", "조회 결과 없음"

    df = pd.DataFrame(rows)
    df = df.rename(columns={"TIME":"ds", "DATA_VALUE":"y"})
    df["ds"] = pd.to_datetime(df["ds"], format="%Y%m%d")
    df["y"] = df["y"].astype(float)
    return df, "OK", None

# --- API 호출 ---
df, code, errmsg = fetch_exchange(start_date, end_date)

if code != "OK":
    st.error(f"API 오류 ({code}): {errmsg}")
    st.stop()

# --- Prophet 예측 ---
days = st.slider("예측일 수", 1, 30, 7)

model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=days)
forecast = model.predict(future)
res = forecast[['ds', 'yhat']].tail(days).rename(columns={'ds':'날짜','yhat':'환율(예측)'})

st.line_chart(res.set_index("날짜"))
st.dataframe(res)
