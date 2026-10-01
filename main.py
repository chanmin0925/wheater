import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용해 장기적인 기온 추세를 확인합니다.")


# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜를 날짜형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 필요한 데이터만 남김
    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()


# -----------------------------
# 2025년까지 + 관측일 300일 이상인 연도만 사용
# -----------------------------
yearly_count = df.groupby("연도")["날짜"].nunique()

valid_years = yearly_count[
    (yearly_count >= 300) &
    (yearly_count.index <= 2025)
].index

yearly_temp = (
    df[df["연도"].isin(valid_years)]
    .groupby("연도", as_index=False)["평균기온"]
    .mean()
)

yearly_temp = yearly_temp.sort_values("연도")


# -----------------------------
# 회귀분석
# 독립변수 = 1908년부터 지난 연수
# -----------------------------
yearly_temp["경과연수"] = yearly_temp["연도"] - 1908

x = yearly_temp["경과연수"].to_numpy()
y = yearly_temp["평균기온"].to_numpy()

# 선형회귀
slope, intercept = np.polyfit(x, y, 1)

# 회귀선 예측값
yearly_temp["회귀기온"] = intercept + slope * yearly_temp["경과연수"]

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# -----------------------------
# 회귀선 전체 계산 범위
# 1900 ~ 2100
# -----------------------------
prediction_years = np.arange(1900, 2101)
prediction_elapsed = prediction_years - 1908

prediction_temp = intercept + slope * prediction_elapsed


# -----------------------------
# 선택 연도 슬라이더
# -----------------------------
selected_year = st.slider(
    "예상할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1,
    format="%d년"
)

selected_elapsed = selected_year - 1908
selected_temperature = intercept + slope * selected_elapsed


# -----------------------------
# 선택 연도의 예상 기온 크게 표시
# -----------------------------
st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{selected_temperature:.2f} °C"
)


# -----------------------------
# 회귀 정보
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "회귀에 사용한 연도 수",
        f"{len(yearly_temp)}개"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{int(yearly_temp['연도'].min())}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{int(yearly_temp['연도'].max())}년"
    )

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )


# -----------------------------
# 산점도 + 회귀선
# -----------------------------
fig = go.Figure()

# 실제 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly_temp["연도"],
        y=yearly_temp["평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            size=7
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "평균기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 회귀선
fig.add_trace(
    go.Scatter(
        x=prediction_years,
        y=prediction_temp,
        mode="lines",
        name="회귀선",
        line=dict(
            width=3
        ),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 예상기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 선택한 연도 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[selected_temperature],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(
            size=14,
            symbol="diamond"
        ),
        hovertemplate=(
            f"연도: {selected_year}년<br>"
            f"예상기온: {selected_temperature:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 회귀선",
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    xaxis=dict(
        type="linear",
        dtick=10
    ),
    hovermode="closest",
    height=600
)

st.plotly_chart(
    fig,
    width="stretch"
)


# -----------------------------
# 회귀식
# -----------------------------
st.subheader("회귀식")

st.latex(
    rf"\hat{{y}} = {intercept:.4f} + {slope:.4f}x"
)

st.write(
    "여기서 x는 1908년부터 지난 연수입니다. "
    "예를 들어 2025년의 x는 117입니다."
)


# -----------------------------
# 데이터 처리 기준 안내
# -----------------------------
st.subheader("분석에 사용한 데이터 기준")

st.write(
    f"- 2025년까지의 데이터만 사용했습니다."
)
st.write(
    f"- 한 해의 관측일이 300일 미만인 연도는 제외했습니다."
)
st.write(
    f"- 회귀에 사용된 기간: "
    f"{int(yearly_temp['연도'].min())}년 ~ "
    f"{int(yearly_temp['연도'].max())}년"
)
st.write(
    f"- 회귀에 사용된 연도 수: {len(yearly_temp)}개"
)
