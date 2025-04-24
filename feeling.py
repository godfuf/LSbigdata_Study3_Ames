import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)

######## 전처리 과정 ########
# 데이터 불러오기
ames = pd.read_csv('ames.csv')
df = ames.copy()

# 그룹 분류 
df_ns = df.groupby('Neighborhood')['SalePrice'].mean()

# 분위수 계산
q1 = df_ns.quantile(0.25)
q2 = df_ns.quantile(0.75)

#df['price_level'] 이라는 칼럼으로 집값 별 분위로 구분
df['price_level'] = np.select(
    [
        df['Neighborhood'].isin(df_ns[df_ns <= q1].index),
        df['Neighborhood'].isin(df_ns[(df_ns > q1) & (df_ns <= q2)].index),
        df['Neighborhood'].isin(df_ns[df_ns > q2].index)
    ],
    ['Low', 'Mid', 'High'],
    default='Unknown'  #  문자열로 통일
)

# 범주형 컬럼만 선택
cat_cols = df.select_dtypes(include='object').columns
# 범주형 결측치 'None'으로 대체
df[cat_cols] = df[cat_cols].fillna('None')


df['amenities'] = (
    (df['PoolArea'] > 0).astype(int) +
    (df['TotalBsmtSF'] > 0).astype(int) +
    (df['GarageArea'] > 0).astype(int) + 
    (df['MiscVal'] > 0).astype(int)
)

df['TotalRooms'] = df['TotRmsAbvGrd'] + df['HalfBath'] + df['FullBath']  # 욕실 제외 방수 + 반욕실 + 풀욕실
df['RoomDensity'] = df['TotalRooms'] / df['GrLivArea']  # 방 밀도 (방수 / 거실 면적)



##### 허위매물 판단 #####


## 조건 : 기준값 상위 25% 이상
df['GrLivArea_th'] = (df['GrLivArea'] / df.groupby('price_level')['GrLivArea'].transform(lambda x: x.quantile(0.75))
)

df['YearRemodAdd_th'] = (df['YearRemodAdd'] / df.groupby('price_level')['YearRemodAdd'].transform(lambda x: x.quantile(0.75))
)

df['RoomDensity_th'] = (df['RoomDensity'] / df.groupby('price_level')['RoomDensity'].transform(lambda x: x.quantile(0.75))
)

# 지역들 분리하는 df 생성
high_df = df[df['price_level'] == 'High'].copy()
mid_df  = df[df['price_level'] == 'Mid'].copy()
low_df  = df[df['price_level'] == 'Low'].copy()
# --------------------------------------------------------------------------------------------
# 그룹별 시각화 : 박스플롯
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 서브플롯 생성 (가로로 3개)
fig1 = make_subplots(rows=1, cols=3, subplot_titles=("GrLivArea by Price Level", 
                                                    "YearRemodAdd by Price Level", 
                                                    "RoomDensity by Price Level"))

# GrLivArea
for level in df['price_level'].unique():
    fig1.add_trace(
        go.Box(y=df[df['price_level'] == level]['GrLivArea'],
               name=level,
               boxmean=True),
        row=1, col=1
    )
# YearRemodAdd
for level in df['price_level'].unique():
    fig1.add_trace(
        go.Box(y=df[df['price_level'] == level]['YearRemodAdd'],
               name=level,
               boxmean=True),
        row=1, col=2
    )
# RoomDensity
for level in df['price_level'].unique():
    fig1.add_trace(
        go.Box(y=df[df['price_level'] == level]['RoomDensity'],
               name=level,
               boxmean=True),
        row=1, col=3
    )

# 전체 레이아웃 설정
fig1.update_layout(
    height=500, width=1200,
    title_text="박스플롯으로 본 연속형 변수와 Price Level 관계",
    showlegend=False
)

fig1.show()
# -----------------------------------------------------------------------------------

# 그룹별 시각화2 : 히스토그램  

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 서브플롯 생성: 3행 3열
fig2 = make_subplots(
    rows=3, cols=3,
    subplot_titles=[
        'OverallQual - Low', 'OverallQual - Mid', 'OverallQual - High',
        'OverallCond - Low', 'OverallCond - Mid', 'OverallCond - High',
        'Amenities - Low', 'Amenities - Mid', 'Amenities - High'
    ]
)
# 변수 리스트 및 시각화 세팅
variables = ['OverallQual', 'OverallCond', 'amenities']
colors = ['skyblue', 'salmon', 'lightgreen']
bins_dict = {
    'OverallQual': list(range(1, 11)),
    'OverallCond': list(range(1, 11)),
    'amenities': list(range(0, 6))
}
# 그래프 추가
for row, var in enumerate(variables, start=1):
    for col, level in enumerate(['Low', 'Mid', 'High'], start=1):
        subset = df[df['price_level'] == level]
        fig2.add_trace(
            go.Histogram(
                x=subset[var],
                xbins=dict(
                    start=min(bins_dict[var]),
                    end=max(bins_dict[var]),
                    size=1
                ),
                marker_color=colors[row-1],
                name=f'{var} - {level}',
                showlegend=False
            ),
            row=row, col=col
        )
# 전체 레이아웃 조정
fig2.update_layout(
    height=900, width=1000,
    title_text="Price Level 별 히스토그램 (OverallQual, OverallCond, Amenities)",
    bargap=0.1
)

fig2.show()
# ------------------------------------------------------------------------------------

############################# 분석 진행 / 조건 플래그 ##############################

# High 그룹
high_med      = high_df['SalePrice'].median()
high_area_th  = high_df['GrLivArea'].quantile(0.75)
high_remod_th = high_df['YearRemodAdd'].quantile(0.75)
high_den_th   = high_df['RoomDensity'].quantile(0.75)

high_df['flag_high_qual']      = (high_df['OverallQual']  >= 9).astype(int)
high_df['flag_good_condition'] = (high_df['OverallCond']  >= 6).astype(int)
high_df['flag_high_area']      = (high_df['GrLivArea']    >= high_area_th ).astype(int)
high_df['flag_high_remod']     = (high_df['YearRemodAdd'] >= high_remod_th).astype(int)
high_df['flag_high_density']   = (high_df['RoomDensity']  >= high_den_th  ).astype(int)
high_df['flag_high_amenities'] = (high_df['amenities']    >= 3            ).astype(int)

# Mid 그룹
mid_med      = mid_df['SalePrice'].median()
mid_area_th  = mid_df['GrLivArea'].quantile(0.75)
mid_remod_th = mid_df['YearRemodAdd'].quantile(0.75)
mid_den_th   = mid_df['RoomDensity'].quantile(0.75)

mid_df['flag_mid_qual']        = (mid_df['OverallQual']  >= 8).astype(int)
mid_df['flag_good_condition']  = (mid_df['OverallCond']  >= 6).astype(int)
mid_df['flag_mid_area']        = (mid_df['GrLivArea']    >= mid_area_th ).astype(int)
mid_df['flag_mid_remod']       = (mid_df['YearRemodAdd'] >= mid_remod_th).astype(int)
mid_df['flag_mid_density']     = (mid_df['RoomDensity']  >= mid_den_th  ).astype(int)
mid_df['flag_mid_amenities']   = (mid_df['amenities']    >= 3            ).astype(int)

# Low 그룹
low_med      = low_df['SalePrice'].median()
low_area_th  = low_df['GrLivArea'].quantile(0.75)
low_remod_th = low_df['YearRemodAdd'].quantile(0.75)
low_den_th   = low_df['RoomDensity'].quantile(0.75)

low_df['flag_low_qual']        = (low_df['OverallQual']  >= 7).astype(int)
low_df['flag_good_condition']  = (low_df['OverallCond']  >= 8).astype(int)
low_df['flag_low_area']        = (low_df['GrLivArea']    >= low_area_th ).astype(int)
low_df['flag_low_remod']       = (low_df['YearRemodAdd'] >= low_remod_th).astype(int)
low_df['flag_low_density']     = (low_df['RoomDensity']  >= low_den_th  ).astype(int)
low_df['flag_low_amenities']   = (low_df['amenities']    >= 3            ).astype(int)


for name, gdf, med in [
    ('High', high_df, high_med),
    ('Mid',  mid_df,  mid_med),
    ('Low',  low_df,  low_med),
]:
    # Median 이하인 매물만 복사본으로
    filt = gdf.loc[gdf['SalePrice'] <= med].copy()
    
    # 이 그룹의 flag 컬럼 리스트
    flags = [c for c in filt.columns 
             if c.startswith(f'flag_{name.lower()}') or c == 'flag_good_condition']
    
    print(f"\n— {name} 그룹 (SalePrice ≤ median={med:.0f}) —")
    
    # 1) 플래그별 1값 개수
    print("플래그별 1값 개수:")
    print(filt[flags].sum(), "\n")
    
    # 2) score 분포 & 3점 이상 건수
    filt['score'] = filt[flags].sum(axis=1)
    print("score 분포:")
    print(filt['score'].value_counts().sort_index(), "\n")
    print(f"score ≥ 3인 건수: { (filt['score'] >= 3).sum() }건")

# (1) score 계산: 6가지 조건을 한 줄로 집계
qual_th = {'Low':7,'Mid':8,'High':9}
cond_th = {'Low':8,'Mid':6,'High':6}

df['score'] = df.apply(lambda r: 
    int(r['OverallQual']  >= qual_th[r['price_level']]) +
    int(r['OverallCond']  >= cond_th [r['price_level']]) +
    int(r['GrLivArea']    >= r['GrLivArea_th']) +
    int(r['YearRemodAdd'] >= r['YearRemodAdd_th']) +
    int(r['RoomDensity']  >= r['RoomDensity_th']) +
    int(r['amenities']    >= 3),
    axis=1
)

# (2) 그룹별 중위값 이하 여부
median_price = df.groupby('price_level')['SalePrice'].transform('median')

# (3) suspect_flag 생성
df['suspect_flag'] = (df['SalePrice'] <= median_price) & (df['score'] >= 3)

# (4) 허위매물 후보만 추출
suspect_df = df[df['suspect_flag']].copy()

# (5) 결과 출력: 모든 컬럼 포함
print(f"허위매물 후보 총 {len(suspect_df)}건")

suspect_df
# -----------------------------------------------------------------------------------------
#### 지도시각화 ####

import plotly.express as px

# 지도 중심을 데이터의 평균 위도·경도로 설정
center = {
    "lat": df["Latitude"].mean(),
    "lon": df["Longitude"].mean()
}

fig3 = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    color="price_level",
    hover_name="Neighborhood",
    hover_data=["SalePrice", "GrLivArea"],
    zoom=11.5,                    
    center=center,              # 지도 중심 좌표
    height=600,
    mapbox_style="open-street-map",
    title="Ames Housing: Price Level by Neighborhood (확대)"
)

fig3.show()
# ---------------------------------------------------------------------------------------
################# 회귀모델 #################

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# 사용할 6개 피처와 타겟 정의
features = [
    'OverallQual',
    'OverallCond',
    'GrLivArea',
    'YearRemodAdd',
    'RoomDensity',
    'amenities'
]
target = 'SalePrice'

# price_level별 모델 학습, 검증, 예측, 시각화
for level in ['Low', 'Mid', 'High']:
    # 1) 해당 그룹 데이터 분리
    df_lvl = df[df['price_level'] == level].copy()
    X = df_lvl[features]
    y = df_lvl[target]
    
    # 2) hold-out test set 생성 (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3) RidgeCV 모델 학습 (내부 5-fold CV 포함)
    # ElasticNet과 달리 Ridge는 l1_ratio가 없고 alpha만 튜닝합니다
    ridge = RidgeCV(
        alphas=np.logspace(-4, 1, 10),  # 다양한 alpha 값 검사 (10개)
        cv=5,                           # 5-fold 교차검증
        scoring='neg_mean_squared_error' # MSE를 최소화하는 alpha 선택
    )
    ridge.fit(X_train, y_train)
    
    # 4) Test set 성능 평가
    y_test_pred = ridge.predict(X_test)
    r2   = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"{level} 그룹 → Test R² = {r2:.3f}, RMSE = {rmse:,.0f}")
    print(f"    최적 alpha = {ridge.alpha_:.6f}")
    
    # 모델의 계수 확인 (Ridge는 모든 변수의 계수를 유지함)
    coefficients = pd.DataFrame({
        'Feature': features,
        'Coefficient': ridge.coef_
    }).sort_values('Coefficient', ascending=False)
    print("Ridge 계수:")
    print(coefficients)
    print("\n" + "-"*50 + "\n")
    
    # 5) 전체 그룹 데이터에 대해 예측 및 residual 계산
    df_lvl['predicted'] = ridge.predict(X)
    df_lvl['residual']  = df_lvl['SalePrice'] - df_lvl['predicted']
    
    # 6) 이상치(허위매물) 플래그: residual 하위 25% 이하면 True
    thresh = df_lvl['residual'].quantile(0.026)
    df_lvl['ridge_flag'] = df_lvl['residual'] <= thresh
    # -------------------------------------------------------------------------------------------------------
    # 7) 인터랙티브 산점도
    fig4 = px.scatter(
        df_lvl,
        x='SalePrice',
        y='predicted',
        color='ridge_flag',
        color_discrete_map={True: 'red', False: 'lightgray'},
        title=f'Actual vs. Predicted ({level}) - Ridge Regression',
        labels={'SalePrice':'실제가격','predicted':'예측가격','ridge_flag':'허위매물 의심'},
        opacity=0.7
    )
    # y = x 대각선 추가 (완벽한 예측 참조선)
    mn, mx = df_lvl[['SalePrice','predicted']].min().min(), df_lvl[['SalePrice','predicted']].max().max()
    fig4.add_shape(
        type='line', x0=mn, y0=mn, x1=mx, y1=mx,
        line=dict(color='black', dash='dash')
    )
    fig4.update_layout(width=600, height=600)
    fig4.show()
    
# -------------------------------------------------------------------------------------------------------------
# 허위매물 비교
# (1) 사용할 피처·타겟 재확인
features = ['OverallQual', 'OverallCond', 'GrLivArea', 'YearRemodAdd', 'RoomDensity', 'amenities']
target   = 'SalePrice'

# (2) 플래그 컬럼 초기화
df['elastic_flag'] = False
df['ridge_flag']   = False

# (3) RidgeCV로 전체 데이터에 대해 플래그 계산
for level in ['Low','Mid','High']:
    mask = df['price_level']==level
    X = df.loc[mask, features]
    y = df.loc[mask, target]
    ridge = RidgeCV(
        alphas=np.logspace(-4, 1, 10),
        cv=5,
        scoring='neg_mean_squared_error'
    )
    ridge.fit(X, y)
    preds = ridge.predict(X)
    resid = y - preds
    thresh = resid.quantile(0.026)
    df.loc[mask, 'ridge_flag'] = resid <= thresh

# (4) 인덱스 집합으로 변환
score_set   = set(df.index[df['suspect_flag']])
elastic_set = set(df.index[df['elastic_flag']])
ridge_set   = set(df.index[df['ridge_flag']])

# (5) 개수 요약 출력
print("=== 허위매물 건수 비교 ===")
print(f"점수제(suspect_flag) : {len(score_set)}건")
print(f"Ridge   기준      : {len(ridge_set)}건\n")
print("=== 교집합 건수 ===")
print(f"점수&ElasticNet 공통 허위매물     : {len(score_set & elastic_set)}건")


# (7) 각 그룹별 예시 뽑아서 보기
print(">>> 두 방법 모두 의심한 매물 (공통)")
df.loc[list(score_set & ridge_set)]

print(">>> 오직 점수제만 의심한 매물")
df.loc[list(score_set - ridge_set)]

print(">>> 오직 Ridge만 의심한 매물")
df.loc[list(ridge_set - score_set)]
# -------------------------------------------------------------------------------
import folium

# 공통 허위매물만 추출 (점수제 + Ridge 회귀 모두 해당)
common_df = df.loc[list(score_set & ridge_set)].copy()
common_df1 = df.loc[list(score_set & ridge_set)].copy()


# 좌표 결측치 제거
common_df = common_df.dropna(subset=["Latitude", "Longitude"])

# 지도 중심 좌표 설정
center_lat = common_df["Latitude"].mean()
center_lon = common_df["Longitude"].mean()

# folium 지도 객체 생성
fig5 = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# price_level별 색상 지정
color_map = {
    'Low': 'blue',
    'Mid': 'green',
    'High': 'red'
}

# 마커 추가 (자동 출력 억제)
for _, row in common_df.iterrows():
    popup_text = f"""
    <b>지역:</b> {row['Neighborhood']}<br>
    <b>매매가:</b> ${row['SalePrice']:,}<br>
    <b>지상면적:</b> {row['GrLivArea']} sqft<br>
    <b>마감 품질:</b> {row['OverallQual']}<br>
    <b>전반 상태:</b> {row['OverallCond']}<br>
    <b>편의시설 수:</b> {row['amenities']}<br>
    <b>리모델링 연도:</b> {row['YearRemodAdd']}<br>
    <b>점수제 점수:</b> {row['score']}
    """
    _ = folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_text, max_width=250),
        icon=folium.Icon(
            color=color_map.get(row['price_level'], 'gray'),
            icon='exclamation-sign',
            prefix='glyphicon'
        )
    ).add_to(fig5)

# 지도 출력
fig5

# -------------------------------------------------------------------------------------


import plotly.express as px


# 색상 맵핑
color_map = {
    'High': '#e41a1c',   # 빨강
    'Mid': '#ff7f00',    # 주황
    'Low': '#4daf4a'     # 초록
}

# 지도 시각화
fig_map = px.scatter_mapbox(
    df,
    lat='Latitude',
    lon='Longitude',
    color='price_level',
    color_discrete_map=color_map,
    hover_name='Neighborhood',
    hover_data={
        'SalePrice': True,
        'price_level': True,
        'Latitude': False,
        'Longitude': False
    },
    zoom=12,
    height=600
)

# 스타일 및 레이아웃 설정
fig_map.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": 42.03, "lon": -93.63},
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    legend=dict(
        title='Price Level',
        x=0.97, y=0.92,  # 살짝 아래로
        xanchor='right', yanchor='top',
        bgcolor='rgba(255, 255, 255, 0.6)',  # 반투명한 배경
        bordercolor='lightgray',
        borderwidth=1
    ),
    font=dict(size=14, color='black'),
)

fig_map.show()





df['Neighborhood'].value_counts()
df['Neighborhood'].describe()
