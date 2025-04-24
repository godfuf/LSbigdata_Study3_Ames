import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)


# 데이터 불러오기
ames = pd.read_csv('C:/Users/USER/Documents/lsbigdata-gen4/1joameshouse/group1_project/ames.csv')

ames['Neighborhood'].unique()

ames.info()

# 복사 및 전처리
df = ames.copy()

# 결측치가 1개 이상 있는 컬럼만 출력
# null_cols = df.columns[df.isnull().any()]
# df[null_cols].isnull().sum().sort_values(ascending=False)


## 1. 구역들을 고가/저가/중간 3개 그룹으로 나누기
df_ns = df.groupby('Neighborhood')['SalePrice'].mean()

# 분위수 계산
q1 = df_ns.quantile(0.25)
q2 = df_ns.quantile(0.75)

df['price_level'] = np.select(
    [
        df['Neighborhood'].isin(df_ns[df_ns <= q1].index),
        df['Neighborhood'].isin(df_ns[(df_ns > q1) & (df_ns <= q2)].index),
        df['Neighborhood'].isin(df_ns[df_ns > q2].index)
    ],
    ['Low', 'Mid', 'High'],
    default=np.nan
)
df
### 시각화



## 2. 모든 범주형 데이터 결측치 >> None 추가

# 범주형 컬럼만 선택
cat_cols = df.select_dtypes(include='object').columns

# 범주형 결측치 'None'으로 대체
df[cat_cols] = df[cat_cols].fillna('None')

# 어매니티 수영장 차고 지하실 전체면적(?) , 추가적인 부동산 특징
df['amenities'] = (
    (df['PoolArea'] > 0).astype(int) +
    (df['TotalBsmtSF'] > 0).astype(int) +
    (df['GarageArea'] > 0).astype(int) + 
    (df['MiscVal'] > 0).astype(int)
)

df['TotalRooms'] = df['TotRmsAbvGrd'] + df['HalfBath'] + df['FullBath']  # 욕실 제외 방수 + 반욕실 + 풀욕실
df['RoomDensity'] = df['TotalRooms'] / df['GrLivArea']  # 방 밀도 (방수 / 거실 면적)


# # 1) 각 그룹별 중위값을 계산해 새로운 컬럼에 저장
# df['SalePrice_median'] = df.groupby('price_level')['SalePrice'].transform('median')

# # 2) 그룹별 중위값 이하인 매물만 필터링
# df_half = df[df['SalePrice'] <= df['SalePrice_median']]


''''''''''''''''''''''''''''''''''''

# import matplotlib.pyplot as plt

# cols   = ['GrLivArea', 'YearRemodAdd']
# levels = ['Low', 'Mid', 'High']

# for col in cols:
#     for level in levels:
#         data = df[df['price_level'] == level][col]
        
#         # fig, ax 객체를 사용
#         fig, ax = plt.subplots(figsize=(8, 3))
#         ax.boxplot(data, vert=False, patch_artist=True,
#                    boxprops=dict(edgecolor='black'))
        
#         ax.set_title(f'{level} 그룹 — {col} 분포 (박스플롯)')
#         ax.set_xlabel(col)
#         plt.tight_layout()
#         plt.show()



''''''''''''''''''''''''''''''''''''


## 3. 허위매물 판단 조건 설정 - 기준값 상위 25% 이상
df['GrLivArea_th']    = df.groupby('price_level')['GrLivArea']   \
                           .transform(lambda x: x.quantile(0.75))
df['YearRemodAdd_th'] = df.groupby('price_level')['YearRemodAdd']\
                           .transform(lambda x: x.quantile(0.75))
df['RoomDensity_th']  = df.groupby('price_level')['RoomDensity'] \
                           .transform(lambda x: x.quantile(0.75))


# 지역들 분리하는 df 생성
high_df = df[df['price_level'] == 'High'].copy()
mid_df  = df[df['price_level'] == 'Mid'].copy()
low_df  = df[df['price_level'] == 'Low'].copy()



# ——————————————————————————————
# 1) 그룹별 중위값, 75% 분위수(threshold) 계산 & 플래그 부여
# ——————————————————————————————

# High 그룹
high_med      = high_df['SalePrice'].median()
high_area_th  = high_df['GrLivArea'].quantile(0.75)
high_remod_th = high_df['YearRemodAdd'].quantile(0.75)
high_den_th   = high_df['RoomDensity'].quantile(0.75)
# 조건
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


# ——————————————————————————————
# 2) “Median 이하” 필터링 & 플래그 통계 (.copy() 사용)
# ——————————————————————————————
for name, gdf, med in [
    ('High', high_df, high_med),
    ('Mid',  mid_df,  mid_med),
    ('Low',  low_df,  low_med),
]:
    # Median 이하인 매물만 복사본으로?
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


# ——————————————————————————————
# 3) 원본 df에 score 계산 & 허위매물 후보 추출
# ——————————————————————————————

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
# r은 각행을나타내는 변수

# (2) 그룹별 중위값 이하 여부    중위값 이하 여부
median_price = df.groupby('price_level')['SalePrice'].transform('median')

# (3) suspect_flag 생성
df['suspect_flag'] = (df['SalePrice'] <= median_price) & (df['score'] >= 3)

# (4) 허위매물 후보만 추출
suspect_df = df[df['suspect_flag']].copy()

# (5) 결과 출력: 모든 컬럼 포함
print(f"허위매물 후보 총 {len(suspect_df)}건")
suspect_df




''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#### 지도시각화 ####

import plotly.express as px

# 지도 중심을 데이터의 평균 위도·경도로 설정
center = {
    "lat": df["Latitude"].mean(),
    "lon": df["Longitude"].mean()
}

fig = px.scatter_mapbox(
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

fig.show()


####### 점수제로 얻은 72개의 허위매물 후보랑
# 릿지 회귀 모델을 사용해서 구한 허위매물들간의 비교를 하겠다.  

'''''''''''''''''''''''''''''''''''''''''''RidgeCV 사용 회귀모델'
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

# 사용할 6개 피처와 타겟 정의 피처: 속성
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
        alphas=np.logspace(-4, 1, 10),  # 다양한 alpha 값 검사 (10개) 과적합을 방지하기 위해 페널티 부여 클수록 규제 강하게
        cv=5,                           # 5-fold 교차검증 5번 과적합 방지 한번만하면 일반적인 패턴 파악 못하고 훈련 데이터만 적합한 결과를 만들기 때문
        scoring='neg_mean_squared_error' # MSE를 최소화하는 alpha 선택
    )
    ridge.fit(X_train, y_train)
    
    # 4) Test set 성능 평가 모델의 일반화 능력 평가를 위한test set 
    y_test_pred = ridge.predict(X_test) # ridge모델로 테스트 데이터 예측 수행
    r2   = r2_score(y_test, y_test_pred) # 모델이 실제 데이터 분산을 얼마나 잘 설명하는지 나타내는 지표
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
    print("\n" + "-"*50 + "\n") # 예측 계수 다뽑음
    
    # 5) 전체 그룹 데이터에 대해 예측 및 residual 계산
    df_lvl['predicted'] = ridge.predict(X)
    df_lvl['residual']  = df_lvl['SalePrice'] - df_lvl['predicted']
    
    # 6) 이상치(허위매물) 플래그: residual 하위 25% 이하면 True
    thresh = df_lvl['residual'].quantile(72/2579) # 우리가 전체 대비 점수제로 뽑은 매물후보 수의비율로 확인 
    df_lvl['ridge_flag'] = df_lvl['residual'] <= thresh
    
    # 7) 인터랙티브 산점도
    fig = px.scatter(
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
    fig.add_shape(
        type='line', x0=mn, y0=mn, x1=mx, y1=mx,
        line=dict(color='black', dash='dash')
    )
    fig.update_layout(width=600, height=600)
    fig.show()
    
#### 예측한 값 중에서 72/2579%만 뽑아서 우리가 선정한 후보 매물이랑 개수 비교교



# ———————————————————————————————————————
# 4. 두 가지 메서드로 뽑힌 허위매물 비교
# ———————————————————————————————————————

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
    thresh = resid.quantile(72/2579)
    df.loc[mask, 'ridge_flag'] = resid <= thresh

# (4) 인덱스 집합으로 변환
score_set   = set(df.index[df['suspect_flag']])
ridge_set   = set(df.index[df['ridge_flag']])

# (5) 개수 요약 출력
print("=== 허위매물 건수 비교 ===")
print(f"점수제(suspect_flag) : {len(score_set)}건")
print(f"Ridge   기준      : {len(ridge_set)}건\n")

print("=== 교집합 건수 ===")
print(f"점수&Ridget 공통 허위매물     : {len(score_set & ridge_set)}건")


# (7) 각 그룹별 예시 뽑아서 보기
print(">>> 두 방법 모두 의심한 매물 (공통)")
df.loc[list(score_set & ridge_set)]

print(">>> 오직 점수제만 의심한 매물")
df.loc[list(score_set - ridge_set)]

print(">>> 오직 Ridge만 의심한 매물")
df.loc[list(ridge_set - score_set)]



import plotly.express as px

# 공통 허위매물 인덱스 추출
common_suspects = list(score_set & ridge_set)

# 공통 허위매물 데이터프레임 생성
common_df = df.loc[common_suspects].copy()

# 지도 중심점 설정
center = {
    "lat": common_df["Latitude"].mean(),
    "lon": common_df["Longitude"].mean()
}

# 지도 시각화
fig = px.scatter_mapbox(
    common_df,
    lat="Latitude",
    lon="Longitude",
    color="price_level",
    hover_name="Neighborhood",
    hover_data=["SalePrice", "GrLivArea", "OverallQual", "amenities"],
    zoom=11.5,
    center=center,
    height=600,
    mapbox_style="open-street-map",
    title="🔍 점수제 + Ridge 공통 허위매물 후보 위치"
)

fig.show()

import folium
df.loc[list(score_set & ridge_set)]
# 공통 허위매물만 추출
common_df = df.loc[list(score_set & ridge_set)].copy()

common_df = common_df.dropna(subset=["Latitude", "Longitude"])

# 지도 중심 좌표 설정
center_lat = common_df["Latitude"].mean()
center_lon = common_df["Longitude"].mean()

# folium 지도 객체 생성
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# price_level별 색상 지정
color_map = {
    'Low': 'blue',
    'Mid': 'green',
    'High': 'red'
}

# 마커 추가
for i, row in common_df.iterrows():
    popup_text = f"""
    <b>지역:</b> {row['Neighborhood']}<br>
    <b>매매가:</b> ${row['SalePrice']:,}<br>
    <b>거실면적:</b> {row['GrLivArea']} sqft<br>
    <b>전체평가:</b> {row['OverallQual']}<br>
    <b>편의시설 수:</b> {row['amenities']}<br>
    <b>점수제 점수:</b> {row['score']}
    """
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=folium.Popup(popup_text, max_width=250),
        icon=folium.Icon(color=color_map[row['price_level']], icon='exclamation-sign', prefix='glyphicon')
    ).add_to(m)

# 지도 출력
m