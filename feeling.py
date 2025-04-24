import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
import plotly.express as px

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 설정
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


# 데이터 불러오기
ames=pd.read_csv('ames.csv')

# 복사 및 전처리
df = ames.copy()

# Neighborhood별 평균 SalePrice 기준 분위수 계산
df_ns = df.groupby('Neighborhood')['SalePrice'].mean()
q1, q2 = df_ns.quantile([0.25, 0.75])


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


## 2. 모든 범주형 데이터 결측치 >> None 추가

# 범주형 컬럼만 선택
cat_cols = df.select_dtypes(include='object').columns
# 범주형 결측치 'None'으로 대체
df[cat_cols] = df[cat_cols].fillna('None')



#편의시설 칼럼 추가[수영장,지하실,차고,MiscFeature칼럼의 부가가치 수치화(Elev,Gar2,Shed,TenC)
df['amenities'] = (
    (df['PoolArea'] > 0).astype(int) +
    (df['TotalBsmtSF'] > 0).astype(int) +
    (df['GarageArea'] > 0).astype(int) + 
    (df['MiscVal'] > 0).astype(int)
)

# 주거면적대비 모든 방수가 차지하는 정도를 비율로 표현
df['TotalRooms'] = df['TotRmsAbvGrd'] + df['HalfBath'] + df['FullBath']  # 욕실 제외 방수 + 반욕실 + 풀욕실
df['RoomDensity'] = df['TotalRooms'] / df['GrLivArea']  # 방 밀도 (방수 / 거실 면적)


''''''''''''''''''''''''''''''''''''
df['GrLivArea']

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

high_df['Neighborhood']
mid_df['Neighborhood']
low_df['Neighborhood']





# ——————————————————————————————
# 1) 그룹별 중위값, 75% 분위수(threshold) 계산 & 플래그 부여
# ——————————————————————————————

# High 그룹
high_med      = high_df['SalePrice'].median() # 
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


# ——————————————————————————————
# 2) “Median 이하” 필터링 & 플래그 통계 (.copy() 사용)
# ——————————————————————————————
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

# (2) 그룹별 중위값 이하 여부
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




'''''''''''''''''''''''''''''''''''''''''''RidgeCV 사용 회귀모델'
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

