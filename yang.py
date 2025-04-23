import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 불러오기
ames = pd.read_csv('C:/Users/USER/Documents/lsbigdata-gen4/1joameshouse/group1_project/ames.csv')
ames.keys()
ames['Neighborhood'].isna().sum()

########################################
ames['Neighborhood'].unique()

ames.info()

# 복사 및 전처리
df = ames.copy()
df['price_per_area'] = df['SalePrice'] / df['GrLivArea']
df['house_age'] = df['YrSold'] - df['YearBuilt']
df['neigh_avg'] = df.groupby('Neighborhood')['SalePrice'].transform('mean')

# 1. 고가/저가/중간 나누기
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

# 2. 모든 범주형 데이터 결측치 >> None 추가

# 범주형 컬럼만 선택
cat_cols = df.select_dtypes(include='object').columns

# 결측치 'None'으로 대체
df[cat_cols] = df[cat_cols].fillna('None')

############################################ 조건 확인
############################################
# 1. overqual(전체 퀄리티 재료 마감 품질) 얼마 이상 
# 2. overcondition(전체 상태 등급 주택 상태) 얼마 이상
# 3. 평수 얼마 이상
# 4. 수영장 지하실 차고 존재 여부(편의시설 개수)
# 5. 집구조 평수가 작은데 화장실이 여러개 이런거 여부
df.keys()
df.groupby('price_level')['OverallQual'].mean()
##############3


high_overallqual = df[df['price_level']=='High']['OverallQual']
mid_overallqual = df[df['price_level']=='Mid']['OverallQual']
low_overallqual = df[df['price_level']=='Low']['OverallQual']

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

# 서브플롯 생성 (1행 3열)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# 히스토그램 그리기
axs[0].hist(high_overallqual, bins=10, color='red', edgecolor='black')
axs[0].set_title('High Price Level')
axs[0].set_xlabel('OverallQual')
axs[0].set_ylabel('Count')

axs[1].hist(mid_overallqual, bins=10, color='orange', edgecolor='black')
axs[1].set_title('Mid Price Level')
axs[1].set_xlabel('OverallQual')

axs[2].hist(low_overallqual, bins=10, color='green', edgecolor='black')
axs[2].set_title('Low Price Level')
axs[2].set_xlabel('OverallQual')

# 레이아웃 조정
plt.tight_layout()
plt.show()

# overallqual
# high 9 10
# mid 8 9
# low 7 8 9 10

high_overallcond = df[df['price_level']=='High']['OverallCond']
mid_overallcond = df[df['price_level']=='Mid']['OverallCond']
low_overallcond = df[df['price_level']=='Low']['OverallCond']

# 서브플롯 생성 (1행 3열)
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# 히스토그램 그리기
axs[0].hist(high_overallcond, bins=10, color='red', edgecolor='black')
axs[0].set_title('High Price Level')
axs[0].set_xlabel('OverallQual')
axs[0].set_ylabel('Count')

axs[1].hist(mid_overallcond, bins=10, color='orange', edgecolor='black')
axs[1].set_title('Mid Price Level')
axs[1].set_xlabel('OverallQual')

axs[2].hist(low_overallcond, bins=10, color='green', edgecolor='black')
axs[2].set_title('Low Price Level')
axs[2].set_xlabel('OverallQual')

# 레이아웃 조정
plt.tight_layout()
plt.show()

# overcond 히스토그램 보고 확인

# high 678910
# mid 678910
# low 8910



# 그릴 컬럼과 price_level 정의
cols   = ['GrLivArea', 'YearRemodAdd']
levels = ['Low', 'Mid', 'High']


# 박스플롯
for col in cols:
    for level in levels:
        data = df[df['price_level'] == level][col]
        
        plt.figure(figsize=(8, 3))
        plt.boxplot(data, vert=False, patch_artist=True,
                    boxprops=dict(edgecolor='black'))
        plt.title(f'{level} 그룹 — {col} 분포 (박스플롯)')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()




# 편의시설 개수 계산 (수영장, 지하실, 차고 존재 여부)
df['amenities'] = (
    (df['PoolArea'] > 0).astype(int) +
    (df['TotalBsmtSF'] > 0).astype(int) +
    (df['GarageArea'] > 0).astype(int) + 
    (df['MiscVal'] > 0).astype(int)
)

plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='price_level', y='amenities', 
              jitter=True, palette='Set2', alpha=0.5)

sns.pointplot(data=df, x='price_level', y='amenities', 
              estimator='mean',
              color='black', markers='D', 
              linestyles='--')

plt.title('Amenities Count by Price Level (Scatter + Mean Line)')
plt.xlabel('Price Level')
plt.ylabel('Number of Amenities')
plt.grid(True)

plt.show()

# 세 그룹 프레임 나누기
high_df = df[df['price_level'] == 'High'].copy()
mid_df = df[df['price_level'] == 'Mid'].copy()
low_df = df[df['price_level'] == 'Low'].copy()

# 세 그룹 프레임 나누기
high_df = df[df['price_level'] == 'High'].copy()
mid_df = df[df['price_level'] == 'Mid'].copy()
low_df = df[df['price_level'] == 'Low'].copy()

# 기준값들 (전체 df 기준으로 계산)
qual_75 = df['OverallQual'].quantile(0.75)
area_median = df['GrLivArea'].median()  # 이건 그대로 사용
amenities_75 = df['amenities'].quantile(0.75)
grlivarea_q1 = df['GrLivArea'].quantile(0.25)
overallcond_q3 = df['OverallCond'].quantile(0.75)

# High 가격대 조건
high_df['flag_high_qual'] = (high_df['OverallQual'] < qual_75).astype(int)
high_df['flag_high_area'] = (high_df['GrLivArea'] < area_median).astype(int)
high_df['flag_high_amenities'] = (high_df['amenities'] < amenities_75).astype(int)
high_df['flag_layout_mismatch'] = (
    (high_df['GrLivArea'] < grlivarea_q1) &
    ((high_df['FullBath'] + high_df['HalfBath']) >= 3)
).astype(int)
high_df['flag_good_condition'] = (
    high_df['OverallCond'] >= overallcond_q3
).astype(int)

# Mid 가격대 조건
mid_df['flag_mid_qual'] = (mid_df['OverallQual'] < qual_75).astype(int)  # 약간 더 완화
mid_df['flag_mid_area'] = (mid_df['GrLivArea'] < area_median).astype(int)
mid_df['flag_mid_amenities'] = (mid_df['amenities'] < amenities_75).astype(int)
mid_df['flag_layout_mismatch'] = (
    (mid_df['GrLivArea'] < grlivarea_q1) &
    ((mid_df['FullBath'] + mid_df['HalfBath']) >= 3)
).astype(int)
mid_df['flag_good_condition'] = (
    mid_df['OverallCond'] >= overallcond_q3
).astype(int)

# Low 가격대 조건
low_df['flag_low_qual'] = (low_df['OverallQual'] < qual_75).astype(int)
low_df['flag_low_area'] = (low_df['GrLivArea'] < area_median).astype(int)
low_df['flag_low_amenities'] = (low_df['amenities'] < amenities_75).astype(int)
low_df['flag_layout_mismatch'] = (
    (low_df['GrLivArea'] < grlivarea_q1) &
    ((low_df['FullBath'] + low_df['HalfBath']) >= 3)
).astype(int)
low_df['flag_good_condition'] = (
    low_df['OverallCond'] >= overallcond_q3
).astype(int)


high_df
mid_df
low_df

high_df['suspicious_flag'] = high_df[[
    'flag_high_qual', 'flag_high_area', 'flag_high_amenities',
    'flag_layout_mismatch', 'flag_good_condition'
]].sum(axis=1)

mid_df['suspicious_flag'] = mid_df[[
    'flag_mid_qual', 'flag_mid_area', 'flag_mid_amenities',
    'flag_layout_mismatch', 'flag_good_condition'
]].sum(axis=1)

low_df['suspicious_flag'] = low_df[[
    'flag_low_qual', 'flag_low_area', 'flag_low_amenities',
    'flag_layout_mismatch', 'flag_good_condition'
]].sum(axis=1)

high_df
mid_df
low_df

suspicious_h = high_df[high_df['suspicious_flag'] >= 4]
suspicious_m = mid_df[mid_df['suspicious_flag'] >= 4]
suspicious_l = low_df[low_df['suspicious_flag'] >= 4]

########################################333#####################333
# 최종 코드
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

## 2. 모든 범주형 데이터 결측치 >> None 추가

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


# # 1) 각 그룹별 중위값을 계산해 새로운 컬럼에 저장
# df['SalePrice_median'] = df.groupby('price_level')['SalePrice'].transform('median')

# # 2) 그룹별 중위값 이하인 매물만 필터링
# df_half = df[df['SalePrice'] <= df['SalePrice_median']]


# ''''''''''''''''''''''''''''''''''''

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



# ''''''''''''''''''''''''''''''''''''


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

## high 3 mid 55 low 14






######
df_all = pd.concat(results, ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt

# 시각화 함수
def plot_cat_dist(var):
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df_all, 
        x=var, 
        hue='elastic_flag', 
        order=df_all[var].value_counts().index,
        palette={True: 'red', False: 'gray'}
    )
    plt.title(f'{var} Distribution (Red: Suspect)')
    plt.xticks(rotation=45)
    plt.show()

# 3개 변수 모두 보기
for col in ['SaleType', 'MiscFeature', 'SaleCondition']:
    plot_cat_dist(col)
# 
['MSZoning']













###########조건 시각화
df_all = pd.concat([high_df, mid_df, low_df], ignore_index=True)

# 그룹별 flag 컬럼 정리
flag_cols = [c for c in df_all.columns if c.startswith('flag_')]

# 점수 계산
df_all['suspicious_score'] = df_all[flag_cols].sum(axis=1)

# 허위매물 플래그: 점수 3 이상이면 1
df_all['suspicious_flag'] = (df_all['suspicious_score'] >= 3).astype(int)

# Neighborhood 기준 가상 좌표 생성
np.random.seed(42)
neighs = df_all['Neighborhood'].unique()
lat_map = dict(zip(neighs, np.random.uniform(42.0, 42.1, len(neighs))))
lon_map = dict(zip(neighs, np.random.uniform(-93.7, -93.5, len(neighs))))

df_all['Latitude'] = df_all['Neighborhood'].map(lat_map)
df_all['Longitude'] = df_all['Neighborhood'].map(lon_map)

# 허위매물만 필터링
final_df = df_all[df_all['suspicious_flag'] == 1].copy()


import plotly.express as px

fig = px.scatter_mapbox(final_df,
                        lat="Latitude",
                        lon="Longitude",
                        size="suspicious_score",
                        color="price_level",
                        hover_name="Neighborhood",
                        hover_data={
                            "SalePrice": True,
                            "GrLivArea": True,
                            "YearRemodAdd": True,
                            "suspicious_score": True,
                            "Latitude": False, 
                            "Longitude": False
                        },
                        size_max=15,
                        zoom=11,
                        height=700)

fig.update_layout(
    mapbox_style="carto-positron",
    margin={"r": 0, "t": 50, "l": 0, "b": 0},
    title="🚩 Suspicious Houses Map (Score ≥ 3)",
    title_font=dict(size=24, color='black', family="Arial"),
    font=dict(size=14, color='black'),
    paper_bgcolor="lightgray",
    plot_bgcolor="white",
    showlegend=True
)

fig.update_traces(marker=dict(opacity=0.7, symbol='circle'))
fig.show()


########################################
