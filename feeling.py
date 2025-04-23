import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# 한글 설정하고 시작
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


# 데이터 불러오기
ames=pd.read_csv('ames.csv')

# 복사 및 전처리
df = ames.copy()


# Neighborhood별 평균 SalePrice 기준 분위수 계산
df_ns = df.groupby('Neighborhood')['SalePrice'].mean()
q1, q2 = df_ns.quantile([0.25, 0.75])

df['price_level'] = np.select(
    [
        df['Neighborhood'].isin(df_ns[df_ns <= q1].index),
        df['Neighborhood'].isin(df_ns[(df_ns > q1) & (df_ns <= q2)].index),
        df['Neighborhood'].isin(df_ns[df_ns > q2].index)
    ],
    ['Low', 'Mid', 'High'],
    default='Unknown'  #  문자열로 통일
)

# 기본 변수 계산

#편의시설[수영장, 지하실, 차고 존재 여부로 점수 부여]
df['amenities'] = (df['PoolArea'] > 0).astype(int) + (df['TotalBsmtSF'] > 0).astype(int) + (df['GarageArea'] > 0).astype(int) + (df['MiscVal'] > 0).astype(int)

#   전체 면적 대비 방의 비율[욕실 제외 방수 + 반욕실+풀욕실][집의 구성이 정상인지 파악]
df['TotalRooms'] = df['TotRmsAbvGrd'] + df['HalfBath'] + df['FullBath']
df['RoomDensity'] = df['TotalRooms'] / df['GrLivArea']

# 
df['YearRemodAdd_th'] = df.groupby('price_level')['YearRemodAdd'].transform(lambda x: x.quantile(0.75))

# 기준값
area_median = df['GrLivArea'].median()
room_density_threshold = 0.01
qual_thresholds = {'Low': 7, 'Mid': 8, 'High': 9}

# price_level별 조건 점수 계산
suspicious_list = []
for level, qual_th in qual_thresholds.items():
    sub = df[df['price_level'] == level].copy()
    sub['suspicious_flag'] = (
        (sub['OverallQual'] >= qual_th).astype(int) +
        (sub['GrLivArea'] < area_median).astype(int) +
        (sub['amenities'] >= 3).astype(int) +
        (sub['RoomDensity'] >= room_density_threshold).astype(int) +
        (sub['OverallCond'] >= 6).astype(int) +
        (sub['YearRemodAdd'] >= sub['YearRemodAdd_th']).astype(int)
    )
    suspicious_list.append(sub[sub['suspicious_flag'] >= 3])

# 최종 suspicious 데이터프레임
suspicious_df = pd.concat(suspicious_list)













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
ames = pd.read_csv('./ames.csv')

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
























# 결측치가 1개 이상 있는 컬럼만 출력
null_cols = df.columns[df.isnull().any()]
df[null_cols].isnull().sum().sort_values(ascending=False)

# 1. 고가/저가/중간 나누기
df_ns = df.groupby('Neighborhood')['SalePrice'].mean()

# 분위수 계산
q1 = df_ns.quantile(0.25)
q2 = df_ns.quantile(0.75)


# 3. 조건 기반 price_level 설정
def get_price_level(neighborhood):
    avg = df_ns.get(neighborhood, np.nan)
    if pd.isna(avg):
        return np.nan
    elif avg <= q1:
        return 'Low'
    elif avg <= q2:
        return 'Mid'
    else:
        return 'High'

# 4. 컬럼 생성
df['price_level'] = df['Neighborhood'].apply(get_price_level)


# 그릴 컬럼과 price_level 정의
cols = ['TotalRooms', 'RoomDensity']
levels = ['Low', 'Mid', 'High']

# 1) 히스토그램
for col in cols:
    for level in levels:
        data = df[df['price_level'] == level][col]
        
        plt.figure(figsize=(8, 5))
        plt.hist(data, bins=30, edgecolor='black')
        plt.title(f'{level} 그룹 — {col} 분포')
        plt.xlabel(col)
        plt.ylabel('빈도')
        plt.tight_layout()
        plt.show()

# 2) 박스플롯
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
    (df['GarageArea'] > 0).astype(int)
)

# 세 그룹 프레임 나누기
high_df = df[df['price_level'] == 'High'].copy()
mid_df = df[df['price_level'] == 'Mid'].copy()
low_df = df[df['price_level'] == 'Low'].copy()



# High Price - 어메니티스 분포
plt.subplot(1, 3, 1)
sns.histplot(high_df['amenities'], kde=True, bins=10, color='blue')
plt.title('High Price - Amenities Count Distribution')
plt.xlabel('Number of Amenities')
plt.ylabel('Frequency')
