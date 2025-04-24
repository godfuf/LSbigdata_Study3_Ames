import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

############################## 전처리 과정 ##################################
# 데이터 불러오기
ames = pd.read_csv('C:/Users/USER/Documents/ls빅데이터/teamproject3/group1_project/ames.csv')

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

df.groupby('price_level')['OverallQual'].mean()

# 편의시설 개수 계산 (수영장, 지하실, 차고 존재 여부)
df['amenities'] = (
    (df['PoolArea'] > 0).astype(int) +
    (df['TotalBsmtSF'] > 0).astype(int) +
    (df['GarageArea'] > 0).astype(int) + 
    (df['MiscVal'] > 0).astype(int)
)

# 그룹 이름 설정
groups = ['Low', 'Mid', 'High']
colors = ['skyblue', 'salmon', 'lightgreen']

# 그림 생성
plt.figure(figsize=(15, 4))

# 각 그룹별로 subplot 생성
for i, group in enumerate(groups):
    plt.subplot(1, 3, i+1)
    subset = df[df['price_level'] == group]
    plt.hist(subset['amenities'], bins=range(0, 6), color=colors[i], edgecolor='black', align='left')
    plt.title(f'{group} Price Level')
    plt.xlabel('Number of Amenities')
    plt.ylabel('Count')
    plt.xticks(range(0, 5))

plt.suptitle('Histogram of Amenities by Price Level', fontsize=16)
plt.tight_layout()
plt.show()


# 기준값들 (전체 df 기준으로 계산)
qual_75 = df['OverallQual'].quantile(0.75)
area_median = df['GrLivArea'].median()  # 이건 그대로 사용
grlivarea_q1 = df['GrLivArea'].quantile(0.25)
overallcond_q3 = df['OverallCond'].quantile(0.75)
room_density_threshold = 0.01

# 5. 집 면적당 방 밀도 (방수 / 거실 면적)[비율이 0.01이상이면 이상치로 판단예정]

df['TotalRooms'] = df['TotRmsAbvGrd'] + df['HalfBath'] + df['FullBath']  # 욕실 제외 방수 + 반욕실 + 풀욕실
df['RoomDensity'] = df['TotalRooms'] / df['GrLivArea']  # 방 밀도 (방수 / 거실 면적)


# 허위매물 판단을 위한 기준값 계산
df['GrLivArea_th'] = df.groupby('price_level')['GrLivArea']\
                          .transform(lambda x: x.quantile(0.75))  # 상위 25% 기준
df['YearRemodAdd_th'] = df.groupby('price_level')['YearRemodAdd']\
                          .transform(lambda x: x.quantile(0.75))  # 상위 25% 기준


# 세 그룹 프레임 나누기
high_df = df[df['price_level'] == 'High'].copy()
mid_df = df[df['price_level'] == 'Mid'].copy()
low_df = df[df['price_level'] == 'Low'].copy()



# High 가격대 조건
high_df['flag_high_qual'] = (high_df['OverallQual'] >= 9).astype(int)
high_df['flag_high_area'] = (high_df['GrLivArea'] < area_median).astype(int)
high_df['flag_high_amenities'] = (high_df['amenities'] >= 3).astype(int)
high_df['flag_room_density'] = (high_df['RoomDensity'] >= room_density_threshold).astype(int)
high_df['flag_good_condition'] = (high_df['OverallCond'] >= 6).astype(int)
high_df['flag_old_remod'] = (high_df['YearRemodAdd'] >= high_df['YearRemodAdd_th']).astype(int)


# Mid 가격대 조건
mid_df['flag_mid_qual'] = (mid_df['OverallQual'] >= 8).astype(int) 
mid_df['flag_mid_area'] = (mid_df['GrLivArea'] < area_median).astype(int)
mid_df['flag_mid_amenities'] = (mid_df['amenities'] >= 3).astype(int)
mid_df['flag_room_density'] = (mid_df['RoomDensity'] >= room_density_threshold).astype(int)
mid_df['flag_good_condition'] = (mid_df['OverallCond'] >= 6).astype(int)
mid_df['flag_old_remod'] = (mid_df['YearRemodAdd'] <= mid_df['YearRemodAdd_th']).astype(int)

# Low 가격대 조건
low_df['flag_low_qual'] = (low_df['OverallQual'] >= 7).astype(int)
low_df['flag_low_area'] = (low_df['GrLivArea'] < area_median).astype(int)
low_df['flag_low_amenities'] = (low_df['amenities'] >= 3).astype(int)
low_df['flag_room_density'] = (low_df['RoomDensity'] >= room_density_threshold).astype(int)
low_df['flag_good_condition'] = (low_df['OverallCond'] >= 8).astype(int)
low_df['flag_old_remod'] = (low_df['YearRemodAdd'] <= low_df['YearRemodAdd_th']).astype(int)

high_df
mid_df
low_df

high_df['suspicious_flag'] = high_df[[
    'flag_high_qual', 'flag_high_area', 'flag_high_amenities',
    'flag_room_density', 'flag_good_condition', 'flag_old_remod'
]].sum(axis=1)

mid_df['suspicious_flag'] = mid_df[[
    'flag_mid_qual', 'flag_mid_area', 'flag_mid_amenities',
    'flag_room_density', 'flag_good_condition', 'flag_old_remod'
]].sum(axis=1)

low_df['suspicious_flag'] = low_df[[
    'flag_low_qual', 'flag_low_area', 'flag_low_amenities',
    'flag_room_density', 'flag_good_condition', 'flag_old_remod'
]].sum(axis=1)

high_df
mid_df
low_df

suspicious_h = high_df[high_df['suspicious_flag'] >= 3]
suspicious_m = mid_df[mid_df['suspicious_flag'] >= 3]
suspicious_l = low_df[low_df['suspicious_flag'] >= 3]


