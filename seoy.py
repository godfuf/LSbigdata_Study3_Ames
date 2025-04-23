import numpy as np
import pandas as pd

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


# 조건 확인별 히스토그램

# 산점도 + 평균선 함께 시각화
plt.figure(figsize=(10, 6))
sns.stripplot(data=df, x='price_level', y='amenities', jitter=True, palette='Set2', alpha=0.5)
sns.pointplot(data=df, x='price_level', y='amenities', estimator='mean', color='black', markers='D', linestyles='--')

plt.title('Amenities Count by Price Level (Scatter + Mean Line)')
plt.xlabel('Price Level')
plt.ylabel('Number of Amenities')
plt.grid(True)
plt.show()



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
    (df['GarageArea'] > 0).astype(int)
)

# 세 그룹 프레임 나누기
high_df = df[df['price_level'] == 'High'].copy()
mid_df = df[df['price_level'] == 'Mid'].copy()
low_df = df[df['price_level'] == 'Low'].copy()

#################### 편의시설 개수 시각화
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(18, 6))

# High Price - 어메니티스 분포
plt.subplot(1, 3, 1)
sns.histplot(high_df['amenities'], kde=True, bins=10, color='blue')
plt.title('High Price - Amenities Count Distribution')
plt.xlabel('Number of Amenities')
plt.ylabel('Frequency')

# Mid Price - 어메니티스 분포
plt.subplot(1, 3, 2)
sns.histplot(mid_df['amenities'], kde=True, bins=10, color='orange')
plt.title('Mid Price - Amenities Count Distribution')
plt.xlabel('Number of Amenities')
plt.ylabel('Frequency')

# Low Price - 어메니티스 분포
plt.subplot(1, 3, 3)
sns.histplot(low_df['amenities'], kde=True, bins=10, color='green')
plt.title('Low Price - Amenities Count Distribution')
plt.xlabel('Number of Amenities')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


###########################################


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


# OverallQual = 9
# OverallCond = 6



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


