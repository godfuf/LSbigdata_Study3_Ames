import numpy as np
import pandas as pd

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
# 하위 25 상위 25 중간 50
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


df.groupby(['price_level'])['OverallQual'].mean().reset_index()



# SalePrice 같은 변수는 대부분 **정규분포가 아니라 오른쪽으로 치우친 분포(skewed)**를 가집니다.

# 그래서 전체 평균만 보는 건 왜곡될 수 있어요!

import matplotlib.pyplot as plt
high_price = df[df['price_level']=='High']['SalePrice']
mid_price = df[df['price_level']=='Mid']['SalePrice']
low_price = df[df['price_level']=='Low']['SalePrice']



# OverallQual
High_OverllQual = df[df['price_level']=='High']['OverallQual']
Mid_OverllQual = df[df['price_level']=='Mid']['OverallQual']
Low_OverllQual = df[df['price_level']=='Low']['OverallQual']

plt.figure(figsize=(8, 5))
plt.hist(Mid_OverllQual, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Sale Prices')
plt.xlabel('high_OverllQual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# OverallQual 주택의 재료, 마감 품질을 종합적으로 평가하는 등급
# high: 9, 10
# mid: 8, 9
# low: 8, 9


# OverallCond 전체 주택상태 등급
High_OverallCond = df[df['price_level']=='High']['OverallCond']
Mid_OverallCond = df[df['price_level']=='Mid']['OverallCond']
Low_OverallCond = df[df['price_level']=='Low']['OverallCond']

plt.figure(figsize=(8, 5))
plt.hist(Mid_OverallCond, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of Sale Prices')
plt.xlabel('high_OverllQual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# high: 5가 대부분, 
# mid: 8, 9
# low: 8, 9





import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='price_level', y='OverallQual')
plt.title('Boxplot of OverallQual by Price Level')
plt.xlabel('Price Level')
plt.ylabel('Overall Quality')
plt.show()


# 1) 그룹별 임계값(threshold) 계산 및 df에 저장
# ——————————————————————————————
df['qual_th'] = df.groupby('price_level')['OverallQual'].transform('mean')
df['cond_th'] = df.groupby('price_level')['OverallCond'].transform('mean')
df['area_th'] = df.groupby('price_level')['GrLivArea'].transform('mean')

# ——————————————————————————————
# 2) 각 조건별 1점 부여
# ——————————————————————————————
df['overqual']      = (df['OverallQual']  >= df['qual_th']).astype(int)
df['overcondition'] = (df['OverallCond']  >= df['cond_th']).astype(int)
df['large_area']    = (df['GrLivArea']    >= df['area_th']).astype(int)

# ——————————————————————————————
# 3) 점수 합산
# ——————————————————————————————
df['score'] = df['overqual'] + df['overcondition'] + df['large_area']

# ——————————————————————————————
# 4) 결과 확인
# ——————————————————————————————
# 각 price_level 그룹별 score 분포 확인
print(df.groupby('price_level')['score']
        .value_counts()
        .unstack(fill_value=0))

# DataFrame의 일부 컬럼을 함께 출력
cols_to_show = [
    'price_level',
    'OverallQual','qual_th','overqual',
    'OverallCond','cond_th','overcondition',
    'GrLivArea','area_th','large_area',
    'score'
]
print(df[cols_to_show].head(10))



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

# High 가격대 조건
high_df['flag_high_qual'] = (high_df['OverallQual'] < 8).astype(int)
high_df['flag_high_area'] = (high_df['GrLivArea'] < df['GrLivArea'].median()).astype(int)
high_df['flag_high_amenities'] = (high_df['amenities'] < 2).astype(int)

# Mid 가격대 조건
mid_df['flag_mid_qual'] = (mid_df['OverallQual'] < 6).astype(int)
mid_df['flag_mid_area'] = (mid_df['GrLivArea'] < df['GrLivArea'].median()).astype(int)
mid_df['flag_mid_amenities'] = (mid_df['amenities'] < 2).astype(int)

# Low 가격대 조건
low_df['flag_low_qual'] = (low_df['OverallQual'] < 5).astype(int)
low_df['flag_low_area'] = (low_df['GrLivArea'] < df['GrLivArea'].median()).astype(int)
low_df['flag_low_amenities'] = (low_df['amenities'] < 2).astype(int)

high_df
mid_df
low_df

high_df['suspicious_flag'] = high_df[['flag_high_qual', 'flag_high_area', 
                                      'flag_high_amenities']].sum(axis=1)

mid_df['suspicious_flag'] = mid_df[['flag_mid_qual', 'flag_mid_area',
                                    'flag_mid_amenities' ]].sum(axis=1)

low_df['suspicious_flag'] = low_df[['flag_low_qual', 'flag_low_area',
                                    'flag_low_amenities']].sum(axis=1)

high_df
mid_df
low_df

suspicious_h = high_df[high_df['suspicious_flag'] >= 3]
suspicious_m = mid_df[mid_df['suspicious_flag'] >= 3]
suspicious_l = low_df[low_df['suspicious_flag'] >= 3]


