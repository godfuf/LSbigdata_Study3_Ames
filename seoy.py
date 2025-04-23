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







# 조건별 플래그 생성
df['flag_cond1'] = ((df['price_level'] == 'High') &
                    (df['OverallQual'] < 8)).astype(int)

df['flag_cond2'] = ((df['price_level'] == 'Mid') &
                    (df['OverallQual'] < 6)).astype(int)

df['flag_cond2'] = ((df['price_level'] == 'Low') &
                    (df['OverallQual'] < 5)).astype(int)

df['flag_cond4'] = ((df['price_level'] == 'High') &
                    (df['GrLivArea'] < df['GrLivArea'].median())).astype(int)

df['flag_cond5'] = ((df['price_level'] == 'Mid') &
                    (df['GrLivArea'] < df['GrLivArea'].median())).astype(int)

df['flag_cond6'] = ((df['price_level'] == 'Low') &
                    (df['GrLivArea'] < df['GrLivArea'].median())).astype(int)


# 최종 suspicious 점수 계산
df['suspicious_flag'] = df[['flag_cond1', 'flag_cond2', 'flag_cond3', 'flag_cond4', 'flag_cond5', 'flag_cond6']].sum(axis=1)

# suspicious 점수가 2 이상인 매물 추출
suspicious_listings = df[df['suspicious_flag'] >= 2]

# 결과 출력
cols_to_show = [
    'PID', 'Neighborhood', 'SalePrice', 'GrLivArea', 'OverallQual', 'OverallCond', 
    'YearBuilt', 'price_per_area', 'house_age',
    'flag_cond1', 'flag_cond2', 'flag_cond3', 'flag_cond4', 'flag_cond5', 'flag_cond6',
    'suspicious_flag'
]

print(suspicious_listings[cols_to_show].sort_values(by='suspicious_flag', ascending=False))

