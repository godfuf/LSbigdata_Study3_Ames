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
# 회귀식 elastic
from sklearn.linear_model import ElasticNet
elasticnet = ElasticNet()
elastic_params = {'alpha' : np.arange(0.1, 1, 0.1),
                  'l1_ratio': np.linspace(0, 1, 5)}

from sklearn.model_selection import KFold, GridSearchCV
cv = KFold(n_splits=5, shuffle=True, random_state=0)

elastic_search = GridSearchCV(estimator=elasticnet,
                              param_grid=elastic_params,
                              cv=cv,
                              scoring='neg_mean_squared_error')

elastic_search.fit(X_train_all, y_train)

print(pd.DataFrame(elastic_search.cv_results_))
print(elastic_search.best_params_)
print(-elastic_search.best_score_)
