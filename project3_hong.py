import numpy as np
import pandas as pd

# 데이터 불러오기
ames = pd.read_csv('./ames.csv')

ames['Neighborhood'].unique()

ames.info()

# 복사 및 전처리
df = ames.copy()

# 결측치가 1개 이상 있는 컬럼만 출력
null_cols = df.columns[df.isnull().any()]
df[null_cols].isnull().sum().sort_values(ascending=False)

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

df

# ——————————————————————————————
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



'''조건들
1. overqual(전체 퀄리티 재료 마감 품질) 얼마 이상 
2. overcondition(전체 상태 등급 주택 상태) 얼마 이상
3. 평수 얼마 이상
 '''