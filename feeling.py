import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# 데이터 불러오기
ames=pd.read_csv('ames.csv')
ames['Neighborhood'].unique()
ames.info()

import numpy as np
import pandas as pd

# 복사 및 전처리
df = ames.copy()

# 1. 지역별 주택 평균가 
df_ns = df.groupby('Neighborhood')['SalePrice'].mean()


# 2. 분위수 기준 나누기 (상위 30%, 중간 40%, 하위 30%)
q1 = df_ns.quantile(0.3)
q2 = df_ns.quantile(0.7)

# 3. 구간 나누기
def price_group(price):
    if price <= q1:
        return 'low'
    elif price <= q2:
        return 'mid'
    else:
        return 'high'

# 4. 그룹 라벨링
df_ns_grouped = df_ns.apply(price_group)
# 결과 확인
df_ns_grouped.value_counts()
# 2. 모든 범주형 데이터 결측치 >> None 추가
# 범주형 컬럼만 선택
cat_cols = df.select_dtypes(include='object').columns
# 결측치 'None'으로 대체
df[cat_cols] = df[cat_cols].fillna('None')


# 1. Neighborhood 평균 가격 기반 그룹 정보 병합
df['Neighborhood_Group'] = df['Neighborhood'].map(df_ns_grouped)



import plotly.express as px

fig = px.box(df, x='Neighborhood_Group', y='SalePrice',
             category_orders={'Neighborhood_Group': ['low', 'mid', 'high']},
             title='주택 가격 분포 (지역 그룹별)',
             labels={'Neighborhood_Group': '지역 그룹', 'SalePrice': '주택 가격'},
             points='all')  # points='all'로 이상치도 표시 가능

fig.update_layout(template='plotly_white')
fig.show()




















# 복사 및 전처리
df = ames.copy()
df['price_per_area'] = df['SalePrice'] / df['GrLivArea']
df['house_age'] = df['YrSold'] - df['YearBuilt']
df['neigh_avg'] = df.groupby('Neighborhood')['SalePrice'].transform('mean')










ames['Neighborhood'].value_counts()
ames['Neighborhood'].describe()

ames['Utilities'].value_counts()
ames['Bathroom'].value_counts()
ames['TotalBsmtSF'].value_counts()

# 1. 데이터 로드 및 전처리

# 예시: 총면적 및 면적당 가격 생성
# 총면적 계산 (1층 + 2층)
ames['TotalArea'] = ames['1st Flr SF'] + ames['2nd Flr SF']
# 면적당 가격 계산 (종속변수 생성, 지하면적은 일반면적대비 60퍼 가치로 적용)
ames['AdjustedTotalArea'] = ames['TotalArea'] + ames['TotalBsmtSF'] * 0.6
ames['Price_per_sqft'] = ames['SalePrice'] / ames['AdjustedTotalArea']
ames = ames[ames['Price_per_sqft'].notnull()]  # 결측 제거



# 결측치 간단 처리 (중요 변수 위주로만)-----------------논의 필요[차고, 벽난로, 석조 마감 면적]
df.fillna({'GarageArea': 0, 'Fireplaces': 0, 'Mas Vnr Area': 0}, inplace=True)







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