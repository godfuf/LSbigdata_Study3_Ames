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
ames['Neighborhood'].unique()
ames.info()

import numpy as np
import pandas as pd

# 복사 및 전처리
df = ames.copy()
df.columns

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

