import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

ames=pd.read_csv('ames.csv')
ames.info()
ames.columns
ames.head()

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



