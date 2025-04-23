import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
import plotly.express as px

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

df['RoomDensity'] = df['TotRmsAbvGrd'] / df['GrLivArea']

room_density_threshold = 0.01

# 0) “허위매물 판단”에 쓸 6개 피처 정의
features = [
    'OverallQual',
    'OverallCond',
    'GrLivArea',
    'YearRemodAdd',
    'RoomDensity',
    'amenities'
]
target = 'SalePrice'

# 1) price_level별 ElasticNetCV 학습 + 예측 + 시각화
for level in ['Low', 'Mid', 'High']:
    # 해당 그룹 데이터
    df_lvl = df[df['price_level'] == level].copy()
    X = df_lvl[features]
    y = df_lvl[target]
    
    # train/test 분할 (모델 일반화 확인용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    
    # ElasticNetCV 모델 학습
    elastic = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99],
        alphas=np.logspace(-4, 1, 6),
        cv=5,
        max_iter=10000,
        random_state=0
    )
    elastic.fit(X_train, y_train)
    
    print(f"{level} 그룹 → 최적 l1_ratio={elastic.l1_ratio_}, alpha={elastic.alpha_:.6f}")
    
    # 전체 그룹에 대해 예측 & 잔차 계산
    df_lvl['predicted'] = elastic.predict(X)
    df_lvl['residual']  = df_lvl['SalePrice'] - df_lvl['predicted']
    
    # elastic_flag: residual 하위 25% 이하면 의심
    thresh = df_lvl['residual'].quantile(0.25)
    df_lvl['elastic_flag'] = df_lvl['residual'] <= thresh
    
    # Plotly 산점도: 실제 vs 예측, 의심 매물만 빨강
    fig = px.scatter(
        df_lvl,
        x='SalePrice',
        y='predicted',
        color='elastic_flag',
        color_discrete_map={True: 'red', False: 'lightgray'},
        title=f'Actual vs. Predicted ({level})',
        labels={'SalePrice':'Actual','predicted':'Predicted','elastic_flag':'Suspect'},
        opacity=0.7
    )
    # y=x 대각선
    mn, mx = df_lvl['SalePrice'].min(), df_lvl['SalePrice'].max()
    fig.add_shape(
        type='line', x0=mn, y0=mn, x1=mx, y1=mx,
        line=dict(color='black', dash='dash')
    )
    fig.update_layout(width=600, height=600)
    fig.show()

################################################
# 빈 리스트에 저장
results = []

for level in ['Low', 'Mid', 'High']:
    df_lvl = df[df['price_level'] == level].copy()
    X = df_lvl[features]
    y = df_lvl[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    elastic = ElasticNetCV(
        l1_ratio=[.1, .5, .7, .9, .95, .99],
        alphas=np.logspace(-4, 1, 6),
        cv=5,
        max_iter=10000,
        random_state=0
    )
    elastic.fit(X_train, y_train)

    df_lvl['predicted'] = elastic.predict(X)
    df_lvl['residual'] = df_lvl['SalePrice'] - df_lvl['predicted']
    thresh = df_lvl['residual'].quantile(0.25)
    df_lvl['elastic_flag'] = df_lvl['residual'] <= thresh
    df_lvl['price_level_group'] = level  # 구분용 컬럼

    results.append(df_lvl)

# 하나로 합치기
df_all = pd.concat(results, ignore_index=True)

import seaborn as sns
import matplotlib.pyplot as plt

# 시각화 함수
def plot_cat_dist(var):
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df_all, 
        x=var, 
        hue='elastic_flag', 
        order=df_all[var].value_counts().index,
        palette={True: 'red', False: 'gray'}
    )
    plt.title(f'{var} Distribution (Red: Suspect)')
    plt.xticks(rotation=45)
    plt.show()

# 3개 변수 모두 보기
for col in ['SaleType', 'MiscFeature', 'SaleCondition']:
    plot_cat_dist(col)


# 야호