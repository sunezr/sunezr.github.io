
# Datawhale 零基础入门数据挖掘-Task4 建模调参 

Tip:此部分为零基础入门数据挖掘的 Task4 建模调参 部分，上接Task3

**赛题：零基础入门数据挖掘 - 二手车交易价格预测**

地址：https://tianchi.aliyun.com/competition/entrance/231784/introduction?spm=5176.12281957.1004.1.38b02448ausjSX 
    

##  内容介绍

1. 线性回归模型：
    - 线性回归对于特征的要求；
    - 处理长尾分布；
    - 理解线性回归模型；
2. 模型性能验证：
    - 评价函数与目标函数；
    - 交叉验证方法；
    - 绘制学习率曲线；
    - 绘制验证曲线；
3. 嵌入式特征选择：
    - Lasso回归；
    - Ridge回归；
    - 决策树；
4. 模型对比：
    - 常用线性模型；
    - 常用非线性模型；
5. 模型调参：
    - 贪心调参方法；
    - 网格调参方法；
    - 贝叶斯调参方法；


```python
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from warnings import filterwarnings
filterwarnings('ignore')
%matplotlib inline
```


```python
path = './datalab/231784/'
train_data = pd.read_csv(path + 'used_car_train_20200313.csv', sep=' ')
test_data = pd.read_csv(path + 'used_car_testA_20200313.csv', sep=' ')
train_data.shape, test_data.shape
```




    ((150000, 31), (50000, 30))




```python
#train_data.drop(index=train_data.index[train_data['model'].isnull()], inplace=True)
# 测试集没有bodyType缺失，bodyType缺失的的许多都含有其他缺失值，这里删除bodyType缺失的数据，约占总训练数据3%，也可以考虑不删除。 
#train_data.fillna({'bodyType': -1}, inplace=True)
```


```python
# price 进行log转换
ground_y = train_data['price']
train_data['price'] = np.log(train_data['price'] + 1)
```


```python
train_data['price'].describe()
```




    count    150000.000000
    mean          8.035271
    std           1.218222
    min           2.484907
    25%           7.170888
    50%           8.086718
    75%           8.949105
    max          11.512925
    Name: price, dtype: float64



## 构造特征
合并训练集和测试集，并构造特征。


```python
train_data['istrain'] = 1
test_data['istrain'] = 0
train_labels = train_data['price']
data = pd.concat([train_data, test_data], ignore_index=True) # 合并train data, test data, 可一次性处理数据
data.drop(columns=['name', 'offerType', 'seller'], inplace=True) # 删除意义不大的特征
data.loc[:, data.columns != 'price']=data.loc[:, data.columns != 'price'].apply(lambda x:x.fillna(x.value_counts().index[0])) # fill NA with most frequent value
#data.fillna({'bodyType': -1, 'brand': -1, 'gearbox': -1}, inplace=True)
#data['power'][data['power'] > 600] = 0
#data['power'].replace(0, mean_power, inplace=True)
date_features = ['regDate', 'creatDate'] # 日期特征
cat_features = [ 'SaleID', 'bodyType', 'brand', 'fuelType', 'gearbox', 'model', 'notRepairedDamage',  'regionCode'] # 类别特征
num_features = ['v_' + str(i) for i in range(15)]  # power bin instead
cat_to_expand = ['kilometer', 'bodyType', 'brand', 'fuelType', 'gearbox', 'model', 'notRepairedDamage'] # 需要统计数据扩展的特征
data.shape, data.columns
```




    ((200000, 29),
     Index(['SaleID', 'bodyType', 'brand', 'creatDate', 'fuelType', 'gearbox',
            'istrain', 'kilometer', 'model', 'notRepairedDamage', 'power', 'price',
            'regDate', 'regionCode', 'v_0', 'v_1', 'v_10', 'v_11', 'v_12', 'v_13',
            'v_14', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9'],
           dtype='object'))




```python
# power 分桶
data['power'] = np.log(1 + data['power'])
bin = [i * 0.3 for i in range(-1, 21)]
data['power_bin'] = pd.cut(data['power'], bin, labels=False)
data['power_bin'][data['power']  >= 6] = 20
cat_to_expand.append('power_bin')
data[['power_bin', 'power']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>power_bin</th>
      <th>power</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>200000.000000</td>
      <td>200000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>14.954130</td>
      <td>4.351212</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.787278</td>
      <td>1.398243</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15.000000</td>
      <td>4.330733</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>16.000000</td>
      <td>4.709530</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>17.000000</td>
      <td>5.017280</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20.000000</td>
      <td>9.903538</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 增加特征regionCode_0, 1, 可以使用前regionCode_expand_num位，暂不清楚多少位好。
regionCode_expand_num = 1 
for i in range(regionCode_expand_num):
    data['regionCode_' + str(i)] =  data['regionCode'] // 10 ** (3-i)
    #cat_features.append('regionCode_' + str(i))
    cat_to_expand.append('regionCode_' + str(i))
```


```python
# 将notRepairedeDamage中'-'替换为0，并将object类型转为float
data['notRepairedDamage'] = data['notRepairedDamage'].replace("-", 0).astype(np.float) 
```


```python
data['regionCode'].head()
```




    0    1046
    1    4366
    2    2806
    3     434
    4    6977
    Name: regionCode, dtype: int64




```python
# regDate中存在月份为0的现象，不能直接转换，将其改为1月
def date_proc(x):
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]


data['regDate'] = pd.to_datetime(data['regDate'].astype('str').apply(date_proc)) 
data['regDate' + '_year'] = data['regDate'].dt.year # 添加注册年份特征
cat_to_expand.append('regDate_year')
#data[f + '_month'] = data[f].dt.month
#data[f + '_day'] = data[f].dt.day
#data[f + '_dayofweek'] = data[f].dt.dayofweek
```


```python
# # 增加特征使用年份
# data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d') - 
#                             pd.to_datetime(data['regDate'], format='%Y%m%d')).dt.days / 365
# cat_to_expand.append('used_time')
```


```python
#data['used_time'].isnull().sum() # 没有日期异常
```


```python
# 有些特征组合后可能会有更好的效果，下面构造交叉特征。
# def create_cross_feature(df, columns):
#     """"
#     在df中增加交叉特征并返会交叉特征列名
#     param df: pd.Dataframe
#     param columns: 构成交叉特征的多个列名组成的列表
#     """
#     cross_feature_name = "cross_" + '_'.join(list(map(str, columns))) 
#     df[cross_feature_name] = df[columns].apply(tuple, axis=1).apply(lambda x: '_'.join(map(str, x)))
#     return cross_feature_name


# cross_columns = [['regDate_year', 'kilometer'], ['brand', 'bodyType'], ['brand', 'model']]
# for columns in cross_columns:
#     cat_to_expand.append(create_cross_feature(data, columns))
```


```python
cat_to_expand
```




    ['kilometer',
     'bodyType',
     'brand',
     'fuelType',
     'gearbox',
     'model',
     'notRepairedDamage',
     'power_bin',
     'regionCode_0',
     'regDate_year']




```python
# 下面是一个增加类别特的统计特征的函数
```


```python
def cat_stat_expand(train_data, data, columns):
    """
    使用统计数据进行扩展， 这里没有inplace, 可尝试改为inplace
    param train_data: 用于生成统计数据的df
    param data: 需要扩展的df
    param columns： 需要使用统计数据扩展的列
    """
    for col in columns:
        train_gb = train_data.groupby(col)
        all_info = {}
        for kind, kind_data in train_gb:
            info = {}
            #kind_data = kind_data[kind_data['price'] > 0]
            #info[col + 'amount'] = len(kind_data) waste
            info[col + '_price_max'] = kind_data.price.max()
            info[col + '_price_median'] = kind_data.price.median()
            info[col + '_price_min'] = kind_data.price.min()
            #info[col + 'price_sum'] = kind_data.price.sum() waste
            #info[col + 'price_std'] = kind_data.price.std() waste
            #info[col + '_brand_price_average'] = round(kind_data.price.sum() / (len(kind_data) + 1), 2) 
            all_info[kind] = info
        col_fea = pd.DataFrame(all_info).T.reset_index().rename(columns={"index": col})
        data = data.merge(col_fea, how='left', on=col)
    return data 
```


```python
data.columns
```




    Index(['SaleID', 'bodyType', 'brand', 'creatDate', 'fuelType', 'gearbox',
           'istrain', 'kilometer', 'model', 'notRepairedDamage', 'power', 'price',
           'regDate', 'regionCode', 'v_0', 'v_1', 'v_10', 'v_11', 'v_12', 'v_13',
           'v_14', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9',
           'power_bin', 'regionCode_0', 'regDate_year'],
          dtype='object')




```python
data = cat_stat_expand(data[data['istrain']==1], data, cat_to_expand) # 特征扩展
```


```python
data['istrain'].isnull().sum().sum() # 训练集无NA
```




    0



###  嵌入式特征筛选


```python
# 连续型特征
continuous_feature_names = [col for col in data.columns if col not in date_features + cat_to_expand + ['price', 'istrain', 'power'] + cat_features ]
#continuous_feature_names 
```


```python
#分离训练集X， y
train_X = data[continuous_feature_names][data.istrain == 1]
train_y = data['price'][data.istrain == 1]
```


```python
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(train_X, train_y)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)




```python
'intercept:'+ str(model.intercept_)

sorted(dict(zip(continuous_feature_names, model.coef_)).items(), key=lambda x:x[1], reverse=True)
```




    [('v_1', 1602804.6174257451),
     ('v_10', 1334123.0587816942),
     ('v_2', 600163.4234153582),
     ('v_0', 65774.89846862372),
     ('v_4', 447.86492500976493),
     ('v_14', 12.767938564038232),
     ('v_9', 2.2915726545736237),
     ('bodyType_price_max', 0.08319580797378033),
     ('kilometer_price_median', 0.08228286343901796),
     ('power_bin_price_median', 0.0816457899820998),
     ('bodyType_price_min', 0.054324569980291795),
     ('regionCode_0_price_max', 0.03812009334639739),
     ('fuelType_price_median', 0.02340883239202859),
     ('power_bin_price_max', 0.005736748987208238),
     ('power_bin_price_min', 0.005335229381007327),
     ('brand_price_min', 0.0028036956085898393),
     ('model_price_min', 0.0008142775751114093),
     ('brand_price_max', -0.006128467576428569),
     ('regDate_year_price_min', -0.006733448425044843),
     ('regDate_year_price_max', -0.01257942772633261),
     ('regionCode_0_price_median', -0.01521979701757619),
     ('model_price_max', -0.018834399915805364),
     ('kilometer_price_min', -0.02022816441830568),
     ('regionCode_0_price_min', -0.020859782507565667),
     ('fuelType_price_min', -0.02748945073042357),
     ('bodyType_price_median', -0.0279803546342002),
     ('regDate_year_price_median', -0.07504962562832902),
     ('fuelType_price_max', -0.09208468672955004),
     ('model_price_median', -0.1025636173668147),
     ('brand_price_median', -0.1345306171935501),
     ('kilometer_price_max', -0.3117292819647402),
     ('v_7', -2.6529852850718574),
     ('notRepairedDamage_price_max', -7.3071582317352295),
     ('v_5', -16.709132738649195),
     ('v_8', -34.33758719377712),
     ('notRepairedDamage_price_median', -44.042748274477674),
     ('v_6', -148.6977423080628),
     ('notRepairedDamage_price_min', -707.251827016516),
     ('v_13', -1034.7655947727287),
     ('gearbox_price_median', -3627.2919623311427),
     ('gearbox_price_min', -24462.563505896458),
     ('v_3', -34123.18024844656),
     ('v_12', -213060.05315039656),
     ('v_11', -1053272.4545566859),
     ('gearbox_price_max', -7591617.513524128)]




```python
subsample_index = np.random.randint(low=0, high=len(train_y), size=50)
```


```python
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error,  make_scorer
```


```python
scores = cross_val_score(model, X=train_X, y=train_y, verbose=1, cv = 5, scoring=make_scorer(mean_absolute_error))
```

    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done   5 out of   5 | elapsed:    0.8s finished
    


```python
scores = pd.DataFrame(scores.reshape(1,-1))
scores.columns = ['cv' + str(x) for x in range(1, 6)]
scores.index = ['MAE']
scores
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cv1</th>
      <th>cv2</th>
      <th>cv3</th>
      <th>cv4</th>
      <th>cv5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MAE</th>
      <td>0.18874</td>
      <td>0.190103</td>
      <td>0.261304</td>
      <td>0.85072</td>
      <td>0.191253</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import learning_curve, validation_curve
```


```python
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,n_jobs=1, train_size=np.linspace(.1, 1.0, 5 )):  
    plt.figure()  
    plt.title(title)  
    if ylim is not None:  
        plt.ylim(*ylim)  
    plt.xlabel('Training example')  
    plt.ylabel('score')  
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_size, scoring = make_scorer(mean_absolute_error))  
    train_scores_mean = np.mean(train_scores, axis=1)  
    train_scores_std = np.std(train_scores, axis=1)  
    test_scores_mean = np.mean(test_scores, axis=1)  
    test_scores_std = np.std(test_scores, axis=1)  
    plt.grid()#区域  
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,  
                     train_scores_mean + train_scores_std, alpha=0.1,  
                     color="r")  
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,  
                     test_scores_mean + test_scores_std, alpha=0.1,  
                     color="g")  
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r',  
             label="Training score")  
    plt.plot(train_sizes, test_scores_mean,'o-',color="g",  
             label="Cross-validation score")  
    plt.legend(loc="best")  
    return plt  
```


```python
# 绘制学习曲线
plot_learning_curve(LinearRegression(), 'Liner_model', train_X[:1000], train_y[:1000], ylim=(0.0, 0.5), cv=5, n_jobs=1)  
```


```python
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
```


```python
train_y_ln = train_y
```


```python
models = [LinearRegression(),
          Ridge(),
          Lasso()]
```


```python
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')
```

    LinearRegression is finished
    Ridge is finished
    Lasso is finished
    


```python
result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LinearRegression</th>
      <th>Ridge</th>
      <th>Lasso</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cv1</th>
      <td>0.188738</td>
      <td>0.189802</td>
      <td>0.478972</td>
    </tr>
    <tr>
      <th>cv2</th>
      <td>0.190103</td>
      <td>0.191522</td>
      <td>0.476315</td>
    </tr>
    <tr>
      <th>cv3</th>
      <td>0.190732</td>
      <td>0.191847</td>
      <td>0.480589</td>
    </tr>
    <tr>
      <th>cv4</th>
      <td>0.186877</td>
      <td>0.188103</td>
      <td>0.469473</td>
    </tr>
    <tr>
      <th>cv5</th>
      <td>0.191253</td>
      <td>0.192687</td>
      <td>0.477174</td>
    </tr>
  </tbody>
</table>
</div>




```python
model = LinearRegression().fit(train_X, train_y_ln)
print(abs(model.coef_))
#sns.barplot(abs(model.coef_), continuous_feature_names)
```

    [6.57749003e+04 1.60280472e+06 1.33412314e+06 1.05327253e+06
     2.13060074e+05 1.03476588e+03 1.27679431e+01 6.00163472e+05
     3.41231848e+04 4.47865064e+02 1.67094574e+01 1.48698183e+02
     2.65304603e+00 3.43375406e+01 2.29147853e+00 3.11729401e-01
     8.22834993e-02 2.02288809e-02 8.31957509e-02 2.79802882e-02
     5.43247690e-02 6.12857122e-03 1.34530538e-01 2.80367937e-03
     9.20845194e-02 2.34089639e-02 2.74893429e-02 5.17948067e+09
     2.34863339e+08 1.60778181e+09 1.88342543e-02 1.02563427e-01
     8.14237395e-04 9.33936510e+06 1.40337321e+08 2.24457093e+09
     5.73758291e-03 8.16449445e-02 5.33608127e-03 3.81194539e-02
     1.52205154e-02 2.08593361e-02 1.25794504e-02 7.50498813e-02
     6.73343419e-03]
    


```python
model = Ridge().fit(train_X, train_y_ln)
print(abs(model.coef_))
#sns.barplot(abs(model.coef_), continuous_feature_names)
```

    [4.33124441e-01 6.52065939e-02 2.44382029e-01 3.87755513e-01
     5.90792807e-01 5.84708651e-01 1.09216731e-01 6.17172439e-02
     1.10349933e-01 4.43652379e-01 3.55281794e-01 8.96036357e-01
     1.09035085e+00 2.15039799e+00 2.75570743e+00 3.02620692e-01
     7.67768540e-02 2.42119182e-02 7.73722939e-02 4.62921342e-02
     5.00261690e-02 3.43756162e-02 2.52851445e-02 1.11243165e-02
     1.17032658e-01 2.17131874e-02 3.50440635e-02 1.25224664e-05
     1.30651028e-02 1.94887983e-03 6.05203110e-02 1.27197962e-01
     3.68513805e-03 0.00000000e+00 1.76670674e-01 1.10459816e-02
     6.75845078e-04 7.69496044e-02 1.59004315e-03 4.77769026e-02
     2.76465407e-02 2.38516555e-02 1.48594458e-02 6.02434521e-02
     6.84340979e-03]
    


```python
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
```


```python
models = [LinearRegression(),
          DecisionTreeRegressor(),
          RandomForestRegressor(),
          GradientBoostingRegressor(),
          MLPRegressor(solver='lbfgs', max_iter=100), 
          XGBRegressor(n_estimators = 100, objective='reg:squarederror'), 
          LGBMRegressor(n_estimators = 100)]
```


```python
result = dict()
for model in models:
    model_name = str(model).split('(')[0]
    scores = cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error))
    result[model_name] = scores
    print(model_name + ' is finished')
```

    LinearRegression is finished
    DecisionTreeRegressor is finished
    RandomForestRegressor is finished
    GradientBoostingRegressor is finished
    MLPRegressor is finished
    XGBRegressor is finished
    LGBMRegressor is finished
    


```python
result = pd.DataFrame(result)
result.index = ['cv' + str(x) for x in range(1, 6)]
result
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>LinearRegression</th>
      <th>DecisionTreeRegressor</th>
      <th>RandomForestRegressor</th>
      <th>GradientBoostingRegressor</th>
      <th>MLPRegressor</th>
      <th>XGBRegressor</th>
      <th>LGBMRegressor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cv1</th>
      <td>0.188738</td>
      <td>0.188086</td>
      <td>0.139013</td>
      <td>0.175457</td>
      <td>0.186808</td>
      <td>0.138239</td>
      <td>0.143722</td>
    </tr>
    <tr>
      <th>cv2</th>
      <td>0.190103</td>
      <td>0.186843</td>
      <td>0.138613</td>
      <td>0.178688</td>
      <td>0.194973</td>
      <td>0.139259</td>
      <td>0.146229</td>
    </tr>
    <tr>
      <th>cv3</th>
      <td>0.190732</td>
      <td>0.189092</td>
      <td>0.140513</td>
      <td>0.178153</td>
      <td>0.189129</td>
      <td>0.139165</td>
      <td>0.145907</td>
    </tr>
    <tr>
      <th>cv4</th>
      <td>0.186877</td>
      <td>0.186656</td>
      <td>0.138176</td>
      <td>0.175579</td>
      <td>0.181161</td>
      <td>0.137077</td>
      <td>0.143593</td>
    </tr>
    <tr>
      <th>cv5</th>
      <td>0.191253</td>
      <td>0.187236</td>
      <td>0.140035</td>
      <td>0.178644</td>
      <td>0.190773</td>
      <td>0.139791</td>
      <td>0.144944</td>
    </tr>
  </tbody>
</table>
</div>




```python
objective = ['regression', 'regression_l1', 'mape', 'huber', 'fair']

num_leaves = [3,5,10,15,20,40, 55]
max_depth = [3,5,10,15,20,40, 55]
bagging_fraction = []
feature_fraction = []
drop_rate = []
```

### 贪心调参


```python
best_obj = dict()
for obj in objective:
    model = LGBMRegressor(objective=obj)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_obj[obj] = score
    
best_leaves = dict()
for leaves in num_leaves:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0], num_leaves=leaves)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_leaves[leaves] = score
    
best_depth = dict()
for depth in max_depth:
    model = LGBMRegressor(objective=min(best_obj.items(), key=lambda x:x[1])[0],
                          num_leaves=min(best_leaves.items(), key=lambda x:x[1])[0],
                          max_depth=depth)
    score = np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
    best_depth[depth] = score
```


```python
#sns.lineplot(x=['0_initial','1_turning_obj','2_turning_leaves','3_turning_depth'], y=[0.143 ,min(best_obj.values()), min(best_leaves.values()), min(best_depth.values())])
```

### Grid Search 调参


```python
from sklearn.model_selection import GridSearchCV
```


```python
parameters = {'objective': objective , 'num_leaves': num_leaves, 'max_depth': max_depth}
model = LGBMRegressor()
clf = GridSearchCV(model, parameters, cv=5)
clf = clf.fit(train_X, train_y)
```


```python
clf.best_params_
```




    {'max_depth': 15, 'num_leaves': 55, 'objective': 'huber'}




```python
model = LGBMRegressor(objective='regression',
                          num_leaves=55,
                          max_depth=15)
```


```python
np.mean(cross_val_score(model, X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)))
```




    0.13831341952745607



### 贝叶斯调参


```python
from bayes_opt import BayesianOptimization
```


```python
def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective = 'regression_l1',
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample = subsample,
            min_child_samples = int(min_child_samples)
        ),
        X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val
```


```python
def rf_cv(num_leaves, max_depth, subsample, min_child_samples):
    val = cross_val_score(
        LGBMRegressor(objective = 'regression_l1',
            num_leaves=int(num_leaves),
            max_depth=int(max_depth),
            subsample = subsample,
            min_child_samples = int(min_child_samples)
        ),
        X=train_X, y=train_y_ln, verbose=0, cv = 5, scoring=make_scorer(mean_absolute_error)
    ).mean()
    return 1 - val
```


```python
rf_bo = BayesianOptimization(
    rf_cv,
    {
    'num_leaves': (2, 100),
    'max_depth': (2, 100),
    'subsample': (0.1, 1),
    'min_child_samples' : (2, 100)
    }
)
```


```python
rf_bo.maximize()
```

    |   iter    |  target   | max_depth | min_ch... | num_le... | subsample |
    -------------------------------------------------------------------------
    | [0m 1       [0m | [0m 0.8682  [0m | [0m 87.25   [0m | [0m 66.72   [0m | [0m 84.19   [0m | [0m 0.5586  [0m |
    | [0m 2       [0m | [0m 0.8434  [0m | [0m 60.49   [0m | [0m 43.07   [0m | [0m 14.45   [0m | [0m 0.8558  [0m |
    | [0m 3       [0m | [0m 0.86    [0m | [0m 71.6    [0m | [0m 69.63   [0m | [0m 41.48   [0m | [0m 0.7334  [0m |
    | [0m 4       [0m | [0m 0.8654  [0m | [0m 10.23   [0m | [0m 2.936   [0m | [0m 65.33   [0m | [0m 0.8168  [0m |
    | [0m 5       [0m | [0m 0.8334  [0m | [0m 25.86   [0m | [0m 28.14   [0m | [0m 9.108   [0m | [0m 0.3843  [0m |
    | [0m 6       [0m | [0m 0.8002  [0m | [0m 2.166   [0m | [0m 99.41   [0m | [0m 97.46   [0m | [0m 0.747   [0m |
    | [95m 7       [0m | [95m 0.8694  [0m | [95m 99.31   [0m | [95m 3.519   [0m | [95m 98.7    [0m | [95m 0.5388  [0m |
    | [0m 8       [0m | [0m 0.7949  [0m | [0m 98.62   [0m | [0m 97.28   [0m | [0m 3.367   [0m | [0m 0.8184  [0m |
    | [95m 9       [0m | [95m 0.8695  [0m | [95m 50.37   [0m | [95m 6.424   [0m | [95m 99.47   [0m | [95m 0.8969  [0m |
    | [0m 10      [0m | [0m 0.8641  [0m | [0m 79.94   [0m | [0m 2.469   [0m | [0m 56.87   [0m | [0m 0.4728  [0m |
    | [0m 11      [0m | [0m 0.8694  [0m | [0m 99.61   [0m | [0m 6.932   [0m | [0m 98.11   [0m | [0m 0.2403  [0m |
    | [0m 12      [0m | [0m 0.866   [0m | [0m 50.2    [0m | [0m 36.85   [0m | [0m 69.78   [0m | [0m 0.9439  [0m |
    | [0m 13      [0m | [0m 0.8658  [0m | [0m 98.95   [0m | [0m 44.37   [0m | [0m 67.32   [0m | [0m 0.66    [0m |
    | [0m 14      [0m | [0m 0.869   [0m | [0m 95.07   [0m | [0m 99.45   [0m | [0m 98.96   [0m | [0m 0.985   [0m |
    | [0m 15      [0m | [0m 0.866   [0m | [0m 46.1    [0m | [0m 3.069   [0m | [0m 68.55   [0m | [0m 0.9624  [0m |
    | [0m 16      [0m | [0m 0.8682  [0m | [0m 9.878   [0m | [0m 2.722   [0m | [0m 99.37   [0m | [0m 0.9156  [0m |
    | [95m 17      [0m | [95m 0.8696  [0m | [95m 97.75   [0m | [95m 63.39   [0m | [95m 99.55   [0m | [95m 0.2375  [0m |
    | [0m 18      [0m | [0m 0.8676  [0m | [0m 99.73   [0m | [0m 95.53   [0m | [0m 81.77   [0m | [0m 0.4033  [0m |
    | [0m 19      [0m | [0m 0.869   [0m | [0m 70.0    [0m | [0m 3.152   [0m | [0m 93.24   [0m | [0m 0.1712  [0m |
    | [0m 20      [0m | [0m 0.8695  [0m | [0m 88.83   [0m | [0m 26.06   [0m | [0m 99.54   [0m | [0m 0.9432  [0m |
    | [95m 21      [0m | [95m 0.8697  [0m | [95m 99.62   [0m | [95m 70.52   [0m | [95m 99.29   [0m | [95m 0.8384  [0m |
    | [0m 22      [0m | [0m 0.8691  [0m | [0m 99.69   [0m | [0m 97.85   [0m | [0m 96.23   [0m | [0m 0.1804  [0m |
    | [0m 23      [0m | [0m 0.8695  [0m | [0m 78.26   [0m | [0m 2.854   [0m | [0m 99.19   [0m | [0m 0.9562  [0m |
    | [0m 24      [0m | [0m 0.8694  [0m | [0m 32.09   [0m | [0m 2.491   [0m | [0m 98.36   [0m | [0m 0.8091  [0m |
    | [0m 25      [0m | [0m 0.8694  [0m | [0m 98.72   [0m | [0m 87.37   [0m | [0m 99.2    [0m | [0m 0.8725  [0m |
    | [0m 26      [0m | [0m 0.8692  [0m | [0m 99.99   [0m | [0m 26.85   [0m | [0m 96.95   [0m | [0m 0.9901  [0m |
    | [0m 27      [0m | [0m 0.8695  [0m | [0m 96.89   [0m | [0m 32.06   [0m | [0m 99.72   [0m | [0m 0.1178  [0m |
    | [0m 28      [0m | [0m 0.8694  [0m | [0m 33.31   [0m | [0m 2.295   [0m | [0m 98.36   [0m | [0m 0.3201  [0m |
    | [0m 29      [0m | [0m 0.8677  [0m | [0m 76.59   [0m | [0m 35.35   [0m | [0m 80.15   [0m | [0m 0.1253  [0m |
    | [0m 30      [0m | [0m 0.8695  [0m | [0m 55.76   [0m | [0m 2.608   [0m | [0m 99.29   [0m | [0m 0.2271  [0m |
    =========================================================================
    


```python
1 - rf_bo.max['target']
```




    0.13027746444666166




```python

```
