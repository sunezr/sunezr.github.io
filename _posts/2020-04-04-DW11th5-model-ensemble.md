
# Datawhale 零基础入门数据挖掘-Task5 模型融合

Tip:此部分为零基础入门数据挖掘的 Task5 模型融合 部分， 以baseline中lgb和xgb为例进行融合。

**赛题：零基础入门数据挖掘 - 二手车交易价格预测**

地址：https://tianchi.aliyun.com/competition/entrance/231784/introduction?spm=5176.12281957.1004.1.38b02448ausjSX 


```python
## 基础工具
import numpy as np
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import jn
from IPython.display import display, clear_output
import time

warnings.filterwarnings('ignore')
%matplotlib inline

## 模型预测的
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

## 数据降维处理的
from sklearn.decomposition import PCA,FastICA,FactorAnalysis,SparsePCA

import lightgbm as lgb
import xgboost as xgb

## 参数搜索和评价的
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
```

### 数据读取


```python
## 通过Pandas对于数据进行读取 (pandas是一个很友好的数据读取函数库)
train_data = pd.read_csv('datalab/231784/used_car_train_20200313.csv', sep=' ')
testA_data = pd.read_csv('datalab/231784/used_car_testA_20200313.csv', sep=' ')

## 输出数据的大小信息
print('train data shape:',train_data.shape)
print('testA data shape:',testA_data.shape)
```

    train data shape: (150000, 31)
    testA data shape: (50000, 30)
    

### 特征与标签构建

#### 1) 提取数值类型特征列名


```python
numerical_cols = train_data.select_dtypes(exclude = 'object').columns
print(numerical_cols)
```

    Index(['SaleID', 'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType',
           'gearbox', 'power', 'kilometer', 'regionCode', 'seller', 'offerType',
           'creatDate', 'price', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6',
           'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13', 'v_14'],
          dtype='object')
    


```python
categorical_cols = train_data.select_dtypes(include = 'object').columns
print(categorical_cols)
```

    Index(['notRepairedDamage'], dtype='object')
    

#### 2) 构建训练和测试样本


```python
# feature engineer
```


```python
train_data["istrain"] = 1
testA_data["istrain"] = 0
```


```python
data = pd.concat([train_data, testA_data])
```


```python
data["regionCode"] //= 1000
```


```python
data['price'] = np.log(data['price'] + 1)
```


```python
data["power"] = np.log(data["power"]  + 1)
```


```python
data["notRepairedDamage"].replace("-", "0.0", inplace=True)
```


```python
def date_proc(x):
    m = int(x[4:6])
    if m == 0:
        m = 1
    return x[:4] + '-' + str(m) + '-' + x[6:]


data['regDate'] = pd.to_datetime(data['regDate'].astype('str').apply(date_proc)) 
data['used_time'] = (pd.to_datetime(data['creatDate'], format='%Y%m%d', errors='coerce') - 
                            pd.to_datetime(data['regDate'], format='%Y%m%d', errors='coerce')).dt.days / 365
```


```python
categorical_cols = ['bodyType', 'brand', 'fuelType', 'gearbox','model', 'notRepairedDamage', 'regionCode']
data[categorical_cols].astype(object)
print(categorical_cols)
```

    ['bodyType', 'brand', 'fuelType', 'gearbox', 'model', 'notRepairedDamage', 'regionCode']
    


```python
non_numerical_cols = categorical_cols + ['price', 'SaleID', 'creatDate', 'name', 'offerType', 'regDate',  'seller', "istrain"]
```


```python
data = pd.get_dummies(data, columns=categorical_cols, dummy_na=True)
```


```python
numerical_cols = [col for col in data.columns if col not in non_numerical_cols]
```


```python
train_data = data[data["istrain"]==1].drop(columns=["istrain"])
testA_data = data[data["istrain"]==0].drop(columns=["istrain"])
```


```python
# end feature engineer
```


```python
## 选择特征列
feature_cols = numerical_cols + categorical_cols

## 提前特征列，标签列构造训练样本和测试样本
X_data = train_data[numerical_cols]
Y_data = train_data['price']
X_test  = testA_data[numerical_cols]

print('X train shape:',X_data.shape)
print('X test shape:',X_test.shape)
```

    X train shape: (150000, 341)
    X test shape: (50000, 341)
    


```python
## 定义了一个统计函数，方便后续信息统计
def Sta_inf(data):
    print('_min',np.min(data))
    print('_max:',np.max(data))
    print('_mean',np.mean(data))
    print('_ptp',np.ptp(data))
    print('_std',np.std(data))
    print('_var',np.var(data))
```

####  统计标签的基本分布信息


```python
print('Sta of label:')
Sta_inf(Y_data)
```

    Sta of label:
    _min 2.4849066497880004
    _max: 11.512925464970229
    _mean 8.035270518212984
    _ptp 9.028018815182229
    _std 1.218218381924893
    _var 1.4840560260597047
    

### 模型训练与预测

####  定义xgb和lgb模型函数


```python
def build_model_xgb(x_train,y_train):
    model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.1, gamma=0, subsample=0.8,\
        colsample_bytree=0.9, max_depth=9) #, objective ='reg:squarederror'
    model.fit(x_train, y_train)
    return model

def build_model_lgb(x_train,y_train):
    estimator = lgb.LGBMRegressor(num_leaves=127,n_estimators = 150)
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(x_train, y_train)
    return gbm
```

#### 切分数据集（train,Val）进行模型训练，评价和预测


```python
## Split data with val
x_train,x_val,y_train,y_val = train_test_split(X_data,Y_data,test_size=0.3)
```


```python
x_train.head()
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
      <th>kilometer</th>
      <th>power</th>
      <th>v_0</th>
      <th>v_1</th>
      <th>v_10</th>
      <th>v_11</th>
      <th>v_12</th>
      <th>v_13</th>
      <th>v_14</th>
      <th>v_2</th>
      <th>...</th>
      <th>regionCode_0.0</th>
      <th>regionCode_1.0</th>
      <th>regionCode_2.0</th>
      <th>regionCode_3.0</th>
      <th>regionCode_4.0</th>
      <th>regionCode_5.0</th>
      <th>regionCode_6.0</th>
      <th>regionCode_7.0</th>
      <th>regionCode_8.0</th>
      <th>regionCode_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>134838</th>
      <td>15.0</td>
      <td>4.330733</td>
      <td>41.268476</td>
      <td>-3.129800</td>
      <td>3.924021</td>
      <td>-0.327354</td>
      <td>-4.215612</td>
      <td>-0.066681</td>
      <td>-0.285391</td>
      <td>-1.972728</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43315</th>
      <td>12.5</td>
      <td>5.081404</td>
      <td>47.439555</td>
      <td>4.999855</td>
      <td>-5.702076</td>
      <td>0.810396</td>
      <td>2.650854</td>
      <td>-0.857724</td>
      <td>1.048630</td>
      <td>1.180750</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>111452</th>
      <td>15.0</td>
      <td>4.867534</td>
      <td>45.312770</td>
      <td>5.254413</td>
      <td>-4.965099</td>
      <td>1.914855</td>
      <td>-1.324986</td>
      <td>-1.514956</td>
      <td>-0.441473</td>
      <td>-0.183836</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>105975</th>
      <td>15.0</td>
      <td>4.356709</td>
      <td>45.666190</td>
      <td>-3.132297</td>
      <td>1.863644</td>
      <td>-2.895368</td>
      <td>1.987538</td>
      <td>-1.358423</td>
      <td>-0.088960</td>
      <td>-0.461548</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>83289</th>
      <td>1.0</td>
      <td>4.709530</td>
      <td>47.995269</td>
      <td>-3.274520</td>
      <td>1.307424</td>
      <td>-3.232975</td>
      <td>4.954365</td>
      <td>0.335077</td>
      <td>0.706583</td>
      <td>1.269067</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 341 columns</p>
</div>




```python
x_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 105000 entries, 134838 to 115541
    Columns: 341 entries, kilometer to regionCode_nan
    dtypes: float64(18), uint8(323)
    memory usage: 47.6 MB
    


```python
print('train lgb...')
model_lgb = build_model_lgb(x_train,y_train)
val_lgb = model_lgb.predict(x_val)
MAE_lgb = mean_absolute_error(np.exp(y_val),np.exp(val_lgb))
print('MAE of val with lgb:',MAE_lgb)

print('Predict lgb...')
model_lgb_pre = build_model_lgb(X_data,Y_data)
subA_lgb = model_lgb_pre.predict(X_test)
print('Sta of Predict lgb:')
Sta_inf(subA_lgb)
```

    train lgb...
    MAE of val with lgb: 587.4938168255009
    Predict lgb...
    Sta of Predict lgb:
    _min 2.873412071732873
    _max: 11.413815527508927
    _mean 8.034187970719984
    _ptp 8.540403455776055
    _std 1.199772654199168
    _var 1.4394544217641163
    


```python
print('train xgb...')
model_xgb = build_model_xgb(x_train,y_train)
val_xgb = model_xgb.predict(x_val)
MAE_xgb = mean_absolute_error(np.exp(y_val),np.exp(val_xgb))
print('MAE of val with xgb:', MAE_xgb)

print('Predict xgb...')
model_xgb_pre = build_model_xgb(X_data,Y_data)
subA_xgb = model_xgb_pre.predict(X_test)
print('Sta of Predict xgb:')
Sta_inf(subA_xgb)
```

    train xgb...
    MAE of val with xgb: 591.1719637145573
    Predict xgb...
    Sta of Predict xgb:
    _min 2.6981957
    _max: 11.401571
    _mean 8.034155
    _ptp 8.703376
    _std 1.1991446
    _var 1.4379479
    

#### 进行两模型的结果加权融合


```python
## 这里我们采取了简单的加权融合的方式
val_Weighted = (1-MAE_lgb/(MAE_xgb+MAE_lgb))*val_lgb+(1-MAE_xgb/(MAE_xgb+MAE_lgb))*val_xgb
print('MAE of val with Weighted ensemble:', mean_absolute_error(np.exp(y_val), np.exp(val_Weighted)))
```

    MAE of val with Weighted ensemble: 573.7765587035032
    


```python
可以发现，加权融合结果比单独模型的结果好， 因为组合模型减小了误差中的方差部分。
```
