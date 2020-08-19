import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import GridSearchCV,train_test_split

import matplotlib.pylab as plt

from matplotlib.pylab import rcParams

#记录程序运行时间
import time
start_time = time.time()

#读入数据
train = pd.read_csv(r'D:\pythonPro\LDS_ALGO\data\xgboost_test\train_modified.csv')
target = 'Disbursed'
IDcol = 'ID'


def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds,verbose_eval=10, show_stdv=True
                          ,stratified=False,seed=1)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


predictors = [x for x in train.columns if x not in [target, IDcol]]
print(predictors)
# xgb1 = XGBClassifier(
#  learning_rate =0.1,  #学习率
#  n_estimators=1000,   #基分类器数量
#  max_depth=5,         #最大深度
#  min_child_weight=1,  #最小叶子节点样本权重之和
#  gamma=0,             #Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守
#  subsample=0.8,       #这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。
#  colsample_bytree=0.8,#和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。
#  objective= 'multi:softmax', #这个参数定义需要被最小化的损失函数
#  num_class=10,      # 类别数，与 multisoftmax 并用
#  n_jobs=4,           #用于运行xgboost的并行线程数。
#  scale_pos_weight=1,  #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
#  seed=27)

xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

modelfit(xgb1, train, predictors)
