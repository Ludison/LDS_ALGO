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
train = pd.read_csv(r'D:\pythonPro\digit\data\train.csv')
tests = pd.read_csv(r'D:\pythonPro\digit\data\test.csv')
X=train.drop(['label'],axis=1)

Y=train.label

print(X.shape)
print(Y.shape)

def modelfit(alg, X, Y, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X.values, label=Y.values)
        print(xgtrain)
        print(alg.get_params()['n_estimators'])
        cvresult = xgb.cv(xgb_param,  # Booster params.
                          xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],  # 迭代次数
                          nfold=cv_folds,  # 交叉验证中折叠的次数
                          metrics='auc',  # 评估指标
                          early_stopping_rounds=early_stopping_rounds,  # 需要在每一轮Early_stopping_rounds中至少改善一次以继续训练
                          verbose_eval=None # 是否显示进度。 如果为None，则返回np.ndarray时将显示进度。 如果为True，则进度将在增强阶段显示。 如果给定一个整数，则将在每个给定的verbose_eval提升阶段显示进度。
                          )
        print(cvresult)
        alg.set_params(n_estimators=cvresult.shape[0])

    # Fit the algorithm on the data
    alg.fit(X, Y, eval_metric='auc')

    # Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:, 1]

    # Print model report:
    print("Model Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(Y.values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Y, dtrain_predprob))

    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')



xgb1 = XGBClassifier(
 learning_rate =0.1,  #学习率
 n_estimators=1000,   #基分类器数量
 max_depth=5,         #最大深度
 min_child_weight=1,  #最小叶子节点样本权重之和
 gamma=0,             #Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守
 subsample=0.8,       #这个参数控制对于每棵树，随机采样的比例。 减小这个参数的值，算法会更加保守，避免过拟合。
 colsample_bytree=0.8,#和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。
 objective= 'multi:softmax', #这个参数定义需要被最小化的损失函数
 num_class=10,      # 类别数，与 multisoftmax 并用
 n_jobs=4,           #用于运行xgboost的并行线程数。
 scale_pos_weight=1,  #在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
 seed=27)

modelfit(xgb1,X,Y)
