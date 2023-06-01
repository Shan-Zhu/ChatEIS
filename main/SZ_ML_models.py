#coding: utf-8


from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import f1_score

# input data
data_input=pd.read_csv('eis_all_data.csv', sep=',')

labels=data_input['label']
features=data_input.drop('label', axis=1)

X_train,X_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=0)

# xgboost
xgbc_model=XGBClassifier()
# 随机森林
rfc_model=RandomForestClassifier()
# ET
et_model=ExtraTreesClassifier()
# 朴素贝叶斯
gnb_model=GaussianNB()
#K最近邻
knn_model=KNeighborsClassifier()
#逻辑回归
lr_model=LogisticRegression()
#决策树
dt_model=DecisionTreeClassifier()
#支持向量机
svc_model=SVC()


# xgboost
xgbc_model.fit(X_train, y_train)
# 随机森林
rfc_model.fit(X_train,y_train)
# ET
et_model.fit(X_train,y_train)
# 朴素贝叶斯
gnb_model.fit(X_train,y_train)
# K最近邻
knn_model.fit(X_train,y_train)
# 逻辑回归
lr_model.fit(X_train,y_train)
# 决策树
dt_model.fit(X_train,y_train)
# 支持向量机
svc_model.fit(X_train,y_train)


print("\tXGBoost：",xgbc_model.score(X_test, y_test))
print("\tRandomForest：",rfc_model.score(X_test, y_test))
print("\tExtraTrees：",et_model.score(X_test, y_test))
print("\tGaussian：",gnb_model.score(X_test, y_test))
print("\tKNeighbors：",knn_model.score(X_test, y_test))
print("\tLogisticRegression：",lr_model.score(X_test, y_test))
print("\tDecisionTree：",dt_model.score(X_test, y_test))
print("\tSVC：",svc_model.score(X_test, y_test))

'''
strKFold = KFold(n_splits=5,shuffle=True,random_state=0) 

print("\n使用５折交叉验证方法得的准确率（每次迭代的准确率的均值）：")
print("\tXGBoost模型：",cross_val_score(xgbc_model,features,labels,cv=strKFold).mean())
print("\t随机森林模型：",cross_val_score(rfc_model,features,labels,cv=strKFold).mean())
print("\tET模型：",cross_val_score(et_model,features,labels,cv=strKFold).mean())
print("\t高斯朴素贝叶斯模型：",cross_val_score(gnb_model,features,labels,cv=strKFold).mean())
print("\tK最近邻模型：",cross_val_score(knn_model,features,labels,cv=strKFold).mean())
print("\t逻辑回归：",cross_val_score(lr_model,features,labels,cv=strKFold).mean())
print("\t决策树：",cross_val_score(dt_model,features,labels,cv=strKFold).mean())
print("\t支持向量机：",cross_val_score(svc_model,features,labels,cv=strKFold).mean())
'''
