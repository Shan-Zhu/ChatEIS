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
data_input=pd.read_csv('eis_all_data-20.csv', sep=',')

labels=data_input['label']
features=data_input.drop('label', axis=1)

X_train,X_test,y_train,y_test=train_test_split(features, labels, test_size=0.2, random_state=0)

xgbc_model=XGBClassifier()
rfc_model=RandomForestClassifier()
et_model=ExtraTreesClassifier()
gnb_model=GaussianNB()
knn_model=KNeighborsClassifier()
lr_model=LogisticRegression()
dt_model=DecisionTreeClassifier()
svc_model=SVC()


xgbc_model.fit(X_train, y_train)
rfc_model.fit(X_train,y_train)
et_model.fit(X_train,y_train)
gnb_model.fit(X_train,y_train)
knn_model.fit(X_train,y_train)
lr_model.fit(X_train,y_train)
dt_model.fit(X_train,y_train)
svc_model.fit(X_train,y_train)


strKFold = KFold(n_splits=5,shuffle=True,random_state=0) 

print("\nK=5 validation:")
print(cross_val_score(xgbc_model,features,labels,cv=strKFold).mean())
print(cross_val_score(rfc_model,features,labels,cv=strKFold).mean())
print(cross_val_score(et_model,features,labels,cv=strKFold).mean())
print(cross_val_score(gnb_model,features,labels,cv=strKFold).mean())
print(cross_val_score(knn_model,features,labels,cv=strKFold).mean())
print(cross_val_score(lr_model,features,labels,cv=strKFold).mean())
print(cross_val_score(dt_model,features,labels,cv=strKFold).mean())
print(cross_val_score(svc_model,features,labels,cv=strKFold).mean())
