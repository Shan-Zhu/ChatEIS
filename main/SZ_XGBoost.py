#coding: utf-8

import numpy as np
import pandas as pd
import sklearn as sk
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import f1_score,accuracy_score

data_input = pd.read_csv('eis_all_data.csv', sep=',')

labels = data_input['label']
features = data_input.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

XGB = XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=500)

# XGB.fit(X_train, y_train)
# XGB.save_model('XGB_model.model')

# 加载模型
loaded_model = XGBClassifier()
loaded_model.load_model('XGB_model.model')

y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# 在测试集上进行预测并计算准确率
y_pred_prob = loaded_model.predict_proba(X_test)
# 获取概率最高的三个类别的索引和概率值
top_3_indices = y_pred_prob.argsort()[:, -3:][:, ::-1]
top_3_labels = [loaded_model.classes_[i] for i in top_3_indices]
top_3_probs = [y_pred_prob[j][i] for j, i in enumerate(top_3_indices)]
# 将结果存储在 Pandas DataFrame 中
output = pd.DataFrame({'y_test': y_test, 'y_pred':y_pred, 'y_pred_top3': top_3_labels})
# 检验正确结果是否存在于这三个结果之中，如果存在，将“1”存入“对比”列，如果不存在，将“0”存入“对比”列
output['Top3_results'] = output.apply(lambda row: 1 if row['y_test'] in row['y_pred_top3'] else 0, axis=1)

# 将结果存储在'XGB_prediction.csv'
output.to_csv('XGB_prediction_Top3.csv', index=False)

'''
# 在测试集上进行预测并计算准确率
y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test data:", accuracy)
output = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
output.to_csv('XGB_prediction.csv', index=False)
'''
