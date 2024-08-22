#!/usr/bin/env python
# coding: utf-8

# # DataPreprocessing

# In[127]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[128]:


dataset = pd.read_csv(r'D:\breast_cancer_detect\data.csv')


# In[129]:


dataset.head()


# In[130]:


#data exploration


# In[131]:


dataset.shape


# In[132]:


dataset.info()


# In[133]:


dataset.select_dtypes(include='object').columns


# In[134]:


len(dataset.select_dtypes(include='object').columns)


# In[135]:


dataset.select_dtypes(include=['float64','int64']).columns


# In[136]:


len(dataset.select_dtypes(include=['float64','int64']).columns)


# In[137]:


#statistical summary
dataset.describe()


# # dealing with the missing values

# In[138]:


dataset.isnull().values.any()


# In[139]:


dataset.isnull().values.sum()


# In[140]:


dataset.columns[dataset.isnull().any()]


# In[141]:


len(dataset.columns[dataset.isnull().any()])


# In[142]:


dataset['Unnamed: 32'].count()


# In[143]:


dataset = dataset.drop(columns='Unnamed: 32')


# In[144]:


dataset.shape


# In[145]:


dataset.isnull().values.any()


# # categorical data

# In[146]:


dataset.select_dtypes(include='object').columns


# In[147]:


dataset['diagnosis'].unique()


# In[148]:


dataset['diagnosis'].nunique()


# In[149]:


# one hot encoding
dataset = pd.get_dummies(data=dataset, drop_first=True)


# In[150]:


dataset.head()


# # countplot

# In[154]:


sns.countplot(dataset['diagnosis_M'], label='Count')
plt.show()


# In[155]:


print(dataset['diagnosis_M'].value_counts())


# # correlation matrix and heatmap

# In[156]:


dataset_2 = dataset.drop(columns='diagnosis_M')


# In[157]:


dataset_2.head()


# In[158]:


dataset_2.corrwith(dataset['diagnosis_M']).plot.bar(
    figsize=(20,10), title = 'Correlated with diagnosis_M', rot=45, grid=True
)
plt.show()


# In[162]:


# Correlation matrix
corr = dataset.corr()
corr


# In[163]:


# heatmap


plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)
plt.show()


# # splitting the dataset train and test set

# In[164]:


dataset.head()


# In[165]:


# matrix of features / independent variables
x = dataset.iloc[:, 1:-1].values


# In[166]:


x.shape


# In[167]:


# target variable / dependent variable
y = dataset.iloc[:, -1].values


# In[168]:


y.shape


# In[169]:


from sklearn.model_selection import  train_test_split


# In[170]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[171]:


x_train.shape


# In[172]:


x_test.shape


# In[173]:


y_train.shape


# In[174]:


y_test.shape


# # feature scaling

# In[175]:


from sklearn.preprocessing import StandardScaler


# In[176]:


sc = StandardScaler()


# In[177]:


x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[178]:


x_train


# In[209]:


x_test


# # building the model

# # 1) logistic regression

# In[210]:


from sklearn.linear_model import LogisticRegression


# In[211]:


classifir_lr = LogisticRegression(random_state=0)


# In[212]:


classifir_lr.fit(x_train, y_train)


# In[213]:


y_pred = classifir_lr.predict(x_test)


# In[214]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# In[215]:


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


# In[216]:


results = pd.DataFrame([['Logistic Regression', acc, f1, prec, rec]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])
results


# In[217]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# # cross validation

# In[218]:


from sklearn.model_selection import cross_val_score


# In[219]:


accuracies = cross_val_score(estimator=classifir_lr, X=x_train, y=y_train, cv=10)


# In[220]:


print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# # 2) random forest

# In[221]:


from sklearn.ensemble import RandomForestClassifier


# In[222]:


classifier_rm = RandomForestClassifier(random_state=0)
classifier_rm.fit(x_train, y_train)

print(classifier_rm.__repr__())
params = classifier_rm.get_params()
print(params)


# In[223]:


y_pred = classifier_rm.predict(x_test)


# In[224]:


from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


# In[225]:


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)


# In[226]:


model_results = pd.DataFrame([['Random forest', acc, f1, prec, rec]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])


# In[227]:


results = results.append(model_results, ignore_index=True)
results


# In[228]:


cm = confusion_matrix(y_test, y_pred)
print(cm)


# # cross validation

# In[229]:


from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifier_rm, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# In[236]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# # Part 3: Randomized Search to find the best parameters (Logistic regression)

# In[237]:


from sklearn.model_selection import RandomizedSearchCV


# In[238]:


parameters = {'penalty':['l1', 'l2', 'elasticnet', 'none'],
              'C':[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
              'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
              }
parameters


# In[239]:


random_search = RandomizedSearchCV(estimator=classifir_lr, param_distributions=parameters, n_iter=5, 
                                   scoring='roc_auc', n_jobs = -1, cv=5, verbose=3)


# In[240]:


random_search.fit(x_train, y_train)


# In[253]:


random_search.fit(x_train, y_train)


# In[242]:


random_search.best_score_


# In[243]:


random_search.best_params_


# # Part 4: Final model (Logistic regression)

# In[254]:


from sklearn.linear_model import LogisticRegression
classifir = LogisticRegression(C=1.5, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l1',
                   random_state=0, solver='saga', tol=0.0001, verbose=0,
                   warm_start=False)
classifir.fit(x_train, y_train)


# In[255]:


y_pred = classifir.predict(x_test)


# In[256]:


acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)

model_results = pd.DataFrame([['Final Logistic Regression', acc, f1, prec, rec]],
               columns = ['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall'])


results = results.append(model_results, ignore_index = True)
results


# # cross validation

# In[257]:


from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=classifir, X=x_train, y=y_train, cv=10)

print("Accuracy is {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is {:.2f} %".format(accuracies.std()*100))


# # predicting a single observation

# In[258]:


dataset.head()


# In[277]:


fst_obs = [[17.99, 10.38, 122.80, 1001.0, 0.11840, 0.27760,	0.3001,	0.14710, 0.2419, 0.07871, 1.0950, 0.9053, 8.589, 153.40, 0.006399, 0.04904,	0.05373, 0.01587, 0.03003, 0.006193, 25.38,
17.33, 184.60, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.11890]]

fst_obs


# In[278]:


scnd_obs = [[15.78,17.89,103.6,781,0.0971,0.1292,0.09954,0.06606,0.1842,0.06082,0.5058,0.9849,3.564,54.16,0.005771,0.04061,0.02791,0.01282,0.02008,0.004144,20.42,27.28,136.5,1299,0.1396,0.5609,0.3965,0.181,0.3792,0.1048
]]
scnd_obs


# In[281]:


classifir.predict(sc.transform(fst_obs))


# In[282]:


classifir.predict(sc.transform(scnd_obs))


# In[283]:


third_obs = [[13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]]
third_obs


# In[285]:


classifir.predict(sc.transform(scnd_obs))


# In[ ]:




