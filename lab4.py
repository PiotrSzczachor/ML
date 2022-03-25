#!/usr/bin/env python
# coding: utf-8

# In[242]:


from sklearn import datasets
import pandas as pd
import numpy as np


# In[243]:


data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
print(data_breast_cancer['DESCR'])


# In[244]:


data_iris = datasets.load_iris(as_frame=True)
print(data_iris['DESCR'])


# In[245]:


from sklearn.model_selection import train_test_split
X_cancer = data_breast_cancer.data
y_cancer = data_breast_cancer.target


# In[246]:


X_cancer.head()


# In[247]:


X_cancer = X_cancer[["mean area", "mean smoothness"]]


# In[248]:


y_cancer.head()


# In[249]:


cancer_X_train, cancer_X_test, cancer_y_train, cancer_y_test = train_test_split(X_cancer, y_cancer, test_size=0.2)


# In[250]:


print(len(cancer_X_train), len(cancer_X_test))


# In[251]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

scaled_svm_clf = Pipeline([("scaler", StandardScaler()),
                          ("linear_svc", LinearSVC(C=1, loss="hinge"))])


# In[252]:


scaled_svm_clf.fit(cancer_X_train, cancer_y_train)


# In[253]:


scaled_cancer_y_pred_test = scaled_svm_clf.predict(cancer_X_test)
scaled_cancer_y_pred_train = scaled_svm_clf.predict(cancer_X_train)


# In[254]:


from sklearn.metrics import accuracy_score
scaled_cancer_svm_acc_test = accuracy_score(cancer_y_test, scaled_cancer_y_pred_test)
print(scaled_cancer_svm_acc_test)


# In[255]:


scaled_cancer_svm_acc_train = accuracy_score(cancer_y_train, scaled_cancer_y_pred_train)
print(scaled_cancer_svm_acc_train)


# In[256]:


not_scaled_svm_clf = Pipeline([("linear_svc", LinearSVC(C=1, loss="hinge"))])


# In[257]:


not_scaled_svm_clf.fit(cancer_X_train, cancer_y_train)


# In[258]:


not_scaled_cancer_y_pred_test = not_scaled_svm_clf.predict(cancer_X_test)
not_scaled_cancer_y_pred_train = not_scaled_svm_clf.predict(cancer_X_train)


# In[259]:


not_scaled_cancer_svm_acc_test = accuracy_score(cancer_y_test, not_scaled_cancer_y_pred_test)
print(not_scaled_cancer_svm_acc_test)


# In[260]:


not_scaled_cancer_svm_acc_train = accuracy_score(cancer_y_train, not_scaled_cancer_y_pred_train)
print(not_scaled_cancer_svm_acc_train)


# In[261]:


acc_list = [not_scaled_cancer_svm_acc_train, not_scaled_cancer_svm_acc_test,
            scaled_cancer_svm_acc_train, scaled_cancer_svm_acc_test]
print(acc_list)


# In[262]:


import pickle
open_file = open("bc_acc.pkl", "wb")
pickle.dump(acc_list, open_file)
open_file.close()


# In[263]:


open_file = open("bc_acc.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[264]:


X_iris = data_iris.data
y_iris = data_iris.target


# In[265]:


X_iris.head()


# In[266]:


X_iris = X_iris[["petal length (cm)", "petal width (cm)"]]


# In[267]:


y_iris.head()


# In[268]:


iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(X_iris, y_iris, test_size=0.2)


# In[269]:


print(len(iris_X_train), len(iris_X_test))


# In[270]:


scaled_svm_clf.fit(iris_X_train, iris_y_train)


# In[271]:


scaled_iris_y_pred_test = scaled_svm_clf.predict(iris_X_test)
scaled_iris_y_pred_train = scaled_svm_clf.predict(iris_X_train)


# In[272]:


scaled_iris_svm_acc_test = accuracy_score(iris_y_test, scaled_iris_y_pred_test)
print(scaled_iris_svm_acc_test)


# In[273]:


scaled_iris_svm_acc_train = accuracy_score(iris_y_train, scaled_iris_y_pred_train)
print(scaled_cancer_svm_acc_train)


# In[274]:


not_scaled_svm_clf.fit(iris_X_train, iris_y_train)


# In[275]:


not_scaled_iris_y_pred_test = not_scaled_svm_clf.predict(iris_X_test)
not_scaled_iris_y_pred_train = not_scaled_svm_clf.predict(iris_X_train)


# In[276]:


not_scaled_iris_svm_acc_test = accuracy_score(iris_y_test, not_scaled_iris_y_pred_test)
print(not_scaled_iris_svm_acc_test)


# In[277]:


not_scaled_iris_svm_acc_train = accuracy_score(iris_y_train, not_scaled_iris_y_pred_train)
print(not_scaled_iris_svm_acc_train)


# In[278]:


acc_list = [not_scaled_iris_svm_acc_train, not_scaled_iris_svm_acc_test,
            scaled_iris_svm_acc_train, scaled_iris_svm_acc_test]
print(acc_list)


# In[279]:


open_file = open("iris_acc.pkl", "wb")
pickle.dump(acc_list, open_file)
open_file.close()


# In[280]:


open_file = open("iris_acc.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[ ]:




