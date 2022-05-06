#!/usr/bin/env python
# coding: utf-8

# In[52]:


from sklearn import datasets
import pandas as pd
import numpy as np
data_breast_cancer = datasets.load_breast_cancer()


# In[53]:


from sklearn.datasets import load_iris
data_iris = load_iris()


# In[54]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[55]:


X_iris = data_iris.data
y_iris = data_iris.target


# In[56]:


X_cancer = data_breast_cancer.data
y_cancer = data_breast_cancer.target


# In[57]:


pca = PCA(n_components=0.9)
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)
X_iris_transformed_scaled = pca.fit_transform(X_iris_scaled)
iris_evr = pca.explained_variance_ratio_
print(iris_evr)


# In[58]:


pca.components_


# In[59]:


index_list_iris = []
for element in pca.components_:
    index = np.argmax(np.abs(element))
    index_list_iris.append(index)
print(index_list_iris)


# In[60]:


X_cancer_scaled = scaler.fit_transform(X_cancer)
X_cancer_transformed_scaled = pca.fit_transform(X_cancer_scaled)
cancer_evr = pca.explained_variance_ratio_
print(cancer_evr)


# In[61]:


pca.components_


# In[62]:


index_list_cancer = []
for element in pca.components_:
    index = np.argmax(np.abs(element))
    index_list_cancer.append(index)
print(index_list_cancer)


# In[63]:


import pickle
open_file = open("pca_bc.pkl", "wb")
pickle.dump(cancer_evr, open_file)
open_file.close()


# In[64]:


open_file = open("pca_bc.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[65]:


open_file = open("pca_ir.pkl", "wb")
pickle.dump(iris_evr, open_file)
open_file.close()


# In[66]:


open_file = open("pca_ir.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[67]:


open_file = open("idx_bc.pkl", "wb")
pickle.dump(index_list_cancer, open_file)
open_file.close()


# In[68]:


open_file = open("idx_bc.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[69]:


open_file = open("idx_ir.pkl", "wb")
pickle.dump(index_list_iris, open_file)
open_file.close()


# In[70]:


open_file = open("idx_ir.pkl", "rb")
loaded_list = pickle.load(open_file)
open_file.close()
print(loaded_list)


# In[ ]:




