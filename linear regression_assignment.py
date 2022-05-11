#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[20]:


df=pd.read_csv('avocado.csv', index_col="Date")
df


# In[21]:


x = df['Total Volume'].to_numpy()
x = x.reshape(-1, 1)
x


# In[22]:


y = df['AveragePrice'].to_numpy()
y = y.reshape(-1, 1)
y


# In[23]:


plt.scatter(x, y)


# In[35]:


model= linear_model.LinearRegression()
model


# In[36]:


model.fit(x,y)


# In[37]:


model.intercept_


# In[38]:


model.coef_


# In[39]:


print("y = " + str(model.intercept_) + " + " + str(model.coef_) + "x")


# In[45]:


predict = model.predict(x[0:100])
predict


# In[46]:


plt.plot(x,predict,c="red")
plt.scatter(x,y)
a = model.intercept_
b = model.coef_

print("截距：", a)
print("斜率：", b)
print("y = " + str(a) + " + " + str(b) + "x")
print((model.score(x,y)*100)) #計算模型分數，也就是統計迴歸分析中的R-square

