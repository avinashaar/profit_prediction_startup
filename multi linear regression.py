
# coding: utf-8

# In[1]:



#import library
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
sns.set()


# In[2]:


dataset = pd.read_csv('/home/avi/Desktop/data/50_Startups.csv')


# In[3]:


# Define Dataset variable X and y
X = dataset[['R&D Spend','Administration','Marketing Spend','State']]
Y = dataset['Profit']


# In[4]:


#convert string to float 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X['State'] = labelencoder.fit_transform(X['State'])


# In[5]:


#fit the linearregression in variable reg
reg = LinearRegression()
reg.fit(X,Y)


# In[6]:


x = sm.add_constant(X)
result = sm.OLS(Y,x).fit()
result.summary()


# In[7]:


#find the coefficient of variable in X
reg.coef_


# In[8]:


#find the coefficient of variable in intercept
reg.intercept_


# In[9]:


#predict the profit
reg.predict(X)


# In[17]:


x1 = dataset['R&D Spend']
x2 = dataset['Administration']
x3 = dataset['Marketing Spend']
x4 = dataset['State']


# In[18]:


x1


# In[23]:


yhat = 50142.50644347625 + 0.80575968*x1  + 0.02722767*x3 + -0.02682585*x2


# In[31]:


plt.scatter(x1,Y)


# In[32]:


plt.scatter(x2,Y)


# In[33]:


plt.scatter(x3,Y)


# In[34]:


plt.scatter(x4,Y)

