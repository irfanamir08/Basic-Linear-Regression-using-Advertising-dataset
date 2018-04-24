
# coding: utf-8

# In[21]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('Advertising.csv')
#df.head()
dataset = df.iloc[:, 1:]
dataset.head()


# In[22]:


dataset.shape


# In[23]:


sns.pairplot(dataset, x_vars = ['TV', 'radio', 'newspaper'], y_vars = 'sales',aspect = 0.7, size = 6, kind = 'reg')
plt.show()


# In[24]:


X = dataset[['TV', 'radio', 'newspaper']]
y = dataset[['sales']]
print(type(X))
print(type(y))
#print(X)
#print(y)
print(X.head())
X.shape


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, shuffle = True)
#print(X_train)
#print(y_train)
#print(X_test)
#print(y_test)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[33]:


linReg = LinearRegression()
linReg.fit(X_train, y_train)
prediction = linReg.predict(X_test)


# In[36]:


plt.plot(prediction ,color='blue', label='predicted')
plt.ylabel('sales')
plt.legend(loc='upper right')
plt.show()
#print(prediction)

