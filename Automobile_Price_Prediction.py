#!/usr/bin/env python
# coding: utf-8

# # Importing the necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


# In[2]:


data = pd.read_csv('price.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum().sum()


# In[7]:


data.head()


# In[8]:


data.columns


# In[9]:


cols_num = [
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curbweight",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg"
]

cols_cat = [
    "symboling",
    "fueltype",
    "aspiration",
    "doornumber",
    "carbody",
    "drivewheel",
    "enginelocation",
    "enginetype",
    "cylindernumber",
    "fuelsystem"
]


# In[10]:


data["wheelbase"].isnull().any()


# In[11]:


data['aspiration'].unique()


# In[12]:


for col in cols_num:
    sns.relplot(x = col, y = "price", hue = "aspiration", data = data)
    plt.show()


# In[13]:


for col in cols_num:
    sns.relplot(x = col, y = "price", kind = "line", data = data)
    plt.show()


# In[14]:


for col in cols_num:
    sns.relplot(x = col, y = "price", kind = "line", data = data, ci = None)
    plt.show()


# In[15]:


for col in cols_cat:
    sns.catplot(x = col, y = "price", data = data)


# In[16]:


for col in cols_cat:
    sns.swarmplot(x = col, y = "price", data = data)
    plt.show()


# In[17]:


for col in cols_cat:
    sns.boxplot(x = col, y = "price", data = data)
    plt.show()


# In[18]:


for col in cols_cat:
    sns.boxenplot(x = col, y = "price", data = data)
    plt.show()


# In[19]:


for col in cols_cat:
    sns.violinplot(x = col, y = "price", data = data)
    plt.show()


# In[20]:


plt.figure(figsize = (10, 8))
plt.bar(data["cylindernumber"], data["price"])
plt.title("Price Vs Cylinder Number")
plt.xlabel("Number Of Cylinders")
plt.ylabel("Price")
plt.show()


# In[21]:


data['cylindernumber'].unique()


# In[22]:


sns.distplot(data["price"])
plt.show()


# In[23]:


sns.kdeplot(data["price"], shade = True)
plt.show()


# In[24]:


data.head()


# In[25]:


df = data.copy()


# In[26]:


df.head()


# In[27]:


le = LabelEncoder()


# In[28]:


def convert_word_to_num(a):
    if a == "zero":
        return 0
    elif a == "one":
        return 1
    elif a == "two":
        return 2
    elif a == "three":
        return 3
    elif a == "four":
        return 4
    elif a == "five":
        return 5
    elif a == "six":
        return 6
    elif a == "seven":
        return 7
    elif a == "eight":
        return 8
    elif a == "nine":
        return 9
    elif a == "ten":
        return 10
    elif a == "eleven":
        return 11
    elif a == "twelve":
        return 12


# In[29]:


df['doornumber'] = df['doornumber'].apply(convert_word_to_num)
df['cylindernumber'] = df['cylindernumber'].apply(convert_word_to_num)


# In[30]:


df.drop(['make'], axis = 1, inplace = True)


# In[31]:


df['fueltype'] = le.fit_transform(df['fueltype'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['carbody'] = le.fit_transform(df['carbody'])
df['drivewheel'] = le.fit_transform(df['drivewheel'])
df['enginelocation'] = le.fit_transform(df['enginelocation'])
df['enginetype'] = le.fit_transform(df['enginetype'])
df['fuelsystem'] = le.fit_transform(df['fuelsystem'])


# In[32]:


df.head()


# In[33]:


df['fueltype'].unique()


# In[34]:


df['aspiration'].unique()


# In[35]:


df['doornumber'].unique()


# In[36]:


df['carbody'].unique()


# In[37]:


df['drivewheel'].unique()


# In[38]:


df['enginelocation'].unique()


# In[39]:


df['enginetype'].unique()


# In[40]:


df['cylindernumber'].unique()


# In[41]:


df['fuelsystem'].unique()


# In[42]:


scaler = StandardScaler()


# In[43]:


cols_num = [
    "wheelbase",
    "carlength",
    "carwidth",
    "carheight",
    "curbweight",
    "enginesize",
    "boreratio",
    "stroke",
    "compressionratio",
    "horsepower",
    "peakrpm",
    "citympg",
    "highwaympg"
]


# In[44]:


for col in cols_num:
    df[col] = scaler.fit_transform(np.array(df[col]).reshape(-1, 1))


# In[45]:


df['wheelbase']


# In[46]:


df['wheelbase'].mean(), df['wheelbase'].std()


# In[47]:


for col in cols_num:
    print("Mean - ", df[col].mean(), "Std - ", df[col].std())


# In[48]:


df.head()


# In[49]:


df.head()


# # Training and Testing data

# In[50]:


X = df.drop(['price'], axis = 1)
y = df['price']


# In[51]:


X.head()


# In[52]:


y.head()


# In[53]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[54]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# ### Linear Regression

# In[55]:


lin_reg = LinearRegression()


# In[57]:


lin_reg.fit(X_train, y_train)


# In[58]:


lin_y_pred = lin_reg.predict(X_test)


# In[59]:


mse = mean_squared_error(y_test, lin_y_pred)


# In[60]:


mse


# In[61]:


rmse = np.sqrt(mse)


# In[62]:


rmse


# In[63]:


score = r2_score(y_test, lin_y_pred)


# In[64]:


score


# In[65]:


to_predict = [[0, 1, 0, 4, 3, 2, 0, 3.00, 2.54, 1.32, 0.72, 0.99, 5, 12, 0.50, 5, 0.88, 0.48, 0.55, 2.00, 0.78, 1.00, 1.50]]


# In[66]:


predicted = lin_reg.predict(to_predict)


# In[67]:


predicted


# In[68]:


cross_val_score(lin_reg, X, y, scoring = "neg_mean_squared_error")


# In[69]:


cross_val_score(lin_reg, X, y, scoring = "r2")


# # Decision Tree Regressor

# In[70]:


decision_tree_reg = DecisionTreeRegressor()


# In[71]:


decision_tree_reg.fit(X_train, y_train)


# In[72]:


tree_y_pred = decision_tree_reg.predict(X_test)


# In[73]:


mse = mean_squared_error(y_test, tree_y_pred)


# In[74]:


mse


# In[75]:


rmse = np.sqrt(mse)


# In[76]:


rmse


# In[77]:


score = r2_score(y_test, tree_y_pred)


# In[78]:


score


# In[79]:


to_predict = [[0, 1, 0, 4, 3, 2, 0, 3.00, 2.54, 1.32, 0.72, 0.99, 5, 12, 0.50, 5, 0.88, 0.48, 0.55, 2.00, 0.78, 1.00, 1.50]]


# In[80]:


predicted = decision_tree_reg.predict(to_predict)


# In[81]:


predicted


# In[82]:


cross_val_score(decision_tree_reg, X, y, scoring = "neg_mean_squared_error")


# In[83]:


cross_val_score(decision_tree_reg, X, y, scoring = "r2")


# ### RandomForestRegressor

# In[84]:


rfg = RandomForestRegressor()


# In[85]:


rfg.fit(X_train, y_train)


# In[86]:


rfg_y_pred = rfg.predict(X_test)


# In[87]:


mse = mean_squared_error(y_test, rfg_y_pred)


# In[88]:


mse


# In[89]:


score = r2_score(y_test, rfg_y_pred)


# In[90]:


score


# In[91]:


X_train.head()


# In[92]:


X_train.columns


# In[98]:


to_predict = [[0, 1, 0, 4, 3, 2, 0, 3.00, 2.54, 1.32, 0.72, 0.99, 5, 12, 0.50, 5, 0.88, 0.48, 0.55, 2.00, 0.78, 1.00, 1.50]]


# In[99]:


predicted = rfg.predict(to_predict)


# In[100]:


predicted


# In[101]:


cross_val_score(rfg, X, y, scoring = "neg_mean_squared_error")


# In[103]:


cross_val_score(rfg, X, y, scoring = "r2")


# In[ ]:




