#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[13]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt


# In[14]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *


# In[15]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


# In[16]:


df_train = pd.read_csv("trainset.csv")
df_train.head()


# In[17]:


np.array(df_train.Date.iloc[800:].values).shape


# In[18]:


df_train[800:]


# In[19]:


training_set1 = df_train.iloc[:800, 1:2].values
training_set1.shape


# ## Train Test Split

# In[20]:


training_data = df_train.iloc[:,1:2]
training_set = training_data[0:800]
test_set      = training_data[800:]
print(training_set.shape)
plt.plot(training_set)
plt.ylabel('Price')


# In[21]:


training_set


# ## Data Normalization

# In[22]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler(feature_range=(0,1))
training_data_scaled = mms.fit_transform(training_set)
plt.plot(training_data_scaled)


# In[23]:


print(np.array([training_data_scaled[60:65,0]]))


# In[24]:


# Create trainind data x_train with window/history of 60 samples and y_train with one future sample. 

def create_train_test_data(training_data_scaled,window,lag=0):
    x_train= []
    y_train= []
  
    for i in range(60,training_data_scaled.shape[0]):
        x_train.append(training_data_scaled[i-60:i,0])
        y_train.append(training_data_scaled[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    print('x_train shape: ',x_train.shape)
    print('y_train shape: ',y_train.shape)
    return x_train ,y_train


# In[25]:


x_train,y_train =create_train_test_data(training_data_scaled,60)
x_train.shape[0]
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# In[26]:


model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))

# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
model.fit(x_train, y_train, epochs = 100, batch_size = 32)


# In[27]:


model.save("my_h5_model.h5")


# In[28]:


test_set.size


# In[30]:


inputs = mms.transform(test_set)
X_test = []
for i in range(60,459):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)


# In[31]:


x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[32]:


X_test.shape


# In[33]:


predicted_stock_price = model.predict(x_test)
predicted_stock_price = mms.inverse_transform(predicted_stock_price)


# In[34]:


#predicted_stock_price.shape[1]

predicted_stock_price = np.reshape(predicted_stock_price,(predicted_stock_price.shape[0],))
predicted_stock_price.shape
predicted_stock_price


# In[35]:


# Visualising the results
plt.plot(df_train.Date[800:].values,test_set, color = "red", label = "Real google Stock Price")
plt.plot(df_train.Date[860:].values,predicted_stock_price, color = "blue", label = "Predicted google Stock Price")
plt.xticks(np.arange(0,459,50))
plt.title('google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('google Stock Price')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




