#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader as pdr
import pandas as pd


# In[2]:


key = "3d4434cb3da000c73a676e1bebed51e1e2017b64"


# In[3]:


df = pdr.get_data_tiingo('AAPL', api_key= key)


# In[4]:


df.to_csv("AAPLl.csv")


# In[5]:


df = pd.read_csv("AAPL.csv")


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df1 = df.reset_index()["close"]
df1


# In[12]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[13]:


# Visualize the closing price history
plt.figure(figsize=(15,8))
plt.title('Close Price History')
plt.plot(df['close'])
plt.xlabel('index',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=20)
plt.show()


# In[14]:


# LSTM are sensitive to the scale of the data . so we will apply MinMax Scaler


# In[3]:


import numpy as np


# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[17]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Create MinMaxScaler with the desired feature range
scaler = MinMaxScaler(feature_range=(0, 1))

# Assuming df1 is your DataFrame or array, reshape it if needed
df1_reshaped = np.array(df1).reshape(-1, 1)

# Fit and transform the data using MinMaxScaler
df1 = scaler.fit_transform(df1_reshaped)


# In[18]:


df1


# In[19]:


df1.shape


# In[20]:


# splitting data set into train and test split
training_size = int(len(df1)*0.65)
test_size = len(df1)-training_size
train_data,test_data = df1[0:training_size,:],df1[training_size: len(df1),:1]


# In[21]:


training_size, test_size


# In[22]:


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[23]:


print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

print("Train data type:", train_data.dtype)
print("Test data type:", test_data.dtype)

# Now, create the datasets
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)


# In[24]:


print(X_train)


# In[25]:


print(X_train.shape) 
print(y_train.shape)


# In[26]:


print(X_test.shape) 
print(y_test.shape)


# In[27]:


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# In[28]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[29]:


model = Sequential ()
model.add(LSTM(50,return_sequences = True, input_shape = (100,1)))
model.add(LSTM(50,return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss = "mean_squared_error", optimizer = "adam")


# In[30]:


model.summary()


# In[31]:


model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100, batch_size = 64, verbose = 1)


# In[32]:


import tensorflow as tf


# In[33]:


tf.__version__


# In[34]:


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


# In[36]:


test_predict


# In[39]:


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)


# In[40]:


test_predict


# In[41]:


import math

from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[42]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[59]:


from math import sqrt
from tensorflow.keras.utils import plot_model
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[60]:


print("Mean Absolute Error:", mean_absolute_error(y_test,test_predict))
print('Mean Squared Error:', mean_squared_error(y_test,test_predict))
print('Root Mean Squared Error:', sqrt(mean_squared_error(y_test,test_predict)))
print("Coefficient of Determination:", r2_score(y_test,test_predict))


# In[63]:


# Create trainPredictPlot and testPredictPlot arrays

look_back = 100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, 0] = train_predict.flatten()

testPredictPlot = np.empty_like(df1)
testPredictPlot[:] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, 0] = test_predict.flatten()

# Plot baseline and predictions
plt.figure(figsize=(10,4))
plt.title('Close Price History')
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.xlabel('index',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=20)
plt.show()


# In[44]:


len(test_data)


# In[45]:


X_input = test_data [341: ].reshape(1,-1)
X_input.shape


# In[46]:


temp_input = list(X_input)
temp_input = temp_input[0].tolist()


# In[47]:


temp_input


# In[48]:


# prediction for next 10 days
lst_output = []
n_steps = 100
i = 0
while(i < 30):
    if len(temp_input) > 100:
        X_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, X_input))
        
        X_input = X_input.reshape(1, n_steps, 1)  # Corrected assignment
        # print(X_input)
        
        yhat = model.predict(X_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        
        # print(temp_input)
        lst_output.extend(yhat.tolist())
        i += 1
    else:
        X_input = X_input.reshape(1, n_steps, 1)  # Corrected assignment
        yhat = model.predict(X_input, verbose=0)
        print(yhat[0])
        
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        
        lst_output.extend(yhat.tolist())
        i += 1
        
print(lst_output)


# In[49]:


day_new = np.arange(1,101)
day_pred = np.arange(101,131)


# In[50]:


import matplotlib.pyplot as plt


# In[51]:


len(df1)


# In[52]:


df3=df1.tolist()
df3.extend(lst_output)


# In[64]:


plt.figure(figsize=(10,4))
plt.title('Close Price History')
plt.plot(day_new, scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
plt.xlabel('index',fontsize=18)
plt.ylabel('Close Price USD ($)',fontsize=20)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




