#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings 
warnings.filterwarnings('ignore') # feel free to comment this out if you want to see warnings 

import csv 
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

from keras.models import Sequential
from keras.layers import Dense, LSTM, Input, Flatten


from statsmodels.tools.eval_measures import rmse

import numpy as np



# In[2]:


mers = pd.read_csv('../data/mers-outbreak-dataset-20122019/weekly_clean.csv')


# In[3]:


mers.head(15)


# What countries were impacted with mers? 

# In[4]:


countries = list(mers["Region"].unique())
countries


# In[5]:


other = mers.loc[mers["Region"] == "Other Countries"]
other = other.reset_index(drop=True)

korea = mers.loc[mers["Region"] == "Republic of Korea"]
korea = korea.reset_index(drop=True)

sa = mers.loc[mers["Region"] == "Saudi Arabia"]
sa = sa.reset_index(drop=True)


# In[6]:


plt.plot(other.index, other['New Cases'], 'r', linewidth = 2, label="Other")
plt.plot(korea.index, korea['New Cases'],'b',linewidth = 2, label="Republic of Korea")
plt.plot(sa.index, sa['New Cases'],'g',linewidth = 2, label="Saudi Arabia")
plt.title("MERS: Korea, Saudi Arabia, and Others Edition")
plt.xlabel("Weeks after first confirmed case")
plt.ylabel("New Cases")
plt.legend()
plt.show()


# In[ ]:
sa.Year = pd.to_datetime(sa.Year.astype(str), format='%Y') + \
             pd.to_timedelta(sa.Week.mul(7).astype(str) + ' days')
             
korea.Year = pd.to_datetime(korea.Year.astype(str), format='%Y') + \
             pd.to_timedelta(korea.Week.mul(7).astype(str) + ' days')

other.Year = pd.to_datetime(other.Year.astype(str), format='%Y') + \
             pd.to_timedelta(other.Week.mul(7).astype(str) + ' days')
             

sa = sa.drop(["Week"],axis=1)
korea = korea.drop(["Week"],axis=1)
other = other.drop(["Week"],axis=1)

sa = sa.drop(["Region"],axis=1)
korea = korea.drop(["Region"],axis=1)
other = other.drop(["Region"],axis=1)

sa = sa.set_index("Year")
korea = korea.set_index("Year")
other = other.set_index("Year")


train_data_sa = sa[:len(sa)-75]
test_data_sa = sa[len(sa)-75:]
train_data_korea = korea[:len(korea)-75]
test_data_korea = korea[len(korea)-75:]
train_data_other = other[:len(other)-75]
test_data_other = other[len(other)-75:]

#%%

scaler = MinMaxScaler()
scaler.fit(train_data_sa)
scaled_train_data = scaler.transform(train_data_sa)
scaled_test_data = scaler.transform(test_data_sa)



 

n_input = 7
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)


lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

lstm_model.fit_generator(generator,epochs=20)

losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);


lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data_sa)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)

lstm_predictions_scaled

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

lstm_predictions

test_data_sa['LSTM_Predictions'] = lstm_predictions
test_data_sa


test_data_sa['New Cases'].plot(figsize = (16,5), legend=True)
test_data_sa['LSTM_Predictions'].plot(legend = True);



lstm_rmse_error_sa = rmse(test_data_sa['New Cases'], test_data_sa["LSTM_Predictions"])
lstm_mse_error_sa = lstm_rmse_error_sa**2
mean_value = sa['value'].mean()

#%%

scaler = MinMaxScaler()
scaler.fit(train_data_korea)
scaled_train_data = scaler.transform(train_data_korea)
scaled_test_data = scaler.transform(test_data_korea)



 

n_input = 7
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)


lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

lstm_model.fit_generator(generator,epochs=20)

losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);


lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data_korea)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)

lstm_predictions_scaled

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

lstm_predictions

test_data_korea['LSTM_Predictions'] = lstm_predictions
test_data_korea


test_data_korea['New Cases'].plot(figsize = (16,5), legend=True)
test_data_korea['LSTM_Predictions'].plot(legend = True);



lstm_rmse_error_korea = rmse(test_data_korea['New Cases'], test_data_korea["LSTM_Predictions"])
lstm_mse_error_korea = lstm_rmse_error_korea**2
mean_value = korea['value'].mean()

#%%

scaler = MinMaxScaler()
scaler.fit(train_data_other)
scaled_train_data = scaler.transform(train_data_other)
scaled_test_data = scaler.transform(test_data_other)



 

n_input = 7
n_features= 1
generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)


lstm_model = Sequential()
lstm_model.add(LSTM(200, activation='relu', input_shape=(n_input, n_features)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

lstm_model.summary()

lstm_model.fit_generator(generator,epochs=20)

losses_lstm = lstm_model.history.history['loss']
plt.figure(figsize=(12,4))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.xticks(np.arange(0,21,1))
plt.plot(range(len(losses_lstm)),losses_lstm);


lstm_predictions_scaled = list()

batch = scaled_train_data[-n_input:]
current_batch = batch.reshape((1, n_input, n_features))

for i in range(len(test_data_other)):   
    lstm_pred = lstm_model.predict(current_batch)[0]
    lstm_predictions_scaled.append(lstm_pred) 
    current_batch = np.append(current_batch[:,1:,:],[[lstm_pred]],axis=1)

lstm_predictions_scaled

lstm_predictions = scaler.inverse_transform(lstm_predictions_scaled)

lstm_predictions

test_data_other['LSTM_Predictions'] = lstm_predictions
test_data_other


test_data_other['New Cases'].plot(figsize = (16,5), legend=True)
test_data_other['LSTM_Predictions'].plot(legend = True);



lstm_rmse_error_other = rmse(test_data_other['New Cases'], test_data_other["LSTM_Predictions"])
lstm_mse_error_other = lstm_rmse_error_other**2
mean_value = other['value'].mean()







