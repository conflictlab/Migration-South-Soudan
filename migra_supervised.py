# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 18:22:09 2021

@author: thoma
"""
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTM,Input,Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
import statistics as st
from sklearn.model_selection import train_test_split

############################ Preprocess the data 


ind_all=pd.date_range(start='01/01/2020',end='30/06/2021',freq='M')
df_temp = pd.read_csv('D:/Pace_data/Migration/South_Soudan/TS/Class_Temp.csv',index_col=0)
df_temp.index=ind_all
df_temp_l=np.array(df_temp).T.reshape(len(df_temp)*len(df_temp.iloc[0,:]))
df_temp_l=pd.DataFrame(df_temp_l)
df_temp_l=pd.get_dummies(df_temp_l,columns=[0])
df_hum= pd.read_csv('D:/Pace_data/Migration/South_Soudan/TS/Class_Hum.csv',index_col=0)
df_hum.index=ind_all
df_hum_l=np.array(df_hum).T.reshape(len(df_hum)*len(df_hum.iloc[0,:]))
df_hum_l=pd.DataFrame(df_hum_l)
df_hum_l=pd.get_dummies(df_hum_l,columns=[0])
df_rr = pd.read_csv('D:/Pace_data/Migration/South_Soudan/TS/Class_RR.csv',index_col=0)
df_rr.index=ind_all
df_rr_l=np.array(df_rr).T.reshape(len(df_rr)*len(df_rr.iloc[0,:]))
df_rr_l=pd.DataFrame(df_rr_l)
df_rr_l=pd.get_dummies(df_rr_l,columns=[0])
df_migra=pd.read_csv('D:/Pace_data/Migration/South_Soudan/TS/Migra.csv',index_col=0)
df_migra.index=ind_all
scaler = MinMaxScaler(feature_range=(0,1))
df_migra = scaler.fit_transform(df_migra)
df_migra_l=df_migra.T.reshape(len(df_migra)*len(df_migra[0,:]))
df_migra_l=pd.DataFrame(df_migra_l)

trigo=[[0.5,st.sqrt(3)/2],[st.sqrt(2)/2,st.sqrt(2)/2],[1,0],[st.sqrt(2)/2,-st.sqrt(2)/2],[0.5,-st.sqrt(3)/2],[0,-1],[-0.5,-st.sqrt(3)/2],[-st.sqrt(2)/2,-st.sqrt(2)/2],[-1,0],[-st.sqrt(2)/2,st.sqrt(2)/2],[-0.5,st.sqrt(3)/2],[0,1]]
trigo=pd.DataFrame(trigo)
trigo.index=range(1,13)

df_test= gpd.read_file('D:/Pace_data/Migration/South_Soudan/SS_adm2.shp')
names_shp=df_test[['geometry']]
names_shp=names_shp.transpose()
names_shp.columns=df_test['ADM2_EN']
lat_long=[]
for i in df_migra.columns:
    lat_long.append(names_shp[i].centroid.bounds.values[0][0:2])

df_tot=[]
count=0
for zones in df_migra.columns:
    for month in ind_all:
        df_tot.append(list(lat_long[df_migra.columns.get_loc(zones)])+list(trigo.loc[month.month])+list(df_temp_l.iloc[count,:])+list(df_hum_l.iloc[count,:])+list(df_rr_l.iloc[count,:])+list(df_migra_l.iloc[count,:]))
        count=count+1

df_tot=pd.DataFrame(df_tot)      

######################## Learning and Testing 

df_tot.columns=range(len(df_tot.columns))
scaler = MinMaxScaler(feature_range=(0,1))
X = df_tot.loc[:,0:(len(df_tot.columns)-2)]
Y = df_tot.loc[:,(len(df_tot.columns)-1):]
X = scaler.fit_transform(X)
X = pd.DataFrame(X)
Y = pd.DataFrame(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

nb_hidden = int(len(X_train.iloc[:,1])/(2*(len(X_train.iloc[1,:])+1)))

model = keras.Sequential()
model.add(Dense(nb_hidden,input_dim=len(X_train.iloc[0,:]), activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
model.compile(loss='mae', optimizer= Adam(learning_rate=0.01))


# Training of the model
n_epochs=1000
es=EarlyStopping(monitor='val_loss', mode ='min',verbose=1,patience=100)
fitted= model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=n_epochs,batch_size=16,verbose=1,shuffle=True)#,callbacks=[es])
plt.plot(fitted.history['loss'], label ='train')
plt.plot(fitted.history['val_loss'], label = 'test')
plt.grid()
plt.legend()
plt.show()
pred = model.predict(X_test)

res= pred-y_test
print(abs(res).mean())
plt.plot(res[y_test>0.5],marker='.', linewidth=0)