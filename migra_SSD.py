# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:33:07 2021

@author: thoma
"""

import pandas as pd 
import numpy as np 
import os 
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import datetime as dt
import geopandas as gpd
from rasterstats import zonal_stats
from tslearn.clustering import TimeSeriesKMeans
from Geo_functions import extract_geo_from_nc4

########################################################################################################################################################################################
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
#                                   Preprocess of the data                                                                                                                             #
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
########################################################################################################################################################################################


######################################################### Migra data

ind_all=pd.date_range(start='01/01/2020',end='30/06/2021',freq='M')
count=0
for elmt in os.listdir('D:/Pace_data/Migration/South_Soudan/Data/'):
    df=pd.read_excel('D:/Pace_data/Migration/South_Soudan/Data/'+elmt)
    df_real=df[df['dep.country']== 'SSD']
    df_group=df_real.groupby(by=['dep.adm2','dest.adm1'])['total.ind'].sum()
    if count == 0: 
        df_tot=pd.DataFrame(df_group)
        df_tot.columns=[str(ind_all[0])[0:7]]
    else: 
        df_tot[str(ind_all[count])[0:7]]=df_group
    count=count+1    

df_tot=df_tot.fillna(0)   
df_tot_2=df_tot.reset_index(level=[0,1]) 
j=df_tot_2.iloc[:,0:2].values.tolist()
df_tot_2=df_tot_2.iloc[:,2:]
df_tot_2=df_tot_2.transpose()
l_col=[]
for elmt in j:
    l_col.append(elmt[0]+' / '+elmt[1])
df_tot_2.columns=l_col
df_tot_2.index=ind_all
del df_real,df_group,elmt,l_col,count


dep_u=[]
dest_u=[]
for elmt in j:
    dep_u.append(elmt[0])
    dest_u.append(elmt[1])
dest_u = list(set(dest_u))
dep_u =  list(set(dep_u))

df_tot_3=df_tot.reset_index(level=[0,1]) 
df_tot_3=df_tot_3.drop('dest.adm1',axis=1)
df_tot_3=df_tot_3.groupby(by='dep.adm2').sum()
df_tot_3=df_tot_3.transpose()
df_tot_3.index=ind_all
ind_mig=pd.date_range(start='01/01/2020',end='30/06/2021',freq='M')
df_mig=df_tot_3
df_mig=df_mig.rename(columns={'Abyei Area':'Abyei Region', 'Canal/Pigi': 'Canal-Pigi','Kajo-Keji':'Kajo-keji'})
df_mig=df_mig.drop(columns=['Unknown'])
df_mig.to_csv('D:/Pace_data/Migration/South_Soudan/TS/Migra.csv')


################################################################# Climate data 



df_test= gpd.read_file('D:/Pace_data/Migration/South_Soudan/SS_adm2.shp')
names_shp=df_test[['geometry']]
names_shp=names_shp.transpose()
names_shp.columns=df_test['ADM2_EN']
ind_all=pd.date_range(start='01/01/2019',end='31/08/2021',freq='D')
extract_geo_from_nc4('D:/Pace_data/Migration/South_Soudan/SSD_2019.nc','t2m',ind_all,'D:/Pace_data/Migration/South_Soudan/Images/Temp/')
extract_geo_from_nc4('D:/Pace_data/Migration/South_Soudan/SSD_2019.nc','swvl1',ind_all,'D:/Pace_data/Migration/South_Soudan/Images/Hum/')
extract_geo_from_nc4('D:/Pace_data/Migration/South_Soudan/SSD_2019.nc','tp',ind_all,'D:/Pace_data/Migration/South_Soudan/Images/RR/')
fold_v='D:/Pace_data/Migration/South_Soudan/Images/'
df_clim_tot=[]
for variable in ['Temp','RR','Hum']:
    df_clim=[]
    for i in range(len(names_shp.iloc[0,:])): 
        df_tot=[]
        for filename in os.listdir(fold_v+variable+'/'):
            test=zonal_stats(names_shp.iloc[0,i], fold_v+variable+'/'+filename,
            stats="mean", nodata=-32767)
            for j in range(len(test)):
                if test[j]['mean'] != float('-inf'):
                    hum=test[j]['mean']
                else:
                    hum=float('NaN')
            df_tot.append(hum) 
        df_clim.append(df_tot)
    df_clim_2=pd.DataFrame(np.array(df_clim).T)
    df_clim_2.index=ind_all
    df_clim_2.columns=names_shp.columns
    df_clim_tot.append(df_clim_2)  
    

df_clim_tot[0].to_csv('D:/Pace_data/Migration/South_Soudan/TS/Temp_raw.csv')  
df_clim_tot[1].to_csv('D:/Pace_data/Migration/South_Soudan/TS/RR_raw.csv')  
df_clim_tot[2].to_csv('D:/Pace_data/Migration/South_Soudan/TS/Hum_raw.csv')        

n_ts_clim_final=[]
for elmt in n_ts_clim_final:
    n_ts_clim_final.append((elmt-elmt.mean())/elmt.std())

n_ts_clim_final[0].to_csv('D:/Pace_data/Migration/South_Soudan/TS/Temp_norm.csv')  
n_ts_clim_final[1].to_csv('D:/Pace_data/Migration/South_Soudan/TS/RR_norm.csv')  
n_ts_clim_final[2].to_csv('D:/Pace_data/Migration/South_Soudan/TS/Hum_norm.csv')    


####### Removing the seasonal component 

f_df_clim_tot=[]
for j in range(len(n_ts_clim_final)):
    f_df_clim= pd.DataFrame()
    for n_series in list(n_ts_clim_final[j].columns):
        result_add = seasonal_decompose(n_ts_clim_final[j][n_series], model='additive', extrapolate_trend='freq')
        diff = list()
        for i in range(len(result_add.seasonal)):
        	value = n_ts_clim_final[j][n_series][i] - result_add.seasonal[i]
        	diff.append(value)
        f_df_clim[n_series]=diff 
    f_df_clim.index=n_ts_clim_final[j].index
    f_df_clim_tot.append(f_df_clim)


f_df_clim_tot[0].to_csv('D:/Pace_data/Migration/South_Soudan/TS/Temp_diff.csv')  
f_df_clim_tot[1].to_csv('D:/Pace_data/Migration/South_Soudan/TS/RR_diff.csv')  
f_df_clim_tot[2].to_csv('D:/Pace_data/Migration/South_Soudan/TS/Hum_diff.csv')   



########################################################################################################################################################################################
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
#                                   Unsupervised learning of the climate TS                                                                                                            #
#                                                                                                                                                                                      #
#                                                                                                                                                                                      #
########################################################################################################################################################################################


length_d=90
number_s=1


################################################################## Precipitation #######################################################################################################

df_s=pd.read_csv('D:/Pace_data/Migration/South_Soudan/TS/RR_diff.csv',index_col=0)
df_s.index=ind_all


ts_seq=[]
for col in df_mig.columns:
    for i in range(len(ind_mig)):
        for j in range(number_s):
            ts_seq.append(df_s.loc[dt.date(ind_mig[i].year,ind_mig[i].month,1)-dt.timedelta(days=length_d*(j+1)):dt.date(ind_mig[i].year,ind_mig[i].month,1)-dt.timedelta(days=length_d*j),col])
ts_seq=np.array(ts_seq)
ts_seq=ts_seq.reshape(len(ts_seq),length_d+1,1)
ts_seq_visu=pd.DataFrame(ts_seq.reshape(len(ts_seq),length_d+1))
ts_seq_visu=ts_seq_visu.transpose()


############# clustering

###### Elbow test
elb_t=pd.DataFrame()
for n_clu in range(2,11):
    km_dba = TimeSeriesKMeans(n_clusters=n_clu, metric="dtw", max_iter=1000,
                              max_iter_barycenter=100,verbose=0,
                              random_state=0).fit(ts_seq)

    y= km_dba.labels_
    y_df=pd.DataFrame(y)
    elb_t[n_clu]=[km_dba.inertia_,y_df[0].value_counts().min()]   
    #print(y_df[0].value_counts().min())
    
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Sum of distances', color=color)
ax1.plot(elb_t.loc[0,:], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Minimum size of cluster', color=color)  # we already handled the x-label with ax1
ax2.plot(elb_t.loc[1,:], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
    
######## Application 

km_dba = TimeSeriesKMeans(n_clusters=7, metric="dtw", max_iter=1000,
                          max_iter_barycenter=100,verbose=0,
                          random_state=0).fit(ts_seq)

y= km_dba.labels_
y_df=pd.DataFrame(y)
elb_t[n_clu]=[km_dba.inertia_,y_df[0].value_counts().min()] 

for i in range(km_dba.n_clusters):
    plt.plot(ts_seq_visu.iloc[:,y==i])
    plt.title('Cluster '+str(i+1))
    plt.show()

y=y.reshape(len(df_mig.iloc[0,:]),(len(df_mig)))
y=pd.DataFrame(y).transpose()
y.columns=df_mig.columns
y.index=df_mig.index
y.to_csv('D:/Pace_data/Migration/South_Soudan/TS/Class_RR.csv') 

######################################################### Temp  #############################################################################

df_s=pd.read_csv('D:/Pace_data/Migration/South_Soudan/TS/Temp_diff.csv',index_col=0)
df_s.index=ind_all


ts_seq=[]
for col in df_mig.columns:
    for i in range(len(ind_mig)):
        for j in range(number_s):
            ts_seq.append(df_s.loc[dt.date(ind_mig[i].year,ind_mig[i].month,1)-dt.timedelta(days=length_d*(j+1)):dt.date(ind_mig[i].year,ind_mig[i].month,1)-dt.timedelta(days=length_d*j),col])
ts_seq=np.array(ts_seq)
ts_seq=ts_seq.reshape(len(ts_seq),length_d+1,1)
ts_seq_visu=pd.DataFrame(ts_seq.reshape(len(ts_seq),length_d+1))
ts_seq_visu=ts_seq_visu.transpose()


########################################### clustering

###### Elbow test
elb_t=pd.DataFrame()
for n_clu in range(2,11):
    km_dba = TimeSeriesKMeans(n_clusters=n_clu, metric="dtw", max_iter=1000,
                              max_iter_barycenter=100,verbose=0,
                              random_state=0).fit(ts_seq)

    y= km_dba.labels_
    y_df=pd.DataFrame(y)
    elb_t[n_clu]=[km_dba.inertia_,y_df[0].value_counts().min()]   
    #print(y_df[0].value_counts().min())
    
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Sum of distances', color=color)
ax1.plot(elb_t.loc[0,:], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Minimum size of cluster', color=color)  # we already handled the x-label with ax1
ax2.plot(elb_t.loc[1,:], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
    
######## Application 

km_dba = TimeSeriesKMeans(n_clusters=6, metric="dtw", max_iter=1000,
                          max_iter_barycenter=100,verbose=0,
                          random_state=0).fit(ts_seq)

y= km_dba.labels_
y_df=pd.DataFrame(y)
elb_t[n_clu]=[km_dba.inertia_,y_df[0].value_counts().min()] 

for i in range(km_dba.n_clusters):
    plt.plot(ts_seq_visu.iloc[:,y==i])
    plt.title('Cluster '+str(i+1))
    plt.show()

y=y.reshape(len(df_mig.iloc[0,:]),(len(df_mig)))
y=pd.DataFrame(y).transpose()
y.columns=df_mig.columns
y.index=df_mig.index
y.to_csv('D:/Pace_data/Migration/South_Soudan/TS/Class_Temp.csv') 

######################################################### Hum  #############################################################################

df_s=pd.read_csv('D:/Pace_data/Migration/South_Soudan/TS/Hum_diff.csv',index_col=0)
df_s.index=ind_all


ts_seq=[]
for col in df_mig.columns:
    for i in range(len(ind_mig)):
        for j in range(number_s):
            ts_seq.append(df_s.loc[dt.date(ind_mig[i].year,ind_mig[i].month,1)-dt.timedelta(days=length_d*(j+1)):dt.date(ind_mig[i].year,ind_mig[i].month,1)-dt.timedelta(days=length_d*j),col])
ts_seq=np.array(ts_seq)
ts_seq=ts_seq.reshape(len(ts_seq),length_d+1,1)
ts_seq_visu=pd.DataFrame(ts_seq.reshape(len(ts_seq),length_d+1))
ts_seq_visu=ts_seq_visu.transpose()


########################################### clustering

###### Elbow test
elb_t=pd.DataFrame()
for n_clu in range(2,11):
    km_dba = TimeSeriesKMeans(n_clusters=n_clu, metric="dtw", max_iter=1000,
                              max_iter_barycenter=100,verbose=0,
                              random_state=0).fit(ts_seq)

    y= km_dba.labels_
    y_df=pd.DataFrame(y)
    elb_t[n_clu]=[km_dba.inertia_,y_df[0].value_counts().min()]   
    #print(y_df[0].value_counts().min())
    
fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Sum of distances', color=color)
ax1.plot(elb_t.loc[0,:], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Minimum size of cluster', color=color)  # we already handled the x-label with ax1
ax2.plot(elb_t.loc[1,:], color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
    
######## Application 

km_dba = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=1000,
                          max_iter_barycenter=100,verbose=0,
                          random_state=0).fit(ts_seq)

y= km_dba.labels_
y_df=pd.DataFrame(y)
elb_t[n_clu]=[km_dba.inertia_,y_df[0].value_counts().min()] 

for i in range(km_dba.n_clusters):
    plt.plot(ts_seq_visu.iloc[:,y==i])
    plt.title('Cluster '+str(i+1))
    plt.show()

y=y.reshape(len(df_mig.iloc[0,:]),(len(df_mig)))
y=pd.DataFrame(y).transpose()
y.columns=df_mig.columns
y.index=df_mig.index
y.to_csv('D:/Pace_data/Migration/South_Soudan/TS/Class_Hum.csv') 


