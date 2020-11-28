#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Trading Data Set exports and imports of india LinK:-https://www.kaggle.com/lakshyaag/india-trade-data#


# In[ ]:





# # Data Importing

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[3]:


Export = pd.read_csv('2018-2010_export.csv')


# In[4]:


Import = pd.read_csv('2018-2010_import.csv')


# In[5]:


Export.head()


# In[6]:


Import.head()


# # Data Cleaning and removing null values

# In[7]:


Import.isnull()


# In[8]:


Export.isnull()


# In[9]:


Import.isnull().sum()


# In[10]:


Export.isnull().sum()


# In[11]:


Import.value.fillna(Import.value.mean(), inplace=True)


# In[12]:


Export.value.fillna(Export.value.mean(), inplace=True)


# In[13]:


Export.isnull().sum()


# In[14]:


Import.isnull().sum()


# In[15]:


Import.head()


# In[16]:


Export.head()


# In[17]:


Export.columns


# In[18]:


Export.describe()


# In[19]:


Import.columns


# In[20]:


Import.describe()


# In[21]:


country_list=list(Import.country.unique())


# In[22]:


country_list


# In[23]:


Import.loc[Import.country=='UNSPECIFIED']


# In[24]:


country_group=Import.groupby('country')
ls=[]
for country_name in country_list:
    ls.append([country_name, country_group.get_group(str(country_name)).value.sum() ])

total = pd.DataFrame(ls, columns = ['country', 'total_imports']) 


# In[25]:


total.loc[total.total_imports==0]


# In[26]:


country_export_list=list(Export.country.unique())


# In[27]:


country_export_list


# In[28]:


Export.loc[Export.country=='UNSPECIFIED']


# In[29]:


country_export_group=Export.groupby('country')
ls=[]
for country_name in country_export_list:
    ls.append([country_name, country_export_group.get_group(str(country_name)).value.sum() ])

total_exports = pd.DataFrame(ls, columns = ['country', 'total_exports']) 


# In[30]:


total_exports.loc[total_exports.total_exports==0]


# # Visualization

# In[31]:


Import.columns


# In[32]:


Export.columns


# In[33]:


px.scatter(Import,'year','value') 


# In[34]:


px.scatter(Export,'year','value')


# In[35]:


px.scatter(Import,'country','value')


# In[36]:


px.scatter(Export,'country','value')


# In[37]:


px.histogram(Import,'year','value',color='year')


# In[38]:


px.histogram(Export,'year','value',color='year')


# In[39]:


px.box(Import, x="Commodity", y="value")


# In[40]:


px.box(Export, x="Commodity", y="value")


# In[41]:


px.pie(Import, values='value', names='year',color='year',hover_data=['country'])


# In[42]:


px.pie(Export, values='value', names='year',color='year',hover_data=['country'])


# In[43]:


px.box(Import, x="country", y="value")


# In[44]:


px.box(Export, x="country", y="value")


# In[45]:


Import.columns


# In[46]:


Export.columns


# In[47]:


px.scatter_matrix(Import,
    dimensions=["Commodity", "value", "country"],
    color="year")


# In[48]:


px.scatter_matrix(Export,
    dimensions=["Commodity", "value", "country"],
    color="year")


# In[49]:


px.scatter(Import,"Commodity","value")


# In[50]:


px.scatter(Export,"Commodity","value")


# In[51]:


px.scatter(Import,"value",color="year", trendline="lowess")


# In[52]:


px.scatter(Export,"value",color="year", trendline="lowess")


# In[53]:


px.scatter(Export,"year","value",color="country", trendline="lowess")


# In[54]:


px.scatter(Import,"year","value",color="country", trendline="lowess")


# # predictive models

# In[55]:


X=Import[['year']]


# In[56]:


y=Import[['value']]


# In[57]:


from sklearn.linear_model import LinearRegression 


# In[58]:


lrm1=LinearRegression()


# In[59]:


lrm1.fit(X,y)


# In[60]:


lrm1.coef_


# In[61]:


lrm1.intercept_


# In[62]:


lrm1.predict([[2019]])


# # Train_test_SPLIT

# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.80)


# In[65]:


X_train.shape


# In[66]:


X_train.head()


# In[67]:


X_test.shape


# In[68]:


X_test.head()


# In[69]:


y_train.head()


# In[70]:


y_test.head()


# In[71]:


from sklearn.linear_model import LinearRegression


# In[72]:


lrm=LinearRegression()


# In[73]:


lrm.fit(X_train,y_train)


# In[74]:


lrm.coef_


# In[75]:


lrm.intercept_


# In[76]:


y_pred=lrm.predict(X_test)


# In[77]:


y_test


# In[78]:


y_pred


# In[79]:


lrm.score(X_test,y_test)


# In[80]:


# Linear Regression is not good for this data set


# # K means

# In[81]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# In[82]:


Export.head()


# In[83]:


Import.head()


# In[84]:


plt.scatter(Import.year,Import['value'])
plt.xlabel('year')
plt.ylabel('value')


# In[85]:


plt.scatter(Export.year,Export['value'])
plt.xlabel('year')
plt.ylabel('value')


# In[86]:


kmI = KMeans(n_clusters=3)
y_predictedI = kmI.fit_predict(Import[['year','value']])
y_predictedI


# In[87]:


kmE = KMeans(n_clusters=3)
y_predictedE = kmE.fit_predict(Export[['year','value']])
y_predictedE


# In[88]:


Import['cluster']=y_predictedI
Import.head()


# In[89]:


Export['cluster']=y_predictedE
Export.head()


# In[90]:


kmI.cluster_centers_


# In[91]:


kmE.cluster_centers_


# In[92]:


Import1=Import[Import.cluster==0]
Import2=Import[Import.cluster==1]
Import3=Import[Import.cluster==2]


# In[93]:


Export1=Export[Export.cluster==0]
Export2=Export[Export.cluster==1]
Export3=Export[Export.cluster==2]


# In[98]:


kmI.cluster_centers_


# In[99]:


kmI.cluster_centers_[:,0]
kmE.cluster_centers_[:,1]
#km.cluster_centers_[:,2]


# In[101]:


plt.scatter(Import1.year,Import1['value'],color='green')
plt.scatter(Import2.year,Import2['value'],color='red')
plt.scatter(Import3.year,Import3['value'],color='blue')
plt.scatter(kmI.cluster_centers_[:,0],kmI.cluster_centers_[:,1],color='purple',marker='*',label='centroid')


# In[102]:


plt.scatter(Export1.year,Export1['value'],color='green')
plt.scatter(Export2.year,Export2['value'],color='red')
plt.scatter(Export3.year,Export3['value'],color='blue')
plt.scatter(kmE.cluster_centers_[:,0],kmE.cluster_centers_[:,1],color='purple',marker='*',label='centroid')


# In[103]:


scaler = MinMaxScaler()

scaler.fit(Import[['value']])
Import['svalue'] = scaler.transform(Import[['value']])

scaler.fit(Import[['year']])
Import['syear'] = scaler.transform(Import[['year']])


# In[104]:


scaler = MinMaxScaler()

scaler.fit(Export[['value']])
Export['svalue'] = scaler.transform(Export[['value']])

scaler.fit(Export[['year']])
Export['syear'] = scaler.transform(Export[['year']])


# In[105]:


Import.head()


# In[106]:


Export.head()


# In[107]:


plt.scatter(Import.syear,Import['svalue'])


# In[108]:


plt.scatter(Export.syear,Export['svalue'])


# In[109]:


km1 = KMeans(n_clusters=3)
y_predictedI1 = km1.fit_predict(Import[['syear','svalue']])
y_predictedI1


# In[110]:


km2 = KMeans(n_clusters=3)
y_predictedE1 = km2.fit_predict(Export[['syear','svalue']])
y_predictedE1


# In[111]:


Import['Scluster']=y_predictedI1
Import.head()


# In[112]:


Export['Scluster']=y_predictedE1
Export.head()


# In[113]:


km1.cluster_centers_


# In[114]:


km2.cluster_centers_


# In[115]:


nImport1=Import[Import.Scluster==0]
nImport2=Import[Import.Scluster==1]
nImport3=Import[Import.Scluster==2]


# In[116]:


nExport1=Export[Export.Scluster==0]
nExport2=Export[Export.Scluster==1]
nExport3=Export[Export.Scluster==2]


# In[117]:


plt.scatter(nImport1.syear,nImport1['svalue'],color='green')
plt.scatter(nImport2.syear,nImport2['svalue'],color='red')
plt.scatter(nImport3.syear,nImport3['svalue'],color='blue')
plt.scatter(km1.cluster_centers_[:,0],km1.cluster_centers_[:,1],color='purple',marker='*',label='centroid')


# In[118]:


plt.scatter(nExport1.syear,nExport1['svalue'],color='green')
plt.scatter(nExport2.syear,nExport2['svalue'],color='red')
plt.scatter(nExport3.syear,nExport3['svalue'],color='blue')
plt.scatter(km2.cluster_centers_[:,0],km2.cluster_centers_[:,1],color='purple',marker='*',label='centroid')


# In[119]:


kmI2 = KMeans(n_clusters=3)
kmI2.fit(Import[['syear','svalue']])
#y_predicted2


# In[120]:


kmE2 = KMeans(n_clusters=3)
kmE2.fit(Import[['syear','svalue']])
#y_predicted2


# In[121]:


Import['Sclusternew'] = kmI2.predict(Import[['syear','svalue']])


# In[122]:


Export['Sclusternew'] = kmE2.predict(Export[['syear','svalue']])


# In[123]:


Import.head()


# In[124]:


Export.head()

Elbow plot
# In[125]:


sseI = []
k_rngI = range(1,10)
for kI in k_rngI:
    kmIM = KMeans(n_clusters=kI)
    kmIM.fit(Import[['year','value']])
    sseI.append(kmIM.inertia_)


# In[126]:


plt.xlabel('KI')
plt.ylabel('Sum of squared error')
plt.plot(k_rngI,sseI)


# In[127]:


sse = []
k_rng = range(1,10)
for k in k_rng:
    kmEx = KMeans(n_clusters=k)
    kmEx.fit(Export[['year','value']])
    sse.append(kmEx.inertia_)


# In[128]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[129]:


sseI


# In[130]:


sse


# In[131]:


# The best fit model for this treading data sets is K-MEANS clusturing. 
# But According to my knowledge this data sets are only for EDA process only. 

