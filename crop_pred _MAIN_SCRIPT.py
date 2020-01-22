#!/usr/bin/env python
# coding: utf-8

# In[20]:


# import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
import pandas as pd
import sklearn
import pickle


# In[21]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from sklearn import preprocessing
from scipy import stats 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[22]:


df = pd.read_csv("crop_data.csv")
# df=np.array(df)
# print(X)
df4= pd.read_csv("crop_data.csv")


# In[23]:


df.Crop=df.Crop.str.strip()
df4.Crop=df4.Crop.str.strip()


# In[24]:


df=df[df.Crop!="Coconut"]
df=df[df.Crop!="Rice"]
df4=df4[df4.Crop!="Coconut"]
df4=df4[df4.Crop!="Rice"]
print(df.shape)
print(df)
print(df4)


# In[25]:


#df4.to_csv('{}.csv'.format('crop_data1'),index=False)


# In[70]:


x=np.array(df4.Area)
y=np.array(df4.Production)

fig, ax = plt.subplots()
ax.scatter(x[1:1000], y[1:1000])
ax.plot(1000, 1000, 'k--', lw=4)
plt.ylim(0, 100000)
plt.xlim(0, 100000)
ax.set_xlabel('Area')
ax.set_ylabel('Production')
plt.show()


# In[75]:


z=np.array(df4.Rainfall)
y=np.array(df4.Production)
fig, ax = plt.subplots()
ax.scatter(z[1:1000], y[1:1000])
ax.plot(1000, 1000, 'k--', lw=4)
plt.ylim(0, 1000)
plt.xlim(0, 1000)
ax.set_xlabel('rainfall')
ax.set_ylabel('Production')


# In[26]:


labelencoder_X = LabelEncoder()
df.Crop = labelencoder_X.fit_transform(df.Crop)
df.District_Name= labelencoder_X.fit_transform(df.District_Name)
df.State_Name= labelencoder_X.fit_transform(df.State_Name)
df.Season= labelencoder_X.fit_transform(df.Season)
print(df)


# In[27]:


df_temp_Crop=df[['Crop']].values
df_temp_State=df[['State_Name']].values
df_temp_Dis=df[['District_Name']].values
df_temp_Season=df[['Season']].values


# inverse transform
# inverse = scaler.inverse_transform(normalized)

# X=df[['Crop','District_Name','State_Name']].values
onehotencoder = OneHotEncoder( )
df_temp_Crop = onehotencoder.fit_transform(df_temp_Crop).toarray()
print(df_temp_Crop.shape)
df_temp_Crop = df_temp_Crop[:,1:]
print(df_temp_Crop.shape)

df_temp_State = onehotencoder.fit_transform(df_temp_State).toarray()
print(df_temp_State.shape)
df_temp_State = df_temp_State[:,1:]
print(df_temp_State.shape)

df_temp_Dis = onehotencoder.fit_transform(df_temp_Dis).toarray()
print(df_temp_Dis.shape)
df_temp_Dis = df_temp_Dis[:,1:]
print(df_temp_Dis.shape)

df_temp_Season = onehotencoder.fit_transform(df_temp_Season).toarray()
print(df_temp_Season.shape)
df_temp_Season = df_temp_Season[:,1:]
print(df_temp_Season.shape)


# In[28]:


df_temp_year=df[['Crop_Year']].values
df_rainfall=df[['Rainfall']].values
scaler = MinMaxScaler()
scaler.fit(df_temp_year)
df_temp_year = scaler.transform(df_temp_year)
scaler.fit(df_rainfall)
df_rainfall = scaler.transform(df_rainfall)
print(df_rainfall)
print(df_temp_year)
print(df)


# In[29]:


df=df.dropna()


# In[30]:


#area=np.array(df.Area).reshape(-1,1)
#rainfall=np.array(df.Rainfall).reshape(-1,1)
#print(area.shape)
X=np.concatenate((df_temp_State,df_temp_Dis,df_temp_year,df_temp_Season,df_temp_Crop,df_rainfall),axis=1)
# X=pd.concat([df_temp_Crop,df_temp_State,df_temp_Dis],axis='columns')
print(X.shape)


# In[31]:


df7=pd.DataFrame(X);
print(df7)
print(df4)


# In[32]:


#df7.to_csv('{}.csv'.format('encoded'),index=False)


# In[33]:


df2=pd.DataFrame(df4.State_Name.unique());
df2.rename(columns={0:'State_Name'},inplace=True);
df3=pd.DataFrame(df4.District_Name.unique());
df3.rename(columns={0:'District_Name'},inplace=True);
df5=pd.DataFrame(df4.Season.unique());
df5.rename(columns={0:'Season'},inplace=True);
df6=pd.DataFrame(df4.Crop.unique());
df6.rename(columns={0:'Crop'},inplace=True);
print(df2);
print(df3);
print(df5);
print(df6);


# In[34]:


# df8=pd.DataFrame(df_temp_Crop);
# df9=pd.DataFrame(df_temp_State);
# df10=pd.DataFrame(df_temp_Dis);
# df11=pd.DataFrame(df_temp_Season);
# print(df8)
# print(df9)
# print(df10)
# print(df11)


# In[35]:


#df2=pd.DataFrame(np.array(X));
#print(df2)
#df2.to_csv('{}.csv'.format('test_data'),index=False)


# In[36]:


#c=df2[df2.State_Name == 'Andaman and Nicobar Islands'].index
#df9.iloc[c,:]


# In[37]:


# g=X[4][:]
# print(g.size)


# In[38]:



# Y=df[['Production']].values/df[['Area']].values
# print(Y)
# print(Y.shape)

Y=np.array(df[['Production']].values)

#imputer=Imputer(missing_values='NaN',strategy='mean')
#imputer=imputer.fit(Y)
#Y=imputer.transform(Y)
Y=Y/(np.array(df[['Area']].values))
area=np.array(df.Area);
Y=np.concatenate((Y,area.reshape(-1,1)),axis=1)
print(Y[:,1])
#scaler = MinMaxScaler()
#scaler.fit(Y)
#Y = scaler.transform(Y)
print(Y)
# Y.describe()
# Y=stats.zscore(Y, axis = 1)
#print(Y)
print(Y.shape)
print(X.shape)
X_train,X_test,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[39]:


col=np.size(Y_train1,1);
Y_train=Y_train1[:,0:col-1];
col=np.size(Y_test1,1);
Y_test=Y_test1[:,0:col-1];
print(Y_train.shape)
print(Y_test.shape)


# In[40]:


print(X.shape)


# In[41]:


NN_model = Sequential()


# In[42]:



# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal',input_dim =712 , activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(128, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(64, kernel_initializer='normal',activation='relu'))
NN_model.add(Dense(32, kernel_initializer='normal',activation='relu'))
# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


# In[43]:


checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
callbacks_list = [checkpoint]


# In[25]:


NN_model.fit(X_train, Y_train, epochs=100, validation_split = 0.2, callbacks=callbacks_list,batch_size=10000)


# In[44]:


wights_file = 'Weights-100--6.54972.hdf5' # choose the best checkpoint 
NN_model.load_weights(wights_file) # load it
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])


# In[ ]:


def make_submission(prediction, sub_name,Y_test,X_test,Yield_pred,test_Yield):
  my_submission = pd.DataFrame({'Id':1,'Priceperarea':prediction,'Y_test':Y_test,'Yield_pred':Yield_pred,'test_yield':test_Yield,})
  my_submission.to_csv('{}.csv'.format(sub_name),index=False)
  print('A submission file has been made')
predictions = NN_model.predict(X_test)
# inverse transform
# X_test = scaler.inverse_transform(X_test)
import matplotlib.pyplot as plt
Yield_pred=np.multiply(predictions,Y_test1[:,1].reshape(-1,1))
test_Yield=np.multiply(Y_test[:,0],Y_test1[:,1])
print(Yield_pred.shape)
fig, ax = plt.subplots()
ax.scatter(test_Yield[1:1000], Yield_pred[1:1000])
ax.plot(1000, 1000, 'k--', lw=4)
ax.set_xlabel('Measured');
ax.set_ylabel('Predicted');
plt.savefig('foo.png')
plt.show()
t=np.mean((test_Yield-Yield_pred)**2)
print(t)
# print(Yield_pred.reshape(-1,).shape)
# print(Y_test[:,0].shape)
# print(Y_test1[:,1].shape)
# print(test_Yield.shape)
# make_submission(predictions[:,0],'submission(NN)',Y_test[:,0],X_test[:,0],Yield_pred.reshape(-1,),test_Yield)
# print(predictions[:,0])


# In[65]:


pickle.dump(NN_model,open('model.pkl','wb'))


# In[66]:


from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pandas as pd
import pickle as p
import json


# In[67]:


df12 = pd.read_csv("crop_data1.csv")
# df=np.array(df)
# print(X)
df13= pd.read_csv("encoded.csv")


# In[68]:


df2=pd.DataFrame(df12.State_Name.unique());
df2.rename(columns={0:'State_Name'},inplace=True);
df3=pd.DataFrame(df12.District_Name.unique());
df3.rename(columns={0:'District_Name'},inplace=True);
df5=pd.DataFrame(df12.Season.unique());
df5.rename(columns={0:'Season'},inplace=True);
df6=pd.DataFrame(df12.Crop.unique());
df6.rename(columns={0:'Crop'},inplace=True);
print(df2);
print(df3);
print(df5);
print(df6);


# In[69]:


# df9=pd.DataFrame(df13.iloc[:,:30]);
# df10=pd.DataFrame(df13.iloc[:,30:584]);
# df11=pd.DataFrame(df13.iloc[:,585:590]);
# df8=pd.DataFrame(df13.iloc[:,590:711]);
# print(df8)
# print(df9)
# print(df10)
# print(df11)


# In[53]:



# # app = Flask(__name__)


# #@app.route('/makecalc', methods=['POST'])
# #def makecalc():
# #     data = request.get_json()
# model1='model.pkl'
# model = p.load(open(model1, 'rb'))
# df31=df9.iloc[df2[df2.State_Name == 'Andaman and Nicobar Islands'].index,:] 
# # print(df31)
# df32=df10.iloc[df3[df3.District_Name =='NICOBARS'].index,:] 
# df33=df15=pd.DataFrame(np.array(2012).reshape(-1,1))
# df34=df11.iloc[df5[df5.Season == 'kharif'].index,:] 
# df35=df8.iloc[df6[df6.Crop == 'Arecanut'].index,:]
# df36=df16=pd.DataFrame(np.array(35).reshape(-1,1))
# df14= pd.concat([df31, df32,df33,df34,df35,df36], axis=1, sort=False)
# df37=np.array(df14)
# # print(df37)
# prediction = model.predict(df37)
# print(prediction)
#    # return jsonify(prediction)

# # if __name__ == '__main__':
# #     modelfile = 'model.pkl'
# #     model = p.load(open(modelfile, 'rb'))
# #     app.run(debug=True,host='localhost', port=5000)


# In[81]:


# import requests

# import json

# url ='http://localhost:5000/'
#   #or 8888 and the host to socket.gethostname()
# data = g.tolist()
# j_data = json.dumps(data)
# headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
# r = requests.post(url, data=j_data, headers=headers)
# print(r,r.text)


# In[ ]:




