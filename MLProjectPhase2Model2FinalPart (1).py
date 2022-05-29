#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import category_encoders as ce 
from sklearn.preprocessing import LabelEncoder


# In[2]:


#Reading Data
Data=pd.read_csv("airline-price-classification2.csv")


# In[3]:


#Exploring Data
Data.head()


# In[4]:


Data.info()


# In[5]:


#Seeing Null Values and Duplicate Values
print(Data.isnull().sum())
Data=Data.drop_duplicates(subset=None,keep='first',inplace=False)


# In[6]:


#Handling Missing Values 
for i in Data.columns:
    if(type(Data[i]=='object')):
        Data[i]=Data[i].fillna(Data[i].mode()[0])
    else:
        Data[i]=Data[i].fillna(Data[i].mean())


# In[7]:


#Exploring Shape of Data
Data.shape


# In[8]:


# a Copy of Data to Work on 
R_Data=Data.copy()


# In[9]:


R_Data.head(2)


# In[10]:


def One_Hot_Encoder(X,Desired_Column):
    X=pd.get_dummies(X,columns=[Desired_Column])
    return X


# In[11]:


def TargetEncoding(Desired_Column,RelatedWith,Data):
    D1=Data.groupby([Desired_Column])[RelatedWith].mean().sort_values().index
    dict4={key:index for index,key in enumerate(D1,0)}
    Data[Desired_Column]=Data[Desired_Column].map(dict4)
    return Data


# In[12]:


def Label_Encoder(Feature):
    le=LabelEncoder()
    le.fit(Feature)
    le.classes_
    Feature=le.transform(Feature)
    return Feature  


# In[13]:


#Turning Date from String to Datetime Function
def Change_Into_Datetime(col):
    R_Data[col]=pd.to_datetime(R_Data[col])


# In[14]:


R_Data.columns


# In[15]:


# Turning Date,Arrival and Departure Time into Datetime
for feature in ['date','dep_time','arr_time'] :
    Change_Into_Datetime(feature)


# In[16]:


R_Data.dtypes


# In[17]:


R_Data['date'].min()


# In[18]:


R_Data['date'].max()


# In[19]:


#Creating a Column of Journey Day
R_Data['Journey_Day']=R_Data['date'].dt.day


# In[20]:


##Creating a Column of Journey Month
R_Data['Journey_Month']=R_Data['date'].dt.month


# In[21]:


##Creating a Column of Journey Year
R_Data['Journey_Year']=R_Data['date'].dt.year


# In[22]:


R_Data.head(2)


# In[23]:


#Extrating Hours and Minutes Function
def extract_hour_min(Data,col):
    Data[col+"_hour"]=Data[col].dt.hour
    Data[col+"_Minute"]=Data[col].dt.minute
    Data.drop(col,axis=1,inplace=True)
    return Data.head(2)


# In[24]:


#Extrating Hours and Minutes 
extract_hour_min(R_Data,'dep_time')


# In[25]:


extract_hour_min(R_Data,'arr_time')


# In[26]:


#Exploring Departure Time
def Flight_Dep_Time(x):
    if(x>4) and (x<=8) : 
        return "Early Morning"
    elif (x>8) and (x<=12) : 
        return "Morning"
    elif (x>12) and (x<=16) : 
        return "Noon"
    elif (x>16) and (x<=20) : 
        return "Evening"
    elif (x>20) and (x<=24) :
        return "Night"
    else : 
        return "Late_Night"
        


# In[27]:


#Importing Important Libraries
import plotly 
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
cf.go_offline()


# In[28]:


# Visualizing Departure Time hours
R_Data['dep_time_hour'].apply(Flight_Dep_Time).value_counts().iplot(kind='bar')


# In[29]:


#Preprocess Duration Function
def Pre_Process_Duration(x):
    if 'h' not in x :
        x='0h '+x
    elif 'm' not in x:
        x=x+' 0m'
    return x


# In[30]:


#Applying on Time Taken
R_Data['time_taken']=R_Data['time_taken'].apply(Pre_Process_Duration)


# In[31]:


R_Data['time_taken']


# In[32]:


int(R_Data['time_taken'][0].split(' ')[1][0:-1])


# In[33]:


# Extracting The Hours taken
R_Data['time_taken_hours']=R_Data['time_taken'].apply(lambda x:int(float(x.split(' ')[0][0:-1])))
R_Data['time_taken_hours']


# In[34]:


#Extracting Minutes and setting '' values to 0
R_Data['time_taken_minutes']=R_Data.time_taken.str.split(" ",expand=True)[1]
R_Data['time_taken_minutes']=R_Data.time_taken_minutes.str.split("m",expand=True)[0]
R_Data['time_taken_minutes'].unique()
R_Data['time_taken_minutes'][91880]='32'
R_Data['time_taken_minutes'][127110]='20'
R_Data['time_taken_minutes'].unique()
R_Data['time_taken_minutes']=R_Data['time_taken_minutes'].astype(int)
R_Data['time_taken_minutes']


# In[35]:


len(R_Data)


# In[36]:


#Assigning Total Time Taken in Minutes
R_Data['total_time_taken']=R_Data['time_taken_hours'].astype(float)*60+R_Data['time_taken_minutes'].astype(float)
R_Data['total_time_taken']


# In[37]:


def extract_cat_num(R_Data):
    cat_col=[col for col in R_Data.columns if R_Data[col].dtype=='object']
    num_col=[col for col in R_Data.columns if R_Data[col].dtype!='object']
    return cat_col,num_col
# Analysing Distribution of Numerical Columns 
cat_col,num_col=extract_cat_num(R_Data)
len(num_col)
plt.figure(figsize=(30,20))
for i,feature in enumerate(num_col):
    plt.subplot(5,3,i+1)
    R_Data[feature].hist()
    plt.title(feature)


# In[38]:


len(cat_col)


# In[39]:


plt.figure(figsize=(20,20))
for i,feature in enumerate(cat_col):
    plt.subplot(3,3,i+1)
    sns.countplot(R_Data[feature])


# In[40]:


R_Data.head(2)


# In[41]:


R_Data['TicketCategory']=Label_Encoder(R_Data['TicketCategory'])
R_Data['TicketCategory']


# In[42]:


#Exploring Distribution of Total Time Taken based on Price
## Problem 1
grid=sns.FacetGrid(R_Data,hue='TicketCategory',aspect=2)
grid.map(sns.kdeplot,'total_time_taken')
grid.add_legend()


# In[43]:


#Preprocessing Route
R_Data['route'].unique()


# In[44]:


#Splitting Source and Destination
R_Data[['source','destination']] = R_Data.route.str.split(",",expand=True)


# In[45]:


#Splitting Source Column
R_Data['source']=R_Data.source.str.split(":",expand=True)[1]
R_Data['source'] = R_Data['source'].astype(str)
R_Data['source']


# In[46]:


# Splitting Destination Column
R_Data['destination']=R_Data.destination.str.split(":",expand=True)[1]
R_Data['destination']=R_Data.destination.str.replace("}","")
R_Data['destination']


# In[47]:


#Plotting Violin Plot to See The Relation between Airline and Price
## Problem 2
plt.figure(figsize=(15,5))
sns.violinplot(x='airline',y='TicketCategory',data=R_Data)
plt.xticks(rotation='vertical')


# In[48]:


R_Data.drop(columns=['route','total_time_taken','Journey_Year'],axis=1,inplace=True)


# In[49]:


#Applying Target encoding on Source
# Problem 4
sources=R_Data.groupby(['source'])['TicketCategory'].mean().sort_values().index
dict2={key:index for index,key in enumerate(sources,0)}
R_Data['source']=R_Data['source'].map(dict2)
R_Data['source']


# In[50]:


R_Data.head(3)


# In[51]:


#Applying Target guided encoding on Airline
# Problem 5 
def TargetEncoding(Desired_Column,RelatedWith,Data):
    D1=Data.groupby([Desired_Column])[RelatedWith].mean().sort_values().index
    dict4={key:index for index,key in enumerate(D1,0)}
    Data[Desired_Column]=Data[Desired_Column].map(dict4)
    return Data
R_Data=TargetEncoding('airline','TicketCategory',R_Data)
print(R_Data['airline'])


# In[52]:


#Applying Target Guided Encoding on Destination
#Problem 6
dest=R_Data.groupby(['destination'])['TicketCategory'].mean().sort_values().index
dict2={key:index for index,key in enumerate(dest,0)}
dict2
R_Data['destination']=R_Data['destination'].map(dict2)
R_Data['destination']


# In[53]:


#Extracting Meaningful features from Stop Column
R_Data['stop'].unique()


# In[54]:


total_stops={'non-stop':0, '2+-stop':2}


# In[55]:


R_Data['stop']=R_Data['stop'].map(total_stops)


# In[56]:


R_Data['stop']


# In[57]:


R_Data['stop']=R_Data['stop'].fillna(1)
R_Data['stop']


# In[58]:


R_Data.drop(columns=['time_taken','date'],axis=1,inplace=True)


# In[59]:


R_Data.dtypes


# In[60]:


#Applying Target Encoding on Ch_Code    
R_Data['ch_code']=Label_Encoder(R_Data['ch_code'])
print(R_Data['ch_code'])


# In[61]:


sns.countplot(R_Data['type'])


# In[62]:


#Applying Target Encoding on Type
#R_Data=One_Hot_Encoder(R_Data,'type')
#print(R_Data)
R_Data=TargetEncoding('type','TicketCategory',R_Data)
print(R_Data)


# In[63]:


plt.figure(figsize=(20,20))
R_Data.corr()
sns.heatmap(R_Data.corr(),annot=True)


# In[64]:


#Applying Feature Selection using Mutual Info Regression
from sklearn.feature_selection import mutual_info_classif


# In[65]:


X=R_Data.drop(['TicketCategory'],axis=1)
Y=R_Data['TicketCategory']
X.dtypes


# In[66]:


mutual_info_classif(X,Y)


# In[67]:


#Ordering Features based on Importance
imp=pd.DataFrame(mutual_info_classif(X,Y),index=X.columns)
imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)


# In[68]:


## Question 
R_Data.drop(columns=['time_taken_hours','Journey_Month','arr_time_hour','dep_time_hour','stop','Journey_Day','source','arr_time_Minute','destination','dep_time_Minute','time_taken_minutes'],axis=1,inplace=True)


# In[69]:


#Applying Random Forest Technique
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[70]:


from sklearn.tree import DecisionTreeClassifier
#HyperParameter Variation
#Option 1
#The Function used to Calculate Decision Trees
import time
ML_Model=DecisionTreeClassifier(criterion='entropy')
start=time.time()
Model=ML_Model.fit(X_train,Y_train)
end=time.time()
testing_start=time.time()
Y_pred=Model.predict(X_test)
testing_end=time.time()
print("Training_Score : {}".format(Model.score(X_train,Y_train)))
Training_Time=end-start
print('The Training Time is ',Training_Time)
print("Predictions are : {}".format(Y_pred))
Testing_Time=testing_end-testing_start
print('The Testing Time is ',Testing_Time)


# In[71]:


#from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn import metrics
class_score=accuracy_score(Y_test,Y_pred)
print('score: {}'.format(class_score))
print('MSE : ',metrics.mean_squared_error(Y_test,Y_pred))
print('RMSE : ',np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))


# In[72]:


## Saving Model
import pickle
file=open(r'D:\ML Project\modelDesTrees.pkl','wb')
pickle.dump(Model,file)
file.close()
forest=pickle.load(open('D:\ML Project\modelDesTrees.pkl','rb'))
forest.predict(X_test)
print(forest)

