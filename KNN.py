
# coding: utf-8

# In[2]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:

#reading file
df=pd.read_excel("F:\WNSMBA_Year2\Internship\Data science\default of credit card clients.xls",header=1)


# In[5]:


df.head()
np.count_nonzero(df.isnull())
sum(df.apply(lambda x: sum(x.isnull().values), axis = 1)>0) #counting null values


# In[ ]:

bplot = sns.boxplot(y='LIMIT_BAL', x='SEX', 
                 data=df, 
                 width=0.5,
                 palette="colorblind")  #Outlier checking
plt.show()
ax = sns.swarmplot(x="SEX", y="LIMIT_BAL", data=df, color=".25")


# In[6]:

sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show() #EDA


# In[ ]:

x=df.drop(['ID','default payment next month'],axis=1)
y=df['default payment next month']
x=scale(x)
y=scale(y)
#Splitting data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=28)


# In[20]:

from sklearn.neighbors import KNeighborsClassifier
#Finding best k -value
neighbors=np.arange(1,9)
train_accuracy=np.empty(len(neighbors))
test_accuracy=np.empty(len(neighbors))
for i,k in enumerate(neighbors):
    knn_1=KNeighborsClassifier(n_neighbors=k)
    knn_1.fit(x_train,y_train)
    train_accuracy[i]=knn_1.score(x_train,y_train)
    test_accuracy[i]=knn_1.score(x_test,y_test)
    
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[23]:

knn1 =KNeighborsClassifier(n_neighbors=8)
knn1.fit(x_train,y_train)
y_pred=knn1.predict(x_test)
knn1.score(x_test,y_test)
from sklearn.model_selection import  GridSearchCV
param_grid={'n_neighbors':np.arange(1,15)}
knn1_cv=GridSearchCV(knn1,param_grid,cv=5)
knn1_cv.fit(x,y)
knn1_cv.best_params_


# In[ ]:



