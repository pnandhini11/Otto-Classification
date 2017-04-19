
# coding: utf-8

# In[1]:

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import  LogisticRegression
import xgboost as xgb
import xgboost as plot_importance
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt; plt.rcdefaults
from pandas.tools.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn import preprocessing, svm
from sklearn.cross_validation import train_test_split,StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation






os.chdir('C:\Kaggle\Otto')

df = pd.read_csv('train.csv')

print(df)

df.target = df.target.map({'Class_1' : 1,
                           'Class_2' : 2,
                           'Class_3' : 3,
                           'Class_4' : 4,
                           'Class_5' : 5,
                           'Class_6' : 6,
                           'Class_7' : 7,
                           'Class_8' : 8,
                           'Class_9' : 9})

print(df)


a=[]

a = df['target'].value_counts()

print(a)








# In[2]:

##for i in [1,4,5,7,9]:
##    df_temp = df[df['target']== i ]
## ##   for j in range(3):
##        df = df.append(df_temp, ignore_index = True)


X = df

y = X[['target']]

X = X.drop(['target'], axis = 1 )

x = X.drop(['id'],axis = 1)



print(x)

model_xgb = xgb.XGBClassifier()

model_xgb.fit(x,y)

f_xgb = model_xgb.feature_importances_

f_xgb = pd.DataFrame(f_xgb)


f_xgb['column'] = x.columns

f_xgb = f_xgb.sort_values([0],ascending = True)

f_new_features = f_xgb['column'].head(30)

plt.figure(figsize=(20,20))

plt.title("Feature Importance")

plt.barh(range(x.shape[1]), f_xgb[0], align = 'center', alpha = 0.5)

plt.yticks(range(x.shape[1]),f_xgb['column'])

plt.show()


model = ExtraTreesClassifier()

model.fit(x,y)

f = model.feature_importances_

f = pd.DataFrame(f)

f['column'] = x.columns

f = f.sort_values([0],ascending = True)

##f_new_features = f['column'].head(18)




print(f_new_features)

print(x[f_new_features])

plt.figure(figsize=(20,20))

plt.title("Feature Importance")

plt.barh(range(x.shape[1]), f[0], align = 'center', alpha = 0.5)

plt.yticks(range(x.shape[1]),f['column'])

plt.show()




# In[3]:


X = X.drop(f_new_features,axis=1)

X = X.drop(['id'],axis=1)


print(X)


corr = X.corr()

corr.to_csv('corrfile.csv', index = False)


print(corr)

desc = X.describe()

desc.to_csv('descfile.csv', index = False)

                            
##X_train,X_test,y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

SSS = StratifiedShuffleSplit(y,10,test_size = 0.2, random_state = 1)

print(SSS)

for train_index, test_index in SSS: 
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    


  
a = []

print(y_train)

##a = predict.Class.value_counts(sort = True, ascending = True)

a = y_train.target.value_counts().sort_index()

print(a)
    
plt.figure(1)

objects = ('Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9')

y_pos = np.arange(len(objects))

plt.barh(y_pos,a,align = 'center',alpha=0.5)


plt.yticks(y_pos,objects)

plt.show()

print(y_test)

##a = predict.Class.value_counts(sort = True, ascending = True)

a = y_test.target.value_counts().sort_index()

print(a)
    
plt.figure(1)

objects = ('Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9')

y_pos = np.arange(len(objects))

plt.barh(y_pos,a,align = 'center',alpha=0.5)


plt.yticks(y_pos,objects)

plt.show()

print(X_train)
print(y_train)


# In[4]:

##crossValidator = cross_validation.StratifiedKFold(y_train,n_folds=4)   

##scores=cross_validation.cross_val_score(
##    model_xgb,X_train,verbose=10,y=y_train,cv=crossValidator,scoring='log_loss')

##print('crossValidation error = ',scores.mean())
print(X_train)

print(X_test)

print(y_train)

print(y_test)

model_xgb = xgb.XGBClassifier( learning_rate =0.3,
 n_estimators=200,
 max_depth=5,
 min_child_weight=8,
 gamma=0.0,
 subsample=0.8,
 colsample_bytree=0.8,

 objective= "multi:softmax",
reg_alpha = 0.005,
 nthread=4,
 scale_pos_weight=1,
 seed=27)

Eval_set = [(X_test, y_test)]

model_xgb.fit(X_train, y_train)

score = model_xgb.score(X_test, y_test)
          
predict = model_xgb.predict(X_test)



predict = pd.DataFrame(data=predict, columns = ['Class'])

a = []


##a = predict.Class.value_counts(sort = True, ascending = True)

a = predict.Class.value_counts().sort_index()

print(a)
    
plt.figure(1)

objects = ('Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9')

y_pos = np.arange(len(objects))

plt.barh(y_pos,a,align = 'center',alpha=0.5)


plt.yticks(y_pos,objects)




##Output.plot(kind='barh',ax=ax, alpha = a, legend = False, color='r', edgecolor='w', xlim=(0,max(predict)))


plt.show()

Output = pd.get_dummies(data=predict, columns = ['Class']).astype(int)

print(score)

print(predict,y_test)

print(Output)








# In[55]:

df = pd.read_csv('test.csv')

print(df)

X = df

y = df[['id']]

X = X.drop(f_new_features,axis=1)

X = X.drop(['id'],axis=1)

print(X.columns)


print(X)


predict = model_xgb.predict(X).round(0).astype(int)

predict = pd.DataFrame(predict, columns = ['Class'])

a = predict.Class.value_counts().sort_index()

print(a)
    
plt.figure(1)

objects = ('Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9')

y_pos = np.arange(len(objects))

plt.barh(y_pos,a,align = 'center',alpha=0.5)


plt.yticks(y_pos,objects)




##Output.plot(kind='barh',ax=ax, alpha = a, legend = False, color='r', edgecolor='w', xlim=(0,max(predict)))


plt.show()


predict = pd.get_dummies(data=predict, columns = ['Class']).round(0).astype(int)

predict = pd.concat([y, predict], axis = 1)

predict.to_csv('sampleSubmission.csv', index = False)

print(predict)


# In[ ]:



