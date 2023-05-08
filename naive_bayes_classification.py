#library import and dataset upload
import sklearn.naive_bayes
from sklearn.metrics import precision_score, recall_score
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import io
from google.colab import files
uploaded = files.upload()
df= pd.read_csv(io.BytesIO(uploaded['diabetes5050.csv']))
df.info()

#change dtype for BMI based on arbitrary categories
  #0=underweight
  #1=normal weight
  #2=overweight
  #3=obese
#df['BMI']=df['BMI'].astype('float')
bins=[0,18.5,25,30,100]
labels=[0,1,2,3]
df['BMI']=pd.cut(df['BMI'], bins=bins, labels=labels)

###naive bayes classification using all features
X = df.iloc[:,1:22]
X.head()

Diabetes_Category = {} # a dictionary with key-value pairs 
Diabetes_Category['feature_names'] = X.columns.values

enc = OrdinalEncoder()
X = enc.fit_transform(X)
X

Diabetes_Category['data']=X

Y = df.iloc[:,0:1]
Diabetes_Category['target_names']=Y['Diabetes_binary'].unique()
Diabetes_Category['target']=Y["Diabetes_binary"].values

Diabetes_Category['target_names']
Diabetes_Category['target']

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

NB_C = CategoricalNB()

scores_NBC = cross_val_score(NB_C, Diabetes_Category['data'], Diabetes_Category['target'], cv=5, scoring='accuracy')
print(scores_NBC)

#mean and 95% confidence level
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_NBC.mean(), scores_NBC.std() * 2))

y_predict= NB_C.fit(Diabetes_Category['data'],Diabetes_Category['target']).predict(Diabetes_Category['data'])
y_pred=NB_C.predict(X_test)

#precision score
precision=precision_score(y_test, y_pred)
print("Precision:", precision)

#recall score
recall=recall_score(y_test, y_pred)
print("Recall:", recall)

df=df.iloc[0:70692]
df['predict_all']= y_predict
print(df)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Diabetes_Category['target'], y_predict))


###naive bayes classification using top 10 features
df_new=df[['Diabetes_binary','HighBP','HighChol','BMI','PhysActivity','GenHlth','PhysHlth','DiffWalk','Age','Education','Income']]

X = df_new.iloc[:,1:11]
X.head()

Diabetes_Category = {} # a dictionary with key-value pairs 
Diabetes_Category['feature_names'] = X.columns.values

enc = OrdinalEncoder()
X = enc.fit_transform(X)
X

Diabetes_Category['data']=X

Y = df_new.iloc[:,0:1]
Diabetes_Category['target_names']=Y['Diabetes_binary'].unique()
Diabetes_Category['target']=Y["Diabetes_binary"].values

Diabetes_Category['target_names']
Diabetes_Category['target']

NB_C = CategoricalNB()

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scores_NBC = cross_val_score(NB_C, Diabetes_Category['data'], Diabetes_Category['target'], cv=5, scoring='accuracy')
print(scores_NBC)

#mean and 95% confidence level
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_NBC.mean(), scores_NBC.std() * 2))

y_predict_new= NB_C.fit(Diabetes_Category['data'],Diabetes_Category['target']).predict(Diabetes_Category['data'])
y_pred=NB_C.predict(X_test)

#precision score
precision=precision_score(y_test, y_pred)
print("Precision:", precision)

#recall score
recall=recall_score(y_test, y_pred)
print("Recall:", recall)

df_new=df_new.iloc[0:70692]
df_new['predict_top10']= y_predict_new
print(df_new)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Diabetes_Category['target'], y_predict_new))


###naive bayes classification using top 15 features
df_new2=df[['Diabetes_binary','HighBP','HighChol','BMI','HeartDiseaseorAttack','GenHlth','PhysHlth','DiffWalk','Age','Education','Income','PhysActivity','Stroke','MentHlth','CholCheck','Smoker']]

X = df_new2.iloc[:,1:16]
X.head()

Diabetes_Category = {} # a dictionary with key-value pairs 
Diabetes_Category['feature_names'] = X.columns.values

enc = OrdinalEncoder()
X = enc.fit_transform(X)
X

Diabetes_Category['data']=X

Y = df_new2.iloc[:,0:1]
Diabetes_Category['target_names']=Y['Diabetes_binary'].unique()
Diabetes_Category['target']=Y["Diabetes_binary"].values

Diabetes_Category['target_names']
Diabetes_Category['target']

NB_C = CategoricalNB()

#splitting
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scores_NBC = cross_val_score(NB_C, Diabetes_Category['data'], Diabetes_Category['target'], cv=5, scoring='accuracy')
scores_NBC

#mean and 95% confidence level of accuracy
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_NBC.mean(), scores_NBC.std() * 2))

y_predict_new2= NB_C.fit(Diabetes_Category['data'],Diabetes_Category['target']).predict(Diabetes_Category['data'])
y_pred=NB_C.predict(X_test)

#precision score
precision=precision_score(y_test, y_pred)
print("Precision:", precision)

#recall score
recall=recall_score(y_test, y_pred)
print("Recall:", recall)

df_new2=df_new2.iloc[0:70692]
df_new2['predict_top15']= y_predict_new2
print(df_new2)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Diabetes_Category['target'], y_predict_new2))
