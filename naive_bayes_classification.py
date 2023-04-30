import sklearn.naive_bayes
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import io

from google.colab import files
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['diabetes.csv']))

df.info()

X = df
X.head()

Diabetes_Category = {} # a dictionary with key-value pairs 
Diabetes_Category['feature_names'] = X.columns.values

enc = OrdinalEncoder()
X = enc.fit_transform(X)
X

Diabetes_Category['data']=X

Y = df
Diabetes_Category['target_names']=Y['Diabetes_binary'].unique()
Diabetes_Category['target']=Y["Diabetes_binary"].values

Diabetes_Category['target_names']

Diabetes_Category['target']

NB_C = CategoricalNB()

scores_NBC = cross_val_score(NB_C, Diabetes_Category['data'], Diabetes_Category['target'], cv=5, scoring='accuracy')
scores_NBC

#mean and 95% confidence level
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_NBC.mean(), scores_NBC.std() * 2))

y_predict= NB_C.fit(Diabetes_Category['data'],Diabetes_Category['target']).predict(Diabetes_Category['data'])

df=df.iloc[0:70692]
df['predict']= y_predict
print(df)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Diabetes_Category['target'], y_predict))
