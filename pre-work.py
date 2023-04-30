import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import io

from google.colab import files
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['diabetes_binary_health_indicators_BRFSS2015.csv']))

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["BMI", "HighBP", "HeartDiseaseorAttack", "GenHlth", "PhysHlth", "Age", "Education", "Income", "DiffWalk", "HighChol"]
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv", usecols=columns)
print("Contents in csv file:\n", df)
plt.show()
df=df.to_csv("file with 10 features.csv", index=False)

df["HighBP"]=df["HighBP"].astype("object")
df["HighChol"]=df["HighChol"].astype("object")
df["BMI"]=df["BMI"].astype("object")
df["HeartDiseaseorAttack"]=df["HeartDiseaseorAttack"].astype("object")
df["GenHlth"]=df["GenHlth"].astype("object")
df["PhysHlth"]=df["PhysHlth"].astype("object")
df["DiffWalk"]=df["DiffWalk"].astype("object")
df["Age"]=df["Age"].astype("object")
df["Education"]=df["Education"].astype("object")
df["Income"]=df["Income"].astype("object")

df.info()

#splitting data into train and test datase

training_data = df.sample(frac=0.8, random_state=25)
testing_data = df.drop(training_data.index)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")
