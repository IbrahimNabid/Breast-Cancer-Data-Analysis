import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score #cross validation
from sklearn.model_selection import train_test_split #split the available dataset for training and testing


df = pd.read_csv('https://ibrahimnabid.github.io/diabetas.csv')

print(df.head())

df.info()

print(df.columns)

plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["BMI", "HighBP", "HeartDiseaseorAttack", "GenHlth", "PhysHlth", "Age", "Education", "Income", "DiffWalk", "HighChol", "Diabetes_binary"]
df = pd.read_csv("https://ibrahimnabid.github.io/diabetas.csv", usecols=columns)
print("Contents in csv file:\n", df)
plt.show()
df =df.to_csv("file with 10 features.csv", index=False)

df =pd.read_csv("file with 10 features.csv")
df.info()

# Convert object columns to categorical data types
df["HighBP"] = df["HighBP"].astype('category')
df["HighChol"] = df["HighChol"].astype('category')
df["BMI"] = df["BMI"].astype('category')
df["HeartDiseaseorAttack"] = df["HeartDiseaseorAttack"].astype('category')
df["GenHlth"] = df["GenHlth"].astype('category')
df["PhysHlth"] = df["PhysHlth"].astype('category')
df["DiffWalk"] = df["DiffWalk"].astype('category')
df["Age"] = df["Age"].astype('category')
df["Education"] = df["Education"].astype('category')
df["Income"] = df["Income"].astype('category')

# Split the dataset into features and labels
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

# create a decision tree classifier
dt = DecisionTreeClassifier(max_depth=2, min_samples_leaf=5, criterion='entropy')
dt.fit(X_train, y_train)

# Perform cross-validation
tree_entropy_cross = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy')
scores = cross_val_score(tree_entropy_cross, X_train, y_train, cv=10, scoring='accuracy')
print("Cross-validation scores:", scores)
print("Average score:", np.mean(scores))

# Make predictions on test set
y_pred = dt.predict(X_test)

# Calculate accuracy of the model on test set
accuracy = np.mean(y_pred == y_test)
print("Accuracy on test set:", accuracy)