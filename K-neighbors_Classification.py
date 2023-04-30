# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import io
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import files

#uploading the dataset as it is and saving it
uploaded = files.upload()
df = pd.read_csv(io.BytesIO(uploaded['diabetes_binary_health_indicators_BRFSS2015.csv']))

#picking 10 top features based on experimenting with the accuarcy and looking at the correlations and then make a file only with those 10 feauture and the class label
columns = ["Diabetes_binary","BMI", "HighBP", "HeartDiseaseorAttack", "GenHlth", "PhysHlth", "Age", "Education", "Income", "DiffWalk", "HighChol"]
df = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv", usecols=columns)
print("Contents in csv file:\n", df)

df.info()

#performing k-neighbors on our dataset
# Split your dataset into training and testing sets
X = df.drop("Diabetes_binary", axis=1)  # class label
y = df["Diabetes_binary"]  # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a k-nearest neighbors classifier object
knn = KNeighborsClassifier(n_neighbors=5)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Perform stratified cross-validation
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(knn, X, y, cv=cv, scoring="accuracy")
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
