# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the dataset
dataset = pd.read_csv('heart_disease_health_indicators (1).csv')
x= dataset.drop(columns='HeartDiseaseorAttack')
y= dataset['HeartDiseaseorAttack']


# checking if data is clean
# print(dataset['HeartDiseaseorAttack'].unique())
# print(dataset['HighBP'].unique())
# print(dataset['HighChol'].unique())
# print(dataset['CholCheck'].unique())
# print(dataset['BMI'].unique())
# print(dataset['Smoker'].unique())
# print(dataset['Stroke'].unique())
# print(dataset['Diabetes'].unique()) --> 0 1 2
# print(dataset['PhysActivity'].unique())
# print(dataset['Fruits'].unique())
# print(dataset['Veggies'].unique())
# print(dataset['HvyAlcoholConsump'].unique())
# print(dataset['AnyHealthcare'].unique())
# print(dataset['NoDocbcCost'].unique())
# print(dataset['GenHlth'].unique())
# print(dataset['MentHlth'].unique())
# print(dataset['PhysHlth'].unique())
# print(dataset['DiffWalk'].unique())
# print(dataset['Sex'].unique())
# print(dataset['Age'].unique())
# print(dataset['Education'].unique())
# print(dataset['Income'].unique())

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting a new result
# print(classifier.predict([[1, 1, 1, 40, 1, 0, 0, 0, 0, 1, 0, 1, 0, 5, 18, 15, 1, 0, 9, 4, 3]]))

# importing libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import  make_pipeline
import pickle

# making object for one hot encoder
ohe = OneHotEncoder()
ohe.fit(x[['Diabetes']])
column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_,handle_unknown='ignore'),['Diabetes']),remainder='passthrough')
# making a pipe
pipe = make_pipeline(column_trans,classifier)
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print(y_pred)

# dumping the model
pickle.dump(pipe, open('HeartFailureModel.pkl', 'wb'))




