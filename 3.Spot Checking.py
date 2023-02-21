#This is the third step

# Logistic Regression Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
array = array.astype('int64')
X = array[:, :-1]
Y = array[:, -1]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Logistic regression :")
print(results.mean())




# LDA Classification
import pandas
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
X = array[:, :-1]
Y = array[:, -1]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = LinearDiscriminantAnalysis()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("LDA :")
print(results.mean())

# KNN Classification
import pandas
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
X = array[:, :-1]
Y = array[:, -1]
random_state = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = KNeighborsClassifier()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("KNN :")
print(results.mean())

# Gaussian Naive Bayes Classification
import pandas
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
url = "final_dataset.csv"
dataframe = pandas.read_csv(url)
array = dataframe.values
X = array[:, :-1]
Y = array[:, -1]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
model = GaussianNB()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print("Guassian Naive Bayes Classification :")
print(results.mean())
# Decision Tree
# Load libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import classification_report, confusion_matrix
col_names = ['AGE', 'SEX', '0A1ILY_HISTORY', 'UREA','CREATININE','CHOLESTEROL','TRIGLYCERIDE','HDL_CHOL','LDL_CHOL','GLUCOSE','T3','T4','TSH','OUTCO1E']
# load dataset
dataset = pd.read_csv(r"C:\Users\HP\PycharmProjects\diabetesProject\final_dataset.csv")
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(splitter="best", max_depth=2)
# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
#Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Decision Tree:")
print(metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Random Forest
import pandas
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
dataset = pd.read_csv(r"C:\Users\HP\PycharmProjects\diabetesProject\final_dataset.csv")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.70, random_state=5) # 70% training and 30% test
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("RandomForestClassifier :")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# SVM
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

# load dataset
url = pd.read_csv(r"C:\Users\HP\Desktop\datasetff.csv")
array = url.values
X = array[:, :-1]
Y = array[:, -1]
seed=7
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=3) # 70% training and 30% test
#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

# Model Accuracy: how often is the classifier correct?
print("SVM:")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# Voting Classifier

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# load dataset
dataset = pd.read_csv(r"C:\Users\HP\Desktop\datasetff.csv")
dataset.keys()
array = dataset.values
X = array[:, :-1]
Y = array[:, -1]
seed=7
X_train, X_test, y_train, y_test = train_test_split(X, Y) # 70% training and 30% test

#instantiate SVM
from sklearn.svm import SVC
svm=SVC()
#Fit the model to the training dataset
svm.fit(X_train,y_train)
#Predict using the test set
predictions=svm.predict(X_test)
#instantiate Evaluation matrics
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Instantiate Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
#Fit the model to the training set and predict using the test set
lr.fit(X_train,y_train)
predictions=lr.predict(X_test)
#Evaluation matrics
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

#Instantiate Decision tree model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
#Fit and predict the model
dt.fit(X_train,y_train)
predictions=dt.predict(X_test)
#Evaluation matrics
print(classification_report(y_test,predictions))

#import Voting Classifier
from sklearn.ensemble import VotingClassifier
#instantiating three classifiers
logReg= LogisticRegression()
dTree= DecisionTreeClassifier()
svm= SVC()
voting_clf = VotingClassifier(estimators=[('supportvectormachine',svm),('DecisionTree',dTree), ('LogReg', logReg)], voting='hard')
#fit and predict using training and testing dataset respectively
voting_clf.fit(X_train, y_train)
predictions = voting_clf.predict(X_test)
#Evaluation matrics
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# Gradient Boosting Classifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
# load data
url = pd.read_csv(r"C:\Users\HP\Desktop\datasetff.csv")
array = url.values
X = array[:, :-1]
Y = array[:, -1]
seed=7
X_train, X_test, y_train, y_test = train_test_split(X, Y)
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_test, y_test)))
    gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report")
print(classification_report(y_test, predictions))

# AdaBoost Classifier
    # AdaBoost Classifier with base classifier Random Forest
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# load data
url = pd.read_csv(r"C:\Users\HP\Desktop\datasetff.csv")
array = url.values
X = array[:, :-1]
Y = array[:, -1]
seed=7
X_train, X_test, y_train, y_test = train_test_split(X, Y)

adaboost = AdaBoostClassifier( random_state=None ,base_estimator=RandomForestClassifier(random_state=None), n_estimators=100)
adaboost.fit(X_train,y_train)
traninng_data= adaboost.score(X_train,y_train)
testing_data=adaboost.score(X_test,y_test)
print("traninng_data",traninng_data)
print("testing_data",testing_data)

      # Adaboost Classifier with base Classifier as decision tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
# load data
url = pd.read_csv(r"C:\Users\HP\Desktop\datasetff.csv")
array = url.values
X = array[:, :-1]
Y = array[:, -1]
seed=7
X_train, X_test, y_train, y_test = train_test_split(X, Y)

adaboost = AdaBoostClassifier() # by default decision tree
adaboost.fit(X_train,y_train)
a=adaboost.score(X_train,y_train)
b=adaboost.score(X_test,y_test)
print("a",a)
print("b",b)

#  XGBoost model for Pima Indians dataset
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
# load data
dataset = pd.read_csv(r"C:\Users\HP\PycharmProjects\diabetesProject\final_dataset.csv")
array = dataset.values
X = array[:, :-1]
Y = array[:, -1]
seed=7
X_train, X_test, y_train, y_test = train_test_split(X, Y)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
y_pred = model.predict(X_test)
# evaluate predictions
print("XGBoost")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

