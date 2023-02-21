#ensemble methods
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBRFClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split

#Reading the dataset and Splitting the dataset into the Training set and Test set

dataset = pd.read_csv(r"C:\Users\HP\PycharmProjects\diabetesProject\final_dataset.csv")
array = dataset.values
X = array[:, :-1]
Y = array[:, -1]
seed=7
X_train, X_test, y_train, y_test = train_test_split(X, Y,random_state = None)

#Defining the machine learning models

model1 = LogisticRegression()
model2 = DecisionTreeClassifier(max_depth = 2)
model3 = SVC()
model4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
model5 = GaussianNB()
model6 = RandomForestClassifier(n_estimators=100)
model7 = LinearDiscriminantAnalysis()
model8 = AdaBoostClassifier( random_state=None ,base_estimator=RandomForestClassifier(random_state=None), n_estimators=100)
model9 = AdaBoostClassifier( )
model10 = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
model11 = XGBClassifier()
model12 = XGBRFClassifier(n_estimators=100)

#Training the machine learning models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)
model7.fit(X_train, y_train)
model8.fit(X_train, y_train)
model9.fit(X_train, y_train)
model10.fit(X_train, y_train)
model11.fit(X_train, y_train)
model12.fit(X_train, y_train)

#Making the prediction

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)
y_pred6 = model6.predict(X_test)
y_pred7 = model7.predict(X_test)
y_pred8 = model8.predict(X_test)
y_pred9 = model9.predict(X_test)
y_pred10 = model10.predict(X_test)
y_pred11 = model11.predict(X_test)
y_pred12 = model11.predict(X_test)

#Confusion matrix

cm_LogisticRegression = confusion_matrix(y_test, y_pred1)
cm_DecisionTree = confusion_matrix(y_test, y_pred2)
cm_SupportVectorClass = confusion_matrix(y_test, y_pred3)
cm_KNN = confusion_matrix(y_test, y_pred4)
cm_NaiveBayes = confusion_matrix(y_test, y_pred5)
cm_RandomForest = confusion_matrix(y_test, y_pred6)
cm_LDA = confusion_matrix(y_test, y_pred7)
cm_adaboost_rf = confusion_matrix(y_test, y_pred8)
cm_adaboost_dt = confusion_matrix(y_test, y_pred9)
cm_gradientboost = confusion_matrix(y_test, y_pred10)
cm_xgboost = confusion_matrix(y_test, y_pred11)
cm_xgboost_RF = confusion_matrix(y_test, y_pred12)
#10-fold cross-validation

kfold = model_selection.KFold(n_splits=10, random_state = None)
result1 = model_selection.cross_val_score(model1, X_train, y_train, cv=kfold)
result2 = model_selection.cross_val_score(model2, X_train, y_train, cv=kfold)
result3 = model_selection.cross_val_score(model3, X_train, y_train, cv=kfold)
result4 = model_selection.cross_val_score(model4, X_train, y_train, cv=kfold)
result5 = model_selection.cross_val_score(model5, X_train, y_train, cv=kfold)
result6 = model_selection.cross_val_score(model6, X_train, y_train, cv=kfold)
result7 = model_selection.cross_val_score(model7, X_train, y_train, cv=kfold)
result8 = model_selection.cross_val_score(model8, X_train, y_train, cv=kfold)
result9 = model_selection.cross_val_score(model9, X_train, y_train, cv=kfold)
result10 = model_selection.cross_val_score(model10, X_train, y_train, cv=kfold)
result11 = model_selection.cross_val_score(model11, X_train, y_train, cv=kfold)
result12 = model_selection.cross_val_score(model12, X_train, y_train, cv=kfold)

#Printing the accuracies achieved in cross-validation
print('Accuracy of Logistic Regression Model = ',result1.mean())
print('Accuracy of Decision Tree Model = ',result2.mean())
print('Accuracy of Support Vector Machine = ',result3.mean())
print('Accuracy of k-NN Model = ',result4.mean())
print('Accuracy of Naive Bayes Model = ',result5.mean())
print('Accuracy of Random Forest Model = ',result6.mean())
print('Accuracy of LDA Model = ',result7.mean())
print('Accuracy of adaboost_rf Model = ',result8.mean())
print('Accuracy of adaboost_dt Model = ',result9.mean())
print('Accuracy of gradientBoost Model = ',result10.mean())
print('Accuracy of xgboost Model = ',result11.mean())
print('Accuracy of xgboost_RF Model = ',result12.mean())

#Defining Hybrid Ensemble Learning Model
# create the sub-models
estimators = []
#Defining  Decision Tree Classifiers
model7 = DecisionTreeClassifier(max_depth = 2)
estimators.append(('DT1',model7))
#Defining  Random Forest Classifiers
model8 = RandomForestClassifier(n_estimators=100)
estimators.append(('rf1',model8))
#Defining  NB Classifiers
model9 = GaussianNB()
estimators.append(('GNB1',model9))
#Defining  Xgboost Classifiers
model10 = XGBClassifier()
estimators.append(('xgb1',model10))
# Defining the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
print(".................................")
print("HybridApproach")
print("...................................")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# define one new instance
Xnew = [[35,0,1,24,0.68,173,172,46,114,530,1.19,8.87,2.47]]
# make a prediction
Ynew = ensemble.predict(Xnew)
print("X=%s,\nPredicted=%s" % (Xnew[0], Ynew[0]))
hybrid = confusion_matrix(y_test,y_pred)
print(hybrid)
print(classification_report(y_test,y_pred))
print("............................")
print("LR",cm_LogisticRegression)
print("DT",cm_DecisionTree)
print("SVM",cm_SupportVectorClass)
print("KNN",cm_KNN)
print("NB",cm_NaiveBayes)
print("RF",cm_RandomForest)
print("LDA",cm_LDA)
print("ADB_RF",cm_adaboost_rf)
print("ADB_DT",cm_adaboost_dt)
print("GB",cm_gradientboost)
print("XGB",cm_xgboost)
print("XGB_RF",cm_xgboost_RF)
print(".......................")
print("LR",classification_report(y_test,y_pred1))
print("DT",classification_report(y_test,y_pred2))
print("SVM",classification_report(y_test,y_pred3))
print("KNN",classification_report(y_test,y_pred4))
print("NB",classification_report(y_test,y_pred5))
print("RF",classification_report(y_test,y_pred6))
print("LDA",classification_report(y_test,y_pred7))
print("ADB_RF",classification_report(y_test,y_pred8))
print("ADB_DT",classification_report(y_test,y_pred9))
print("GB",classification_report(y_test,y_pred10))
print("XGB",classification_report(y_test,y_pred11))
print("XGB",classification_report(y_test,y_pred12))