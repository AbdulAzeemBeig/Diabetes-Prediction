#This is the fourth and last step

# KNN Classification
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
from sklearn.neighbors import KNeighborsClassifier

#load dataset
url = "final_dataset.csv"
dataframe = pd.read_csv(url)
dataframe.head()
dataframe.describe()
array = dataframe.values
# Splitting the dataset into training and testing sets.
x = dataframe.iloc[:, :-1]
y = dataframe.iloc[:, -1]
x_train, x_test, y_train, y_true = train_test_split(x, y, random_state = 7, test_size = 0.7)

# Creating the DT model.
model = DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("DT:", accuracy_score(y_true, y_pred))

confusionMatrix = confusion_matrix(y_true,y_pred)
print("-------------")
print(confusionMatrix)
print("-------------")
print(classification_report(y_true,y_pred))

# define one new instance
Xnew = [[35,0,1,24,0.68,173,172,46,114,530,1.19,8.87,2.47]]
# make a prediction
Ynew = model.predict(Xnew)
print("X=%s,\nPredicted=%s" % (Xnew[0], Ynew[0]))

# Creating the KNN model.
model = KNeighborsClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("KNN:", accuracy_score(y_true, y_pred))

confusionMatrix = confusion_matrix(y_true,y_pred)
print("-------------")
print(confusionMatrix)
print("-------------")
print(classification_report(y_true,y_pred))

# define one new instance
Xnew = [[35,0,1,24,0.68,173,172,46,114,530,1.19,8.87,2.47]]
# make a prediction
Ynew = model.predict(Xnew)
print("X=%s,\nPredicted=%s" % (Xnew[0], Ynew[0]))
# Creating the RF model.
model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("RF:", accuracy_score(y_true, y_pred))

confusionMatrix = confusion_matrix(y_true,y_pred)
print("-------------")
print(confusionMatrix)
print("-------------")
print(classification_report(y_true,y_pred))

# define one new instance
Xnew = [[35,0,1,24,0.68,173,172,46,114,530,1.19,8.87,2.47]]
# make a prediction
Ynew = model.predict(Xnew)
print("X=%s,\nPredicted=%s" % (Xnew[0], Ynew[0]))

# Creating the AdaBoost Classifier model.
model = AdaBoostClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("AdaBoostClassifier:", accuracy_score(y_true, y_pred))

confusionMatrix = confusion_matrix(y_true,y_pred)
print("-------------")
print(confusionMatrix)
print("-------------")
print(classification_report(y_true,y_pred))

# define one new instance
Xnew = [[35,0,1,24,0.68,173,172,46,114,530,1.19,8.87,2.47]]
# make a prediction
Ynew = model.predict(Xnew)
print("X=%s,\nPredicted=%s" % (Xnew[0], Ynew[0]))


# Creating the GradientBoosting Classifier model.
model = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_features=2, max_depth=2, random_state=0)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

print("GradientBoostingClassifier:", accuracy_score(y_true, y_pred))

confusionMatrix = confusion_matrix(y_true,y_pred)
print("-------------")
print(confusionMatrix)
print("-------------")
print(classification_report(y_true,y_pred))

# define one new instance
Xnew = [[35,0,1,24,0.68,173,172,46,114,530,1.19,8.87,2.47]]
# make a prediction
Ynew = model.predict(Xnew)
print("X=%s,\nPredicted=%s" % (Xnew[0], Ynew[0]))

