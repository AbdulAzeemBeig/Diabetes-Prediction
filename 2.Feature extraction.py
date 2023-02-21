####################Feature extraction starts
#This is the second step
# Feature Importance with Extra Trees Classifier
# random forest for feature importance on a classification problem
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
import pandas as pd
# load data
###############Remember this data you have to manually copy paste
###############been deleted during normalization
url = "final_dataset.csv"
dataframe = pd.read_csv(url)
array = dataframe.values
X = array[:, :-1]
Y = array[:, -1]
# feature extraction
model = RandomForestClassifier(n_estimators=10)
model.fit(X, Y)
print(model.feature_importances_)
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


