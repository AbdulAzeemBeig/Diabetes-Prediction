# Program to visualize missing values in dataset
# This is the first step

# Importing the libraries
import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt

# Loading the dataset
df = pd.read_csv('diabetes.csv')
#print first 3 rows to get idea of dataset
print(df.head(3))
# Visualize the number of missing
# values as a bar chart
msno.bar(df)
plt.show()



# example of imputing missing values using scikit-learn
from numpy import nan
from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer
# load the dataset
dataset = read_csv('diabetes.csv')
print(dataset.head())
# count the number of missing values for each column
dataset.replace(r'^\s*$', nan, regex=True)
num_missing = dataset.isnull().sum()
# report the results
print(num_missing)
# retrieve the numpy array
values = dataset.values
# define the imputer
imputer = SimpleImputer(missing_values=nan, strategy='mean')
# transform the dataset
transformed_values = imputer.fit_transform(values)
# count the number of NaN values in each column
print('Missing: %d' % isnan(transformed_values).sum())
print(transformed_values)

#save the new dataset with no missing values
import pandas as pd
pd.DataFrame(transformed_values).to_csv("no_missing_values.csv",index=None)

# plot missing values in new dataset
df = pd.read_csv('no_missing_values.csv')
#print first 3 rows to get idea of dataset
print(df.head(3))
# Visualize the number of missing
# values as a bar chart
msno.bar(df)
plt.show()


#Normalize the data
# visualize a minmax scaler transform
# evaluate knn
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
# load dataset
dataset = read_csv("no_missing_values.csv")
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# define the pipeline
trans = MinMaxScaler()
model = KNeighborsClassifier()
pipeline = Pipeline(steps=[('t', trans), ('m', model)])
# evaluate the pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report pipeline performance
print("For Normalization :")
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#standardize the data
# evaluate knn on the sonar dataset with standard scaler transform
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
# load dataset
dataset = read_csv("no_missing_values.csv")
data = dataset.values
# separate into input and output columns
X, y = data[:, :-1], data[:, -1]
# ensure inputs are floats and output is an integer label
X = X.astype('float32')
y = LabelEncoder().fit_transform(y.astype('str'))
# define the pipeline
trans = StandardScaler()
model = KNeighborsClassifier()
pipeline = Pipeline(steps=[('t', trans), ('m', model)])
# evaluate the pipeline
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report pipeline performance
print("For Standardization :")
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#now we normalize our data and save in separate dataset
# visualize a minmax scaler transform of the sonar dataset
from pandas import read_csv
from pandas import DataFrame
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
# load dataset
url = "no_missing_values.csv"
dataset = read_csv(url)
# retrieve just the numeric input values
data = dataset.values[:, :-1]
# perform a robust scaler transform of the dataset
trans = MinMaxScaler()
data = trans.fit_transform(data)
# convert the array back to a dataframe
dataset = DataFrame(data)
# summarize
print(dataset.describe())
# histograms of the variables
#dataset.hist()
#pyplot.show()

#save the new dataset
import pandas as pd
pd.DataFrame(data).to_csv("normalized_data.csv",index=None)

