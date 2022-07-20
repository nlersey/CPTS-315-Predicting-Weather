import pandas
import numpy 
import seaborn 
import matplotlib.pyplot as matplot
import missingno 
from scipy import stats
from sklearn.metrics._plot.roc_curve import plot_roc_curve
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import resample

dataframe = pandas.read_csv("./RainData.csv")

#can delete
print("Data: (rows, columns)") #(145460, 23)
print(dataframe.shape)

#Matrix shows large amounts of missing data in: evaporation, sunshine, cloud9am, cloud3pm
missingno.matrix(dataframe)
#matplot.show()
#Shows coorelation between features
matplot.subplots()
corrMatrix = seaborn.heatmap(
    dataframe.corr(),
    annot=True,
    square=True,
    mask=numpy.tril(dataframe.corr()),
)
#matplot.show()

#Plot for Zscore
matplot.subplots()
seaborn.boxenplot(data=dataframe)
#matplot.show()

#Data cleaning
#Droping any duplicates
dataframe.drop_duplicates(inplace=True)
#Drop any missing values from RainTomorrow
dataframe.dropna(subset=["RainTomorrow"], inplace=True)
#Drop features evaporation, sunshine, cloud9am, cloud3pm due to missing data
dataframe.drop(["Evaporation", "Sunshine", "Cloud9am", "Cloud3pm"], axis=1, inplace=True)
#Drop features temp9am and pressure9am due to high coorelation that would cause Multicollinearity 
dataframe.drop(["Temp9am", "Pressure9am"], axis=1, inplace=True)
#Oversample Raintomorrow to balance out "no" and "yes" 
yes=dataframe[dataframe["RainTomorrow"]==1]
no=dataframe[dataframe["RainTomorrow"]==0]
upSamppleYes=resample(yes, replace=True, n_samples=len(no), random_state=123)
#removing any outliers that are more than 3 standard deviations away from the mean
z_Score = numpy.abs(stats.zscore(dataframe._get_numeric_data(), nan_policy="omit"))
dataframe = dataframe[(z_Score < 4).all(axis=1)]
#Change date categorical feature to numeric
dataframe["Date"] = pandas.to_datetime(dataframe["Date"])
dataframe["Date"] = dataframe["Date"].apply(lambda day: day.dayofyear)
#Change any remaining categoricals to numeric 
encoder = LabelEncoder()
for x in dataframe.columns[dataframe.dtypes == 'object']:
    dataframe[x].fillna(dataframe[x].mode()[0], inplace=True)
    dataframe[x] = encoder.fit_transform(dataframe[x])
#scale feature values 
scaled = MinMaxScaler().fit_transform(dataframe)
dataframe = pandas.DataFrame(
    scaled, columns=dataframe.columns
)  

#create label
label=dataframe["RainTomorrow"]
#drop RainTomorrow from features
features=dataframe.drop(["RainTomorrow"], axis=1)

labels = dataframe["RainTomorrow"]
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=123)
#decision tree classifer
dtclf=DecisionTreeClassifier(max_depth=64)
dtclf.fit(x_train, y_train)
y_pred=dtclf.predict(x_test)
print("Decision tree:", classification_report(y_test, y_pred))
#perceptron classifier
mpclf=MLPClassifier(hidden_layer_sizes=(32, 32), activation="relu", solver="adam", max_iter=128)
mpclf.fit(x_train, y_train)
y_pred=mpclf.predict(x_test)
print("MLP:", classification_report(y_test, mpclf.predict(x_test)))
#support vector classifier
