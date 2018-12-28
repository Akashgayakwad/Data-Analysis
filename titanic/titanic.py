# Conversion to dataframe
df = pd.DataFrame(X_train)
df2 = pd.DataFrame(X_test)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Importing the dataset
#for training set
train = pd.read_csv('train.csv')
train = train.dropna(subset=['Embarked']) #removing rows with Embarked as nan
#for test set
test = pd.read_csv('test.csv')
result = pd.read_csv('gender_submission.csv')
test = test.join(result['Survived'])
test = test.dropna(subset=['Embarked'])  #removing rows with Embarked as nan

#shuffling the data for unbaised prediction
#for training set
train = shuffle(train)
#for test set
test = shuffle(test)

#selecting the desired variables
#for training set
X_train = train.iloc[:, [2, 4, 5, 6, 7, 9, 10, 11]].values
y_train = train.iloc[:, 1].values
#for test set
X_test = test.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10]].values
y_test = test.iloc[:, 11].values

#handling missing value of age and fare
#for training set
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X_train[:, [2, 5]])
X_train[:, [2, 5]] = imputer.transform(X_train[:, [2, 5]])
#for test set
imputer = imputer.fit(X_test[:, [2, 5]])
X_test[:, [2, 5]] = imputer.transform(X_test[:, [2, 5]])

#updating cabin column by number of cabins allotted
#for training set
for i in range(0, X_train.shape[0], 1):
    if(str(X_train[i, 6]) == "nan"):
        count = 0
    else:
        cabins = X_train[i, 6].split()
        count = len(cabins)
    X_train[i, 6] = count
#for test set
for i in range(0, X_test.shape[0], 1):
    if(str(X_test[i, 6]) == "nan"):
        count = 0
    else:
        cabins = X_test[i, 6].split()
        count = len(cabins)
    X_test[i, 6] = count

# Encoding categorical data
#for training set
labelencoder_X_train_1 = LabelEncoder()
labelencoder_X_train_2 = LabelEncoder()
X_train[:, 1] = labelencoder_X_train_1.fit_transform(X_train[:, 1])
X_train[:, 7] = labelencoder_X_train_2.fit_transform(X_train[:, 7])
#for test set
labelencoder_X_test_1 = LabelEncoder()
labelencoder_X_test_2 = LabelEncoder()
X_test[:, 1] = labelencoder_X_test_1.fit_transform(X_test[:, 1])
X_test[:, 7] = labelencoder_X_test_2.fit_transform(X_test[:, 7])

#OneHotEncoding
#for training set
onehotencoder1 = OneHotEncoder(categorical_features = [1, 7])
X_train = onehotencoder1.fit_transform(X_train).toarray()
#for test set
onehotencoder2 = OneHotEncoder(categorical_features = [1, 7])
X_test = onehotencoder2.fit_transform(X_test).toarray()

#avoiding dummy variable trap
#for training set(removing 0th  and 2nd column)
X_train = X_train[:, [1, 3, 4, 5, 6, 7, 8, 9, 10]] 
#for test set(removing 0th  and 2nd column)
X_test = X_test[:,[1, 3, 4, 5, 6, 7, 8, 9, 10]] 

#Applying Feature Scaling
#for training set
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#for test set
X_test = sc_X.transform(X_test)

'''DATA PREPROCESSING COMPLETED DATA PREPROCESSING COMPLETED DATA PREPROCESSING COMPLETED
   DATA PREPROCESSING COMPLETED DATA PREPROCESSING COMPLETED DATA PREPROCESSING COMPLETED
   DATA PREPROCESSING COMPLETED DATA PREPROCESSING COMPLETED DATA PREPROCESSING COMPLETED
'''

'''LOGISTIC REGRESSION CLASSIFIER'''
# Fitting Logistic Regression to the Training set
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Accuracy =  {:0.2f}% for logistic Regression Classifier".format(classifier.score(X_test,y_test)*100))

'''K-NEAREST NEIGHOUR CLASSIFIER'''
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy =  {:0.2f}% for K-NN Classifier".format(classifier.score(X_test,y_test)*100))

'''SUPPORT VECTOR MACHINE CLASSIFIER'''
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy =  {:0.2f}% for SVM Classifier".format(classifier.score(X_test,y_test)*100))

'''KERNAL SVM CLASSIFIER'''
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy =  {:0.2f}% for Kernal SVM Classifier".format(classifier.score(X_test,y_test)*100))

'''NAIVE BAYES CLASSIFIER'''
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy =  {:0.2f}% for Naive Bayes Classifier".format(classifier.score(X_test,y_test)*100))

'''DECISION TREE CLASSIFIER'''
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy =  {:0.2f}% for Decision Tree Classifier".format(classifier.score(X_test,y_test)*100))

'''RANDOM FOREST CLASSIFIER'''
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy =  {:0.2f}% for Random Forest Classifier".format(classifier.score(X_test,y_test)*100))
