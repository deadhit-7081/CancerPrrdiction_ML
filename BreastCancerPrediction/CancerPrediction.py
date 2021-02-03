import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")
print(df.head())
print(df.columns)

#removing unnamed column
print(df['Unnamed: 32'])
df = df.drop('Unnamed: 32',axis=1)
print(df.head())

#droping index column
df.drop('id', axis=1, inplace=True)
# df = df.drop('id', axis=1)
print(df.head())

l = list(df.columns)
print(l)

features_mean = l[1:11]
features_se = l[11:21]
features_worst = l[21:]

print(features_mean)
print(features_se)
print(features_worst)

print(df['diagnosis'].unique())
sns.countplot(df['diagnosis'], label="Count")
#plt.show()

print(df['diagnosis'].value_counts())

print(df.describe())

#correlation plot
corr = df.corr()
print(corr)

plt.figure(figsize=(8,8))
sns.heatmap(corr)
#plt.show()

df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
print(df.head())

#Creating question answer data set seperated
X = df.drop('diagnosis',axis = 1)
print(X.head())
y = df['diagnosis']
print(y.head())

#splitting the tranning and testing dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

#compressing the dataset around zero for better prediction
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train) #compressing training dataset around zero and learning dataset questions
X_test = ss.transform(X_test) #learning dataset answers
print(X_train)

'''Machine Learning'''
#Logistic Regression

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)

print(y_pred)

from sklearn.metrics import accuracy_score
lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
print(y_pred)

dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
print(y_pred)

rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)

#Support Vector Classifier
from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
print(y_pred)

svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)