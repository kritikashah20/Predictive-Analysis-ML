# Basic Packages
import warnings
warnings.filterwarnings('ignore')
import pandas as pd  # to read the input
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# Read DataSheet
data = pd.read_csv('creditcard.csv')

# Data exploration
# print(data)  # all data will be printed
print(data.head())  # to display first five

''' Sampling ->  Under & Over Sampling -> reducing majority class into minority, increasing minority equal to majority
'''

# Sampling
sns.countplot('Class', data = data)  # check whether dataset is in balanced form or not
# class = 0 not a fraud user , 1 a fraud user

# Pre-Processing
from sklearn.preprocessing import StandardScaler
data['NormalizeAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data = data.drop(['Amount'], axis = 1)
print(data.head())

data = data.drop(['Time'], axis = 1)
print(data.head())

# Sampling
shuffled_df = data.sample(frac = 1, random_state = 4)

# Steps to do Under Sampling
fraud_df = shuffled_df.loc[shuffled_df['Class'] == 1]
not_fraud_df = shuffled_df.loc[shuffled_df['Class'] == 0].sample(n=490, random_state = 42)

sampling_data = pd.concat([fraud_df, not_fraud_df])
print(sampling_data.head())
print(sampling_data.shape)

sns.countplot('Class', data = sampling_data)

#Train and Test
x = data.iloc[:, data.columns!='Class']
y = data.iloc[:, data.columns=='Class']
print(x.head())
print(y.head())

from sklearn.model_selection import train_test_split
# x=data
# y=class(output)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 45)
print(xtrain.head())
print(ytrain.head())

# Algorithm
# SVM

from sklearn import svm
supportvm = svm.LinearSVC()
supportvm.fit(xtrain, ytrain)
y_pred_svm = supportvm.predict(xtest)

from sklearn.metrics import accuracy_score, confusion_matrix
model1 = accuracy_score(ytest, y_pred_svm)
print(model1)

# confusion matrix
cnf_matrix_svm = confusion_matrix(ytest, y_pred_svm)
plot_confusion_matrix(supportvm, xtest, ytest)
plt.show()

# Decision Tree

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(xtrain, ytrain)
y_pred_tree = tree.predict(xtest)
model2 = accuracy_score(ytest, y_pred_tree)
print(model2)
	
# confusion matrix
cnf_matrix_tree = confusion_matrix(ytest, y_pred_tree)
plot_confusion_matrix(tree, xtest, ytest)
plt.show()

# AdaBoost (Adative Boosting)
from sklearn.ensemble import AdaBoostClassifier
boost = AdaBoostClassifier(n_estimators = 100, base_estimator = tree, learning_rate = 1.0, random_state = 0) 
boost.fit(xtrain, ytrain)
y_pred_Boost = boost.predict(xtest)
model3 = accuracy_score(ytest, y_pred_Boost)
print(model3)

# confusion matrix
cnf_matrix_Boost = confusion_matrix(ytest, y_pred_Boost)
plot_confusion_matrix(boost, xtest, ytest)
plt.show()

import matplotlib.pyplot as plt
plt.rcdefaults()

objects = ('Support Vector', 'Decision Tree', 'AdaBoost')
y_pos = np.arange(len(objects))
performance = [model1, model2, model3]

plt.bar(y_pos, performance, align = 'center', alpha = 0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('SVM vs Decision Tree vs AdaBoost')

plt.show()






