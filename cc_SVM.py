"""
Created on Thu Jan 25 18:40:43 2018
@author: swara
"""

import numpy as np
import pandas as pd

''' Import the data set'''
data = pd.read_csv("C:/Users/swara/Desktop/DS/kaggle/credit card/creditcard.csv")

print(data.info())
print(data.head())
'''
Data columns (total 31 columns):
Time      284807 non-null float64
V1        284807 non-null float64
V2        284807 non-null float64
V3        284807 non-null float64
V4        284807 non-null float64
V5        284807 non-null float64
V6        284807 non-null float64
V7        284807 non-null float64
V8        284807 non-null float64
V9        284807 non-null float64
V10       284807 non-null float64
V11       284807 non-null float64
V12       284807 non-null float64
V13       284807 non-null float64
V14       284807 non-null float64
V15       284807 non-null float64
V16       284807 non-null float64
V17       284807 non-null float64
V18       284807 non-null float64
V19       284807 non-null float64
V20       284807 non-null float64
V21       284807 non-null float64
V22       284807 non-null float64
V23       284807 non-null float64
V24       284807 non-null float64
V25       284807 non-null float64
V26       284807 non-null float64
V27       284807 non-null float64
V28       284807 non-null float64
Amount    284807 non-null float64
Class     284807 non-null int64
dtypes: float64(30), int64(1)
In this data set, time is provided. There are 28 features V1 to V28 obfuscated for confidentiality purposes.
The class variable denote if the transaction is normal or fraud, Class = 1 for fraud, 0 for normal.
Lets explore the data a bit more.
'''

print (data.Amount[data.Class == 1].describe())
'''Fraud Data
count     492.000000
mean      122.211321
std       256.683288
min         0.000000
25%         1.000000
50%         9.250000
75%       105.890000
max      2125.870000
Name: Amount, dtype: float64'''

print (data.Amount[data.Class == 0].describe())
'''Normal Data
count    284315.000000
mean         88.291022
std         250.105092
min           0.000000
25%           5.650000
50%          22.000000
75%          77.050000
max       25691.160000
Name: Amount, dtype: float64'''


''' In this dataset there are very less cases of fraud- 492 compared to normal cases which are 284315.
While training any model, we'll need to make sure to train it on sufficient number of both cases.
We will implement undersampling, that is using less samples from case 0 in order to train the model correctly.
'''

normal_class = data[data.Class == 0]
fraud_class = data[data.Class == 1]
''' Here we are using same amout of normal class data as fraud class data '''
undersampled_data = pd.concat([normal_class.sample(frac = (len(fraud_class)/len(normal_class))), fraud_class.sample(frac=1)],axis=0)
X = pd.DataFrame(undersampled_data.iloc[:,undersampled_data.columns!='Class'])
y = undersampled_data.iloc[:,undersampled_data.columns == 'Class']


''' We need to scale the data'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X["scaled_Amount"]=  sc.fit_transform(X.iloc[:,29].values.reshape(-1,1))

'''Dropping Time and Old amount'''
X= X.drop(["Time","Amount"], axis= 1)


'''Splitting in train and test set'''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

'''We will make use of Support vector machine classifier'''

from sklearn.svm import SVC
classifier= SVC(C= 10, kernel= 'rbf', random_state= 0)
classifier.fit(X_train, np.array(y_train).ravel())

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("The accuracy is "+str((cm[1,1]+cm[0,0])/(cm[0,0] + cm[0,1]+cm[1,0] + cm[1,1])*100) + " %")
print("The recall is "+ str(cm[1,1]/(cm[1,0] + cm[1,1])*100) +" %")
print("The precision is "+ str(cm[1,1]/(cm[0,1] + cm[1,1])*100) +" %")

'''For these parameters the results are as follows:
The accuracy is 91.3705583756 %
The recall is 85.8490566038 %
The precision is 97.8494623656 %
'''

'''Lets make use of gridsearch CV to tune the parameters'''

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
              {'C': [1, 10, 100], 'kernel': ['rbf']}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, np.array(y_train).ravel())
best_parameters = grid_search.best_params_

'''
best_parameters: {'C': 10, 'kernel': 'rbf'}'''


'''After running the model with these parameters here are the results:
The accuracy is 92.385786802 %
The recall is 92.4528301887 %
The precision is 93.3333333333 %
'''


'''We will now try the model on entire data set'''

data["scaled_Amount"]=  sc.fit_transform(data.iloc[:,29].values.reshape(-1,1))
data= data.drop(["Time","Amount"], axis= 1)

label = data.iloc[:,28]
data = data.drop(["Class"],axis=1)


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.25, random_state = 0)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("The accuracy is "+str((cm[1,1]+cm[0,0])/(cm[0,0] + cm[0,1]+cm[1,0] + cm[1,1])*100) + " %")
print("The recall is "+ str(cm[1,1]/(cm[1,0] + cm[1,1])*100) +" %")
print("The precision is "+ str(cm[1,1]/(cm[0,1] + cm[1,1])*100) +" %")

'''The accuracy is 92.7853150192 %
The recall is 95.8333333333 %
The precision is 2.19172860682 %'''

'''For the whole data set we get a high amount of false positives which leads to low precision.'''