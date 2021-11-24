import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train_data = pd.read_csv("../input/titanic/train.csv")
test_data = pd.read_csv("../input/titanic/test.csv")

train_test_dataset = [train_data, test_data]

for dataset in train_test_dataset:
    dataset.drop(['PassengerId','Name'], axis=1, inplace=True)
    
for dataset in train_test_dataset:
    dataset.drop(['Ticket','Cabin'], axis=1, inplace=True)
 
for dataset in train_test_dataset:
    dataset.loc[(dataset['Sex']=='male'),'Sex'] = 0
    dataset.loc[(dataset['Sex']=='female'),'Sex'] = 1
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = pd.qcut(dataset['Age'],5).astype('category').cat.codes
    dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)
    dataset['Fare'] = pd.qcut(dataset['Fare'],5).astype('category').cat.codes
    dataset['Embarked'].fillna('S', inplace=True)
    dataset['Embarked'] = dataset['Embarked'].astype('category').cat.codes

for dataset in train_test_dataset:
    dataset.loc[(dataset['Parch']>=2),'Parch']=2
    
train_data = train_data.astype('object')
test_data = test_data.astype('object')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def train_and_test(model):
    model.fit(train_data, train_label)
    prediction = model.predict(test_data)
    accuracy = round(model.score(train_data, train_label) * 100, 2)
    print("Accuracy : ", accuracy, "%")
    return prediction
  
train_label = train_data['Survived']
train_data.drop(['Survived'],axis=1, inplace=True)

train_label = train_label.astype('int64')

train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)

# Logistic Regression
log_pred = train_and_test(LogisticRegression())
# SVM
svm_pred = train_and_test(SVC())
#kNN
knn_pred_4 = train_and_test(KNeighborsClassifier())
# Random Forest
rf_pred = train_and_test(RandomForestClassifier())
# Navie Bayes
nb_pred = train_and_test(GaussianNB())

# Random Forest
test_df = pd.read_csv("../input/titanic/test.csv")

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_data, train_label)
test_pred = random_forest.predict(test_data)
random_forest.score(train_data, train_label)

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],"Survived": test_pred})
submission.to_csv('submission.csv', index=False)
