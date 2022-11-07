'''
#for numerical computing
import numpy as np

# for datarames
import pandas as pd


#Ignore warnings
import warnings
warnings.filterwarnings("ignore")


#to split train and test set
from sklearn.model_selection import train_test_split

#Machine learnings models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#to save the final model on disk

from sklearn.externals import joblib 

df=pd.read_csv('dataset_forest.csv')
print(df.shape)
print(df.columns)
print(df.head())
print(df.describe())
print(df.corr())

df = df.drop_duplicates()
print(df.shape)
print(df.isnull().sum())
df=df.dropna()
print(df.isnull().sum())


y = df.Fire_Occurence
#create separate object for input features

X = df.drop('Fire_Occurence', axis=1)

#split X and Y into train and test sets
X_train,  X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=0)

#print number of observation in X train, X test, Y_train and y_test

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model1 = LogisticRegression()
model2 = RandomForestClassifier(n_estimators=100)
model3 = KNeighborsClassifier(n_neighbors=3)
model4 = SVC()
model5 = GaussianNB()
model6 = DecisionTreeClassifier()

#training

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)

#predict test set results

y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)
y_pred6 = model6.predict(X_test)

acc1 = accuracy_score(y_test, y_pred1)
print("ACCURACY of Logistic Regression is",format(acc1*100))

acc2 = accuracy_score(y_test, y_pred2)
print("ACCURACY of RandomForestClassifier is",format(acc2*100))

acc3 = accuracy_score(y_test, y_pred3)
print("ACCURACY of KneighborsClassifier is",format(acc3*100))

acc4 = accuracy_score(y_test, y_pred4)
print("ACCURACY of SVC is",format(acc4*100))

acc5 = accuracy_score(y_test, y_pred5)
print("ACCURACY ofGaussianNB  is",format(acc5*100))

acc6 = accuracy_score(y_test, y_pred6)
print("ACCURACY of DecisionTreeClassifier is",format(acc6*100))

#from sklearn.externals import joblib
import joblib
#save the model as a pickle in a file

joblib.dump(model1, 'dataset_forest.pkl')

#load the model from the file

final_model = joblib.load('dataset_forest.pkl')

pred=final_model.predict(X_test)

acc= accuracy_score(y_test, pred)

print("ACCURACY OF FINAL MODEL IS",format(acc*100))
'''



import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("dataset_forest.csv")
data = np.array(data)

X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = log_reg.predict_proba(final)


pickle.dump(log_reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))







