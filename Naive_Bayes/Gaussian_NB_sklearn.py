import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

data=load_breast_cancer()
X=data.data
y=data.target

print("Total Records:",X.shape[0])
print("Total Features:",X.shape[1])


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model=GaussianNB()

model.fit(X_train,y_train)
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

train_accuracy=accuracy_score(y_train,y_train_pred)
test_accuracy=accuracy_score(y_test,y_test_pred)

print("\nTrain Accuracy:",train_accuracy)
print("Test Accuracy:",test_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_test_pred))
