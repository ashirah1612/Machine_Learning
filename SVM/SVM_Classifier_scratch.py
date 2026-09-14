import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv("SVM/titanic.csv")
features=["Pclass","Sex","Age","SibSp","Parch","Fare"]
X=df[features].copy()

X["Sex"]=X["Sex"].map({"male":0,"female":1})
X["Age"]=X["Age"].fillna(X["Age"].mean())
X["Fare"]=X["Fare"].fillna(X["Fare"].mean())

X=X.values.astype(float)
y=df["Survived"].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)

X_train=(X_train-mean)/std
X_test=(X_test-mean)/std

y_train_svm=np.where(y_train==1,1,-1)

def train_svm(X,y,C=1.0,learning_rate=0.001,epochs=10000):
    w=np.zeros(X.shape[1])
    b=0
    n=len(y)

    for epoch in range(epochs):
        margins=y*(X@w+b)
        misclassified=margins<1

        if np.any(misclassified):
            dw=w-C*np.mean(y[misclassified,None]*X[misclassified],axis=0)
            db=-C*np.mean(y[misclassified])
        else:
            dw=w
            db=0

        w=w-learning_rate*dw
        b=b-learning_rate*db

    return w,b

w,b=train_svm(X_train,y_train_svm)

def predict(X,w,b):
    scores=X@w+b
    return np.where(scores>=0,1,0)

y_pred=predict(X_test,w,b)

accuracy=np.mean(y_pred==y_test)

print("Accuracy:",accuracy)

new_passenger=np.array([[3,0,22,1,0,7.25]])

new_passenger=(new_passenger-mean)/std

prediction=predict(new_passenger,w,b)

if prediction[0]==1:
    print("New Passenger: Survived")
else:
    print("New Passenger: Did Not Survive")

        