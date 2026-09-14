import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df=pd.read_csv("SVM/titanic.csv")
features=["Pclass","Sex","Age","SibSp","Parch"]
X=df[features].copy()

X["Sex"]=X["Sex"].map({"male":0,"female":1})
X["Age"]=X["Age"].fillna(X["Age"].mean())

X=X.values.astype(float)
y=df["Fare"].values

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

mean=np.mean(X_train,axis=0)
std=np.std(X_train,axis=0)

X_train=(X_train-mean)/std
X_test=(X_test-mean)/std

def train_svr(X,y,C=1.0,epsilon=0.1,learning_rate=0.001,epochs=10000):
    w=np.zeros(X.shape[1])
    b=0
    for epoch in range(epochs):
        predictions=X@w+b
        errors=y-predictions

        outside=np.abs(errors)>epsilon

        if np.any(outside):
            dw=w-C*np.mean(np.sign(errors[outside])[:,None]*X[outside],axis=0)
            db=-C*np.mean(np.sign(errors[outside]))
        else:
            dw=w
            db=0

        w=w-learning_rate*dw
        b=b-learning_rate*db

    return w,b

w,b=train_svr(X_train,y_train)

def predict(X,w,b):
    return X@w+b
y_pred=predict(X_test,w,b)

mae=np.mean(np.abs(y_test-y_pred))
mse=np.mean((y_test-y_pred)**2)
rmse=np.sqrt(mse)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

new_passenger=np.array([[3,0,22,1,0]])

new_passenger=(new_passenger-mean)/std

prediction=predict(new_passenger,w,b)

print("Predicted Fare:",prediction[0])