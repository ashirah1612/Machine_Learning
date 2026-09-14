import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error

df=pd.read_csv("SVM/titanic.csv")

features=["Pclass","Sex","Age","SibSp","Parch","Fare"]

X=df[features].copy()
X["Sex"]=X["Sex"].map({"male":0,"female":1})
X["Age"]=X["Age"].fillna(X["Age"].mean())
X["Fare"]=X["Fare"].fillna(X["Fare"].mean())

y_class=df["Survived"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y_class,test_size=0.2,random_state=42,stratify=y_class
)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

classifier=SVC(kernel="linear",C=1.0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

print("SVM Classification Accuracy:",accuracy_score(y_test,y_pred))

reg_features=["Pclass","Sex","Age","SibSp","Parch"]

X_reg=df[reg_features].copy()
X_reg["Sex"]=X_reg["Sex"].map({"male":0,"female":1})
X_reg["Age"]=X_reg["Age"].fillna(X_reg["Age"].mean())

y_reg=df["Fare"]
X_train_reg,X_test_reg,y_train_reg,y_test_reg=train_test_split(
    X_reg,y_reg,test_size=0.2,random_state=42
)

reg_scaler=StandardScaler()
X_train_reg=reg_scaler.fit_transform(X_train_reg)
X_test_reg=reg_scaler.transform(X_test_reg)

regressor=SVR(kernel="linear",C=1.0,epsilon=0.1)
regressor.fit(X_train_reg,y_train_reg)

y_pred_reg=regressor.predict(X_test_reg)
mae=mean_absolute_error(y_test_reg,y_pred_reg)
mse=mean_squared_error(y_test_reg,y_pred_reg)
rmse=mse**0.5

print("SVR MAE:",mae)
print("SVR MSE:",mse)
print("SVR RMSE:",rmse)

new_passenger_class=pd.DataFrame(
    [[3,0,22,1,0,7.25]],
    columns=features
)
new_passenger_class=scaler.transform(new_passenger_class)

class_result=classifier.predict(new_passenger_class)

if class_result[0]==1:
    print("New Passenger: Survived")
else:
    print("New Passenger: Did Not Survive")

new_passenger_reg=pd.DataFrame(
    [[3,0,22,1,0]],
    columns=reg_features
)

new_passenger_reg=reg_scaler.transform(new_passenger_reg)

fare_prediction=regressor.predict(new_passenger_reg)

print("Predicted Fare:",fare_prediction[0])