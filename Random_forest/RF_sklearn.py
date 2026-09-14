from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
import numpy as np

data=load_wine()

X=data.data
y_class=data.target

X_class_train,X_class_test,y_class_train,y_class_test=train_test_split(X,y_class,test_size=0.2,random_state=42,stratify=y_class)

classifier=RandomForestClassifier(n_estimators=5,max_depth=4,max_features=3,random_state=42)
classifier.fit(X_class_train,y_class_train)
class_prediction=classifier.predict(X_class_test)
print("Random Forest Classification Accuracy:",accuracy_score(y_class_test,class_prediction))

new_wine_class=np.array([[13.05,1.77,2.1,17.0,97.0,2.65,2.4,0.28,2.0,5.2,1.12,3.18,1050]])
class_result=classifier.predict(new_wine_class)
print("New wine predicted class:",class_result[0])

X_reg=np.delete(X,0,axis=1)
y_reg=X[:,0]


X_reg_train,X_reg_test,y_reg_train,y_reg_test=train_test_split(X_reg,y_reg,test_size=0.2,random_state=42)

regressor=RandomForestRegressor(n_estimators=5,max_depth=4,max_features=3,random_state=42)
regressor.fit(X_reg_train,y_reg_train)

reg_prediction=regressor.predict(X_reg_test)

mae=mean_absolute_error(y_reg_test,reg_prediction)
mse=mean_squared_error(y_reg_test,reg_prediction)
rmse=np.sqrt(mse)

print("Random Forest Regression MAE:",mae)
print("Random Forest Regression MSE:",mse)
print("Random Forest Regression RMSE:",rmse)

new_wine_reg=np.array([[1.77,2.1,17.0,97.0,2.65,2.4,0.28,2.0,5.2,1.12,3.18,1050]])

reg_result=regressor.predict(new_wine_reg)

print("New Wine Predicted Alcohol:",reg_result[0])