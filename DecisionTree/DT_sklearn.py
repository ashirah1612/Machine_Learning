import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
df=pd.read_csv("DecisionTree/stress_level_dataset.csv")
features = ["Hours_Studied", "Sleep_Hours", "Exercise_Hours", "Screen_Time", "Work_Hours", "Age"]

X=df[features]
y_class=df["Stress_Level"]
y_reg=df["Stress_Score"]

X_train,X_test,y_class_train,y_class_test=train_test_split(X,y_class,test_size=0.2,random_state=42,stratify=y_class)

X_train_reg,X_test_reg,y_reg_train,y_reg_test=train_test_split(X,y_reg,test_size=0.2,random_state=42)

classifier=DecisionTreeClassifier(random_state=42)
regressor=DecisionTreeRegressor(random_state=42)

classifier.fit(X_train,y_class_train)
regressor.fit(X_train_reg,y_reg_train)

class_prediction=classifier.predict(X_test)
reg_prediction=regressor.predict(X_test_reg)


print("Classification Accuracy:",accuracy_score(y_class_test,class_prediction))

print("Regression MAE:",mean_absolute_error(y_reg_test,reg_prediction))

print("Regression MSE:",mean_squared_error(y_reg_test,reg_prediction))

print("Regression RMSE:",mean_squared_error(y_reg_test,reg_prediction)**0.5)

new_person=pd.DataFrame([[2.5,5,0,10,7,26]],columns=features)

print("Predicted Stress Level:",classifier.predict(new_person)[0])

print("Predicted Stress Score:",regressor.predict(new_person)[0])