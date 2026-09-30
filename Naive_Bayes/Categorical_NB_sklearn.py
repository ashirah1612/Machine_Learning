import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

data=fetch_openml("titanic",version=1,as_frame=True)
df=data.frame
df = df[["pclass","sex","age","sibsp","parch","embarked","survived"]].dropna()

df["age"] = pd.cut(df["age"], [0,12,18,35,60,100],labels=["Child","Teen","Young","Adult","Senior"])
df["pclass"] = df["pclass"].astype(str)
df["sibsp"] = df["sibsp"].astype(str)
df["parch"] = df["parch"].astype(str)

X = df[["pclass","sex","age","sibsp","parch","embarked"]]
y=df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

encoder=OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)

X_train=encoder.fit_transform(X_train)
X_test=encoder.transform(X_test)

model=CategoricalNB(alpha=1)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("Training Accuracy:",model.score(X_train,y_train))
print("Testing Accuracy:",accuracy_score(y_test,y_pred))

print("\nClassification Report:")
print(classification_report(y_test,y_pred))

print("\nEnter passenger details:")

pclass = input("Passenger Class (1/2/3): ")
sex = input("Sex (male/female): ").lower()
age = input("Age (Child/Teen/Young/Adult/Senior): ").capitalize()
sibsp = input("Siblings/Spouses (0/1/2/3/4/5/8): ")
parch = input("Parents/Children (0/1/2/3/4/5/6): ")
embarked = input("Embarked (C/Q/S): ").capitalize()

new_passenger = pd.DataFrame([{
    "pclass":pclass,
    "sex":sex,
    "age":age,
    "sibsp":sibsp,
    "parch":parch,
    "embarked":embarked
}])

new_passenger=encoder.transform(new_passenger)
print(new_passenger)

prediction=model.predict(new_passenger)

if prediction[0] == 1:
    print("Prediction: Survived")
else:
    print("Prediction: Did not survive")