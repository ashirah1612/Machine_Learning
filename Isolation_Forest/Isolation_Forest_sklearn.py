import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix


df=pd.read_csv("Isolation_Forest/thyroid_dataset.csv")

X=df.drop(columns=["Outlier_label"]).select_dtypes(include="number")
X=X.fillna(X.median())

model=IsolationForest(n_estimators=100,max_samples=256,contamination=0.05,random_state=42)
model.fit(X)
predictions=model.predict(X)

df["Prediction"]=predictions
df["Prediction"]=df["Prediction"].map({1:"Normal",-1:"Anomaly"})

actual=df["Outlier_label"].str.lower().map({"n":"Normal","o":"Anomaly"})

print("Total Records:",len(df))
print("Total Features",X.shape[1])

print("Detected Anomalies:",(df["Prediction"]=="Anomaly").sum())
print("Detected Normal:",(df["Prediction"]=="Normal").sum())

print("\nCustomer Predictions:")
print(df[["Prediction","Outlier_label"]].head(20).to_string(index=False))

accuracy=accuracy_score(actual,df["Prediction"])
precision=precision_score(actual,df["Prediction"],pos_label="Anomaly")
recall=recall_score(actual,df["Prediction"],pos_label="Anomaly")
f1=f1_score(actual,df["Prediction"],pos_label="Anomaly")

print("\nEvaluation")
print("Accuracy:",round(accuracy,3))
print("Precision:",round(precision,3))
print("Recall:",round(recall,3))
print("F1 Score:",round(f1,3))

print("\nConfusion Matrix:")
print(confusion_matrix(actual,df["Prediction"],labels=["Normal","Anomaly"]
))

print("\nDetected Anomalies:")
print(
    df[
        df["Prediction"]=="Anomaly"
    ][
        ["Prediction","Outlier_label"]
    ].head(20).to_string(index=False)
)