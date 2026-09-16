import numpy as np
import pandas as pd

df=pd.read_csv("Isolation_Forest/thyroid_dataset.csv")

X=df.drop(columns=["Outlier_label"]).select_dtypes(include=np.number).values.astype(float)
X=np.nan_to_num(X,nan=np.nanmedian(X,axis=0))

def c(n):
    if n<=1:
        return 0
    if n==2:
        return 1
    return 2*(np.log(n-1)+0.5772156649)-2*(n-1)/n

def isolation_tree(X,depth=0,max_depth=8):
    if len(X)<=1 or depth>=max_depth:
        return {"leaf":True,"size":len(X),"depth":depth}

    feature=np.random.randint(0,X.shape[1])
    min_value=np.min(X[:,feature])
    max_value=np.max(X[:,feature])

    if min_value==max_value:
        return {"leaf":True,"size":len(X),"depth":depth}

    split=np.random.uniform(min_value,max_value)
    left=X[X[:,feature]<split]
    right=X[X[:,feature]>=split]

    if len(left)==0 or len(right)==0:
        return {"leaf":True,"size":len(X),"depth":depth}

    return{ 
        "leaf":False,
        "feature":feature,
        "split":split,
        "left":isolation_tree(left,depth+1,max_depth),
        "right":isolation_tree(right,depth+1,max_depth)
    }

def path_length(point,tree):
    if tree["leaf"]:
        return tree["depth"]+c(tree["size"])
    if point[tree["feature"]]<tree["split"]:
        return path_length(point,tree["left"])
    else:
        return path_length(point,tree["right"])

def isolation_forest(X,n_trees=100,sample_size=256):
    trees=[]
    max_depth=int(np.ceil(np.log2(sample_size)))

    for _ in range(n_trees):
        size=min(sample_size,len(X))
        indices=np.random.choice(len(X),size=size,replace=False)
        sample=X[indices]
        tree=isolation_tree(sample,max_depth=max_depth)
        trees.append(tree)

    return trees

def anomaly_scores(X,trees,sample_size):
    scores=[]
    for point in X:
        path_lengths=[]

        for tree in trees:
            path_lengths.append(path_length(point,tree))

        average_path=np.mean(path_lengths)
        score=2**(-average_path/c(sample_size))
        scores.append(score)

    return np.array(scores)

np.random.seed(42)

print("Total Records:",len(X))
print("Total features",X.shape[1])

trees=isolation_forest(X,n_trees=100,sample_size=256)
scores=anomaly_scores(X,trees,256)

threshold=np.percentile(scores,95)

predictions=np.where(scores>=threshold,"Anomaly","Normal")

df["Anomaly Score"]=scores
df["Prediction"]=predictions

print("\nAnomaly Threshold:",round(threshold,4))
print("Detected Anomalies:",np.sum(predictions=="Anomaly"))
print("Detected Normal:",np.sum(predictions=="Normal"))

print("\nFirst 20 Predictions:")
print(df[["Anomaly Score","Prediction"]].head(20).to_string(index=False))

actual=df["Outlier_label"].str.lower().map({"o":"Anomaly","n":"Normal"})
comparison=pd.DataFrame({
    "Actual":actual,
    "Predicted":predictions
})

accuracy=np.mean(actual==predictions)

print("\nEvaluation")
print("Accuracy:",round(accuracy,4))

print("\nDetected Anomalies:")
print(
    df[
        df["Prediction"]=="Anomaly"
    ][
        ["Anomaly Score","Prediction","Outlier_label"]
    ].head(20).to_string(index=False)
)
