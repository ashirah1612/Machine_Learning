import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df=pd.read_csv("DecisionTree/stress_level_dataset.csv")
features = ["Hours_Studied", "Sleep_Hours", "Exercise_Hours", "Screen_Time", "Work_Hours", "Age"]

X=df[features].values
y=df["Stress_Level"].values

def gini(y):
    classes,counts=np.unique(y,return_counts=True)
    probabilities=counts/len(y)
    return 1-np.sum(probabilities**2) 

def split_data(X,y,feature,threshold):
    left=X[:,feature]<=threshold
    right=X[:,feature]>threshold
    return X[left], X[right], y[left], y[right]

def gini_reduction(y,y_left,y_right):
    parent_gini=gini(y)
    n=len(y)
    weighted_gini=(len(y_left)/n)*gini(y_left)+(len(y_right)/n)*gini(y_right)
    return parent_gini-weighted_gini


def best_split(X,y):
    best_feature=None
    best_threshold=None
    best_reduction=-1
    for feature in range(X.shape[1]):
        thresholds=np.unique(X[:,feature])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split_data(X, y, feature, threshold)
            if len(y_left)==0 or len(y_right)==0:
                continue
            reduction=gini_reduction(y,y_left,y_right)

            if reduction>best_reduction:
                best_reduction=reduction
                best_feature=feature
                best_threshold=threshold

    return best_feature,best_threshold

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

def build_tree(X,y,depth=0,max_depth=4):
    if len(np.unique(y))==1:
        return Node(value=np.unique(y)[0])

    if depth>=max_depth:
        classes,counts=np.unique(y,return_counts=True)
        return Node(value=classes[np.argmax(counts)])

    feature, threshold = best_split(X, y)

    if feature is None:
        classes, counts = np.unique(y, return_counts=True)
        return Node(value=classes[np.argmax(counts)])

    X_left, X_right, y_left, y_right = split_data(X, y, feature, threshold)

    left = build_tree(X_left, y_left, depth + 1, max_depth)
    right = build_tree(X_right, y_right, depth + 1, max_depth)

    return Node(feature, threshold, left, right)

def predict_one(node,x):
    if node.value is not None:
        return node.value

    if x[node.feature]<=node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right,x)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

tree=build_tree(X_train,y_train)

new_person=[2.5,5,0,10,7,26]
prediction=predict_one(tree,new_person)
print("Predicted Stress_Level :",prediction)

y_pred_test = [predict_one(tree, x) for x in X_test]
accuracy = np.mean(y_pred_test == y_test)
print("Accuracy:", accuracy)