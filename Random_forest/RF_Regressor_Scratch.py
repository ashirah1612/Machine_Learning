from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

data=load_wine()
X=data.data
y=X[:,0]
X=np.delete(X,0,axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

rng=np.random.default_rng(42)

def variance(y):
    mean=np.mean(y)
    return np.mean((y-mean)**2)

def split_data(X,y,feature,threshold):
    left=X[:,feature]<=threshold
    right=X[:,feature]>threshold
    return X[left],X[right],y[left],y[right]

def variance_reduction(y,y_left,y_right):
    parent_variance=variance(y)
    n=len(y)
    weighted_variance=(len(y_left)/n)*variance(y_left)+(len(y_right)/n)*variance(y_right)
    return parent_variance-weighted_variance


def best_split(X,y,max_features):
    best_feature=None
    best_threshold=None
    best_reduction=-1
    feature_indices=rng.choice(X.shape[1],size=max_features,replace=False)
    for feature in feature_indices:
        thresholds=np.unique(X[:,feature])
        for threshold in thresholds:
            X_left,X_right,y_left,y_right=split_data(X,y,feature,threshold)
            if len(y_left)==0 or len(y_right)==0:
                continue
            reduction=variance_reduction(y,y_left,y_right)
            if reduction>best_reduction:
                best_reduction=reduction
                best_feature=feature
                best_threshold=threshold
    return best_feature,best_threshold

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

def build_tree(X,y,depth=0,max_depth=4,max_features=3):
    if len(y)==1:
        return Node(value=np.mean(y))
    if depth>=max_depth:
        return Node(value=np.mean(y))
    feature,threshold=best_split(X,y,max_features)
    if feature is None:
        return Node(value=np.mean(y))
    X_left,X_right,y_left,y_right=split_data(X,y,feature,threshold)
    left=build_tree(X_left,y_left,depth+1,max_depth,max_features)
    right=build_tree(X_right,y_right,depth+1,max_depth,max_features)
    return Node(feature,threshold,left,right)

def predict_one(node,x):
    if node.value is not None:
        return node.value
    if x[node.feature]<=node.threshold:
        return predict_one(node.left,x)
    else:
        return predict_one(node.right,x)

def bootstrap_sample(X,y):
    n=len(X)
    indices=rng.choice(n,size=n,replace=True)
    return X[indices],y[indices]

n_trees=5
max_depth=4
max_features=3

trees=[]

for i in range(n_trees):
    X_bootstrap,y_bootstrap=bootstrap_sample(X_train,y_train)
    tree=build_tree(X_bootstrap,y_bootstrap,max_depth=max_depth,max_features=max_features)
    trees.append(tree)

def random_forest_predict(trees,X):
    predictions=[]
    for x in X:
        tree_predictions=[predict_one(tree,x) for tree in trees]
        final_prediction=np.mean(tree_predictions)
        predictions.append(final_prediction)
    return predictions

rf_train_pred=random_forest_predict(trees,X_train)
rf_test_pred=random_forest_predict(trees,X_test)

mae=np.mean(np.abs(y_test-rf_test_pred))
mse=np.mean((y_test-rf_test_pred)**2)
rmse=np.sqrt(mse)

print("Random Forest Train MAE:",np.mean(np.abs(y_train-rf_train_pred)))
print("Random Forest Test MAE:",mae)
print("Random Forest Test MSE:",mse)
print("Random Forest Test RMSE:",rmse)

new_wine=np.array([1.77,2.1,17.0,97.0,2.65,2.4,0.28,2.0,5.2,1.12,3.18,1050])

tree_predictions=[predict_one(tree,new_wine) for tree in trees]
print()

for i,prediction in enumerate(tree_predictions):
    print("Tree",i+1,"Prediction:",prediction)

final_prediction=np.mean(tree_predictions)
print("Random Forest Predicted Alcohol:",final_prediction)