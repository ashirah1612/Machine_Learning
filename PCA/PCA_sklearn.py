import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

data= load_breast_cancer()
X=data.data
y=data.target

print("Total Records:", X.shape[0])
print("Total Features:", X.shape[1])

scaler= StandardScaler()
X_scaled=scaler.fit_transform(X)

pca=PCA()
X_all=pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance=np.cumsum(explained_variance)

print("\nExplained Variance:")

for i, value in enumerate(explained_variance):
       print(f"PC{i + 1}: {value * 100:.2f}%")

print("\nCumulative Explained Variance:")

for i, value in enumerate(cumulative_variance):
    print(f"PC{i + 1}: {value * 100:.2f}%")

target_variance=0.95
n_components=np.argmax(cumulative_variance>=target_variance)+1

print("\nTarget Variance:", target_variance * 100, "%")
print("Components Required:", n_components)

pca=PCA(n_components=n_components)
X_pca=pca.fit_transform(X_scaled)

print("\nOriginal Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)

print("\nFirst 5 PCA Rows:")
print(X_pca[:5])
