import numpy as np
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()

X=data.data
y=data.target

print("Total Records:",X.shape[0])
print("Total Features:",X.shape[1])

mean=np.mean(X,axis=0)

X_centered=X-mean
std=np.std(X_centered,axis=0)

X_scaled=X_centered/std

covariance_matrix=(X_scaled.T @ X_scaled)/(X_scaled.shape[0]-1)
print("\nCovariance Matrix Shape:", covariance_matrix.shape)

eigenvalues,eigenvectors = np.linalg.eigh(covariance_matrix)

sorted_indices= np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

explained_variance=eigenvalues/np.sum(eigenvalues)

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

selected_vectors = eigenvectors[:, :n_components]
X_pca=X_scaled @ selected_vectors

print("\nOriginal Shape:", X.shape)
print("Reduced Shape:", X_pca.shape)

print("\nFirst 5 PCA Rows:")
print(X_pca[:5])