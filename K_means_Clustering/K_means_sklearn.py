import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("K_means_clustering/Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

wcss_values=[]

for k in range(1,9):
    model=KMeans(n_clusters=k,random_state=42,n_init=10)
    model.fit(X_scaled)
    wcss_values.append(model.inertia_)
    print("K =", k, "WCSS =", round(model.inertia_, 2))

k=5
model=KMeans(n_clusters=k,random_state=42,n_init=10)
df["Cluster"]=model.fit_predict(X_scaled)

silhouette = silhouette_score(X_scaled, df["Cluster"])

print("\nSilhouette Score:", round(silhouette, 3))

print("\nCluster Centers:")

centers = scaler.inverse_transform(model.cluster_centers_)

for i in range(k):
    print("Cluster", i,
        ": Income =", round(centers[i][0], 2),
        "Spending =", round(centers[i][1], 2)
    )

overall_income = df["Annual Income (k$)"].mean()
overall_spending = df["Spending Score (1-100)"].mean()

cluster_names = {}

for i in range(k):

    income_mean = df.loc[
        df["Cluster"] == i,
        "Annual Income (k$)"
    ].mean()

    spending_mean = df.loc[
        df["Cluster"] == i,
        "Spending Score (1-100)"
    ].mean()

    if income_mean < overall_income and spending_mean < overall_spending:
        name = "Low Income - Low Spending"

    elif income_mean < overall_income and spending_mean >= overall_spending:
        name = "Low Income - High Spending"

    elif income_mean >= overall_income and spending_mean < overall_spending:
        name = "High Income - Low Spending"

    else:
        name = "High Income - High Spending"

    cluster_names[i] = name

df["Cluster Name"] = df["Cluster"].map(cluster_names)

print("\nCustomer Clusters:")
print(
    df[
        [
            "CustomerID",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Cluster",
            "Cluster Name"
        ]
    ].to_string(index=False)
)

new_customer = [[70, 70]]

new_customer_scaled = scaler.transform(new_customer)

new_cluster = model.predict(new_customer_scaled)[0]

print("\nNew Customer")
print("Annual Income:", 70)
print("Spending Score:", 70)
print("Predicted Cluster:", new_cluster)
print("Cluster Name:", cluster_names[new_cluster])

plt.plot(range(1, 9), wcss_values, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()