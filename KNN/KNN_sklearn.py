from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

X,y=load_iris(return_X_y=True)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=20)

scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train_scaled,y_train)


y_pred_train=model.predict(X_train_scaled)

y_pred_test=model.predict(X_test_scaled)

train_acc=accuracy_score(y_train,y_pred_train)
test_acc=accuracy_score(y_test,y_pred_test)

print("Training Accuracy:",train_acc)
print("testing Accuracy:",test_acc)

plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train)
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, marker="*")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Classification")
plt.show()