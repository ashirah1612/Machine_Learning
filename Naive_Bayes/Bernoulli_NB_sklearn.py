import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data[:, [0, 1, 2, 3]]
y = data.target

feature_names = ["mean radius","mean texture","mean perimeter","mean area"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

thresholds = np.median(X_train, axis=0)

X_train_binary = (X_train >= thresholds).astype(int)
X_test_binary = (X_test >= thresholds).astype(int)

model = BernoulliNB(alpha=1)

model.fit(X_train_binary, y_train)

y_train_pred = model.predict(X_train_binary)
y_test_pred = model.predict(X_test_binary)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Total Records:", X.shape[0])
print("Total Features Used:", X.shape[1])

print("\nFeature Thresholds:")

for feature, threshold in zip(feature_names, thresholds):
    print(f"{feature}: {threshold:.2f}")

print("\nTraining Accuracy:", train_accuracy * 100, "%")
print("Testing Accuracy:", test_accuracy * 100, "%")

print("\nEnter New Patient Values:")

new_patient = []

for feature, threshold in zip(feature_names, thresholds):
    value = float(input(f"{feature}: "))

    if value >= threshold:
        new_patient.append(1)
    else:
        new_patient.append(0)

new_patient = np.array(new_patient).reshape(1, -1)

prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)

print("\nBinary Input:", new_patient)

print("\nPrediction:", data.target_names[prediction[0]])

print("\nPrediction Probabilities:")

for class_name, probability_value in zip(data.target_names, probability[0]):
    print(f"{class_name}: {probability_value * 100:.2f}%")