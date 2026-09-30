import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report

data = {
    "Review": [
        "phone battery camera",
        "phone screen battery",
        "laptop screen keyboard",
        "laptop battery keyboard",
        "phone camera screen",
        "tablet screen battery",
        "laptop keyboard battery",
        "phone battery charger",
        "tablet camera screen",
        "laptop screen battery",
        "shirt cotton size",
        "shirt cotton color",
        "jeans size denim",
        "dress cotton color",
        "jeans denim size",
        "dress size cotton",
        "sofa wood comfort",
        "table wood design"
    ],
    "Product": [
        "Electronics","Electronics","Electronics","Electronics","Electronics",
        "Electronics","Electronics","Electronics","Electronics","Electronics",
        "Clothing","Clothing","Clothing","Clothing","Clothing","Clothing",
        "Furniture","Furniture"
    ]
}

df = pd.DataFrame(data)

X=df["Review"]
y=df["Product"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=42,stratify=y)

vectorizer=CountVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model=ComplementNB(alpha=1)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("Training Accuracy:", model.score(X_train, y_train))
print("Testing Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

new_review = input("\nEnter a new review: ")

new_review_vector = vectorizer.transform([new_review])

prediction = model.predict(new_review_vector)

print("Prediction:", prediction[0])