from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

data = fetch_20newsgroups(subset="all",
    categories=["sci.space","rec.sport.baseball","comp.graphics"]
)

X_text=data.data
y=data.target

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text,y,test_size=0.2,random_state=42,stratify=y)

vectorizer=CountVectorizer()

X_train=vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

model = MultinomialNB(alpha=1)

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=data.target_names
))

print("\nEnter a new text:")
new_text = input()

new_text_vector = vectorizer.transform([new_text])

prediction = model.predict(new_text_vector)

print("\nPrediction:", data.target_names[prediction[0]])