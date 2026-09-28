import pandas as pd

data = pd.DataFrame({
    "free": [3, 2, 1, 4, 2, 0, 0, 1, 0, 0, 1, 0],
    "offer": [2, 1, 2, 1, 3, 0, 1, 0, 0, 0, 1, 0],
    "meeting": [0, 0, 0, 0, 0, 3, 2, 1, 3, 2, 1, 2],
    "urgent": [2, 1, 0, 2, 1, 0, 0, 1, 0, 0, 0, 1],
    "spam": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
})

print("Dataset:")
print(data)

features = ["free", "offer", "meeting", "urgent"]
target = "spam"

alpha = 1
total_records = len(data)

spam_data = data[data[target] == 1]
not_spam_data = data[data[target] == 0]

spam_count = len(spam_data)
not_spam_count = len(not_spam_data)

prior_spam = spam_count / total_records
prior_not_spam = not_spam_count / total_records

print("\nPrior Probabilities:")
print("P(Spam):", prior_spam)
print("P(Not Spam):", prior_not_spam)

spam_total_count = spam_data[features].sum().sum()
not_spam_total_count = not_spam_data[features].sum().sum()

vocabulary_size = len(features)

print("\nTotal Feature Counts:")
print("Spam:", spam_total_count)
print("Not Spam:", not_spam_total_count)
print("Number of Features:", vocabulary_size)

spam_denominator = spam_total_count + alpha * vocabulary_size
not_spam_denominator = not_spam_total_count + alpha * vocabulary_size

print("\nEnter New Email Word Counts:")

new_email = {}

for feature in features:
    new_email[feature] = int(input(f"{feature} count: "))

spam_score = prior_spam
not_spam_score = prior_not_spam

print("\nFeature Probabilities:")

for feature in features:

    spam_feature_count = spam_data[feature].sum()
    not_spam_feature_count = not_spam_data[feature].sum()

    spam_probability = (spam_feature_count + alpha) / spam_denominator

    not_spam_probability = (not_spam_feature_count + alpha) / not_spam_denominator

    print(f"\n{feature}:")
    print("Count in Spam:", spam_feature_count)
    print("Count in Not Spam:", not_spam_feature_count)
    print("P(feature | Spam):", spam_probability)
    print("P(feature | Not Spam):", not_spam_probability)
    print("New Email Count:", new_email[feature])

    spam_score *= spam_probability ** new_email[feature]
    not_spam_score *= not_spam_probability ** new_email[feature]

print("\nFinal Scores:")
print("Spam Score:", spam_score)
print("Not Spam Score:", not_spam_score)

if spam_score > not_spam_score:
    prediction = "Spam"
else:
    prediction = "Not Spam"

print("\nPrediction:", prediction)