import pandas as pd

data = pd.DataFrame({
    "free": [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1],
    "offer": [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1],
    "meeting": [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
    "urgent": [1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
    "spam": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
})

print("dataset:")
print(data)

features = ["free", "offer", "meeting", "urgent"]
target = "spam"

alpha=1
total_records=len(data)

spam_data=data[data[target]==1]
not_spam_data=data[data[target]==0]

spam_count=len(spam_data)
not_spam_count=len(not_spam_data)

prior_spam=spam_count/total_records
prior_not_spam=not_spam_count/total_records

print("\nPrior Probabilities:")
print("P(Spam):", prior_spam)
print("P(Not Spam):", prior_not_spam)

print("\nEnter New mail values:")
new_mail={}

for feature in features:
    new_mail[feature]=int(input(f"{feature} (0 or 1): "))

spam_score=prior_spam
not_spam_score=prior_not_spam

print("\nFeature Probabilities:")

for feature in features:

    spam_ones=spam_data[feature].sum()
    not_spam_ones = not_spam_data[feature].sum()

    spam_probability_1=(spam_ones + alpha)/(spam_count + 2 * alpha)
    not_spam_probability_1 = (not_spam_ones + alpha) / (not_spam_count + 2 * alpha)

    spam_probability_0 = 1 - spam_probability_1
    not_spam_probability_0 = 1 - not_spam_probability_1

    if new_mail[feature] == 1:
        spam_probability = spam_probability_1
        not_spam_probability = not_spam_probability_1
    else:
        spam_probability = spam_probability_0
        not_spam_probability = not_spam_probability_0

    print(f"\n{feature}:")
    print("P(feature=1 | Spam):", spam_probability_1)
    print("P(feature=0 | Spam):", spam_probability_0)
    print("P(feature=1 | Not Spam):", not_spam_probability_1)
    print("P(feature=0 | Not Spam):", not_spam_probability_0)

    spam_score *= spam_probability
    not_spam_score *= not_spam_probability

print("\nFinal Scores:")
print("Spam Score:", spam_score)
print("Not Spam Score:", not_spam_score)

if spam_score > not_spam_score:
    prediction = "Spam"
else:
    prediction = "Not Spam"

print("\nPrediction:", prediction)