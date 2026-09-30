import pandas as pd

data = {
    "Age": ["Young","Young","Adult","Senior","Senior","Adult","Young","Senior"],
    "Education": ["Graduate","School","Graduate","Graduate","School","School","Graduate","Graduate"],
    "City": ["Chennai","Madurai","Chennai","Delhi","Madurai","Delhi","Madurai","Chennai"],
    "Buy": ["Yes","Yes","Yes","No","No","No","Yes","No"]
}

df = pd.DataFrame(data)

features = ["Age","Education","City"]
alpha=1

classes=df["Buy"].unique()
probabilities={}

for current_class in classes:
    class_data=df[df["Buy"]==current_class]
    class_count=len(class_data)
    probabilities[current_class]={}

    for feature in features:
        categories=df[feature].unique()
        category_prob={}

        for category in categories:
            count=len(class_data[class_data[feature]==category])
            category_count=len(categories)
            probability=(count+alpha)/(class_count+alpha*category_count)
            category_prob[category]=probability

        probabilities[current_class][feature]=category_prob

print("Probabilities:")

for current_class in classes:
    print("\nClass:", current_class)

    for feature in features:
        print(feature, probabilities[current_class][feature])


print("\nEnter new customer details:")

age = input("Age (Young/Adult/Senior): ").capitalize()
education = input("Education (School/Graduate): ").capitalize()
city = input("City (Chennai/Madurai/Delhi): ").capitalize()

new_input = {
    "Age": age,
    "Education": education,
    "City": city
}

scores={}

for current_class in classes:
    prior = len(df[df["Buy"] == current_class]) / len(df)
    score = prior
    for feature in features:
        category = new_input[feature]
        probability = probabilities[current_class][feature][category]
        score = score * probability
    scores[current_class] = score


for current_class in classes:
    print(current_class, scores[current_class])

prediction = max(scores, key=scores.get)

print("\nPrediction:", prediction)
