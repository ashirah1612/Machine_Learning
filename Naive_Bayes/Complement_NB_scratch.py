import pandas as pd
import numpy as np

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

features = ["phone","battery","camera","screen","keyboard",
            "tablet","charger","shirt","cotton","size",
            "color","jeans","denim","dress","sofa","wood",
            "comfort","table","design"]

for feature in features:
    df[feature]=df["Review"].apply(lambda x: x.split().count(feature))

classes=df["Product"].unique()
alpha=1

complement_probabilities={}

for current_class in classes:
    complement_data=df[df["Product"]!=current_class]

    total_counts=complement_data[features].sum()
    total_words=total_counts.sum()
    vocabulary_size=len(features)

    probabilities={}

    for feature in features:
        probabilities[feature] = (
            total_counts[feature] + alpha
        ) / (
            total_words + alpha * vocabulary_size
        )

    complement_probabilities[current_class] = probabilities

new_review="phone battery camera"

new_counts={}

for feature in features:
    new_counts[feature]=new_review.split().count(feature)

scores = {}

for current_class in classes:
    score = 0

    for feature in features:
        if new_counts[feature] > 0:
            probability = complement_probabilities[current_class][feature]
            score += new_counts[feature] * np.log(probability)

    scores[current_class] = score

prediction = min(scores, key=scores.get)

print("\nComplement Scores:")

for current_class in classes:
    print(current_class, scores[current_class])

prediction = min(scores, key=scores.get)

print("\nPrediction:", prediction)