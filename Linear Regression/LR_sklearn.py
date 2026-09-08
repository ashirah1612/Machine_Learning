import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv("linear Regression/study_scores.csv")

x=data[["Study_Time"]]
y=data["Score"]

model=LinearRegression()
model.fit(x,y)

m=model.coef_[0]
b=model.intercept_

print("Slope(m):",m)
print("Intercept(b):",b)

plt.scatter(data["Study_Time"], data["Score"], color="black")

plt.plot(
    data["Study_Time"],
    model.predict(x),
    color="red"
)

plt.xlabel("Study Time")
plt.ylabel("Score")
plt.title("Linear Regression using Scikit-learn")

plt.show()