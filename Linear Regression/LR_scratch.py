import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Linear Regression/study_scores.csv")


def loss_function(m, b, points):
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Score

        total_error += (y - (m * x + b)) ** 2

    return total_error/float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].Study_Time
        y = points.iloc[i].Score

        prediction = m_now * x + b_now
        error = y - prediction

        m_gradient += -(2 / n) * x * error
        b_gradient += -(2 / n) * error

    m = m_now - L * m_gradient
    b = b_now - L * b_gradient

    return m, b


m = 0
b = 0
L = 0.001
epochs = 50000


for i in range(epochs):

    m, b = gradient_descent(m, b, data, L)

    if i % 1000 == 0:
        print(
            f"Epoch: {i}, "
            f"m: {m:.4f}, "
            f"b: {b:.4f}, "
            f"loss: {loss_function(m, b, data):.4f}"
        )


print("Final values:")
print("m =", m)
print("b =", b)


plt.scatter(
    data.Study_Time,
    data.Score,
    color="black"
)

plt.plot(
    list(range(1, 11)),
    [m * x + b for x in range(1, 11)],
    color="red"
)

plt.xlabel("Study Time")
plt.ylabel("Score")
plt.title("Linear Regression")

plt.show()