# Machine Learning

Hands-on implementations of Machine Learning algorithms using Python, with a focus on understanding the concepts, mathematics, and implementation behind each algorithm.

This repository contains **from-scratch implementations, scikit-learn implementations, and experiments with different datasets**.

## Algorithms Covered

* Linear Regression
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Means Clustering
* Principal Component Analysis (PCA)
* Isolation Forest
* Apriori / Association Rules
* Naive Bayes and its variants

## Approach

I have been revisiting Machine Learning algorithms by going beyond simply using built-in model functions.

My general approach is:

**Concept → Mathematics → From-Scratch Implementation → Dataset → Evaluation → Scikit-learn**

Where an algorithm supports both classification and regression, I have created separate implementations for the respective tasks.

The number and type of files vary depending on the algorithm. Some folders contain separate classification and regression implementations, while others contain a single implementation or multiple algorithm variants.

For example, the repository includes separate classification and regression implementations for algorithms such as **Decision Tree, Random Forest, and SVM**.

Naive Bayes includes multiple variants such as:

* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Bernoulli Naive Bayes
* Categorical Naive Bayes
* Complement Naive Bayes

## Repository Structure

```text
Machine_Learning/
│
├── Apriori_AssociationRule/
├── DecisionTree/
├── Isolation_Forest/
├── KNN/
├── K_means_Clustering/
├── Linear_Regression/
├── Logistic_Regression/
├── Naive_Bayes/
├── PCA/
├── Random_forest/
├── SVM/
│
├── .gitignore
└── README.md
```

Each folder contains the implementations and experiments related to that algorithm.

## What I Focused On

* Understanding the concepts behind ML algorithms
* Working through the mathematics behind the algorithms
* Implementing algorithms from scratch
* Understanding classification and regression separately where applicable
* Working with datasets
* Training and testing models
* Evaluating model performance
* Implementing the same algorithms using scikit-learn
* Exploring different variants of algorithms

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Purpose

This repository is part of my ongoing Machine Learning revision.

Instead of only using functions such as `.fit()` and `.predict()`, I wanted to revisit the fundamentals, work through the mathematics, and understand what happens behind the implementation.

## Future Work

I plan to continue adding more algorithms, experiments, mathematical notes, and implementations as I continue revisiting Machine Learning.

---
