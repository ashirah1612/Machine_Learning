import numpy as np

X=np.array([
    [2,60],
    [3,65],
    [4,70],
    [7,80],
    [8,85],
    [9,90]
])

y=np.array(["Fail","Fail","Fail","Pass","Pass","Pass"])

study_hours=float(input("Enter Study Hours:"))
attendance=float(input("Enter Attendance:"))

new_student=np.array([study_hours,attendance])

classes=np.unique(y)

priors={}
means={}
variances={}

for c in classes:
    X_class=X[y==c]
    priors[c]=len(X_class)/len(X)
    means[c]=np.mean(X_class,axis=0)
    variances[c]=np.var(X_class,axis=0)

def gaussian_probability(x,mean,variance):
    return (1/np.sqrt(2*np.pi*variance))*np.exp(-(x-mean)**2)/(2*variance)

scores={}

for c in classes:
    score=priors[c]
    for i in range(X.shape[1]):
        probability=gaussian_probability(new_student[i],means[c][i],variances[c][i])
        score*=probability
    scores[c]=score

prediction=max(scores,key=scores.get)

print("\nPrediction:",prediction)
print("Fail Score:",scores["Fail"])
print("Pass Score:",scores["Pass"])