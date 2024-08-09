import pandas as pd 
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import load_iris
import bentoml

iris= datasets.load_iris()
X,y=iris.data,iris.target

# X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=34)

model=svm.SVC()
model.fit(X,y)

saved_model=bentoml.sklearn.save_model("irismodel",model)

print(f"Model Saved as {saved_model}")





