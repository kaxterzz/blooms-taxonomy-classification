import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('../dataset/Blooms_Taxonomy_keywords.csv')

# new_np_array = OneHotEncoder().fit_transform(data.dropna())

new_np_array = pd.get_dummies(data)

print(new_np_array)

# new_np_array.reshape(-1,1)

X = new_np_array['remember_Choose']
y = new_np_array['remember_Choose']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
