from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data = pd.read_csv(os.path.join(BASE_DIR,'data/car.data'))

label_encoder  = preprocessing.LabelEncoder()

for column in list(data.columns.values):
	data[column] = label_encoder.fit_transform(data[column])

y = data['class']
data = data.drop('class', axis=1)

# print(data.head())

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.1)

max_accuracy = 0
it = 0

for i in range(1, 40):
	model = KNeighborsClassifier(n_neighbors=i)
	model.fit(X_train, y_train)
	acc = model.score(X_test, y_test)
	if(acc > max_accuracy):
		max_accuracy = acc
		it = i

print(it, max_accuracy)