from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

classes = ['malignant', 'benign']

# Support Vector Machine Model
clf1 = svm.SVC(kernel='linear', C=2)
clf1.fit(X_train, y_train)
predictions1 = clf1.predict(X_test)
accuracy1 = metrics.accuracy_score(y_test, predictions1)

# K Nearest Neighbours Model
clf2 = KNeighborsClassifier(n_neighbors=11)
clf2.fit(X_train, y_train)
prediction2 = clf2.predict(X_test)
accuracy2 = metrics.accuracy_score(y_test, prediction2)

# Accuracy comparison.
print(accuracy1)
print(accuracy2)