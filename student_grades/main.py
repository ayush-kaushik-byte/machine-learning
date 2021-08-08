from models.SKLearn import get_model
from models.gradient import GradientDescent
from utils.plot import Plotter
from utils.utility import read_data
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

X, y = read_data(os.path.join(DATA_DIR, 'student-mat.csv'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

linear_model = get_model(X_train, y_train)

predictions1 = linear_model.predict(X_test)

# print('MAE:', metrics.mean_absolute_error(y_test, predictions))
# print('MSE:', metrics.mean_squared_error(y_test, predictions))
# print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
# print('Accuracy:', linear_model.score(X_test, y_test))

gd = GradientDescent(X_train, y_train, X_test)

gd.fit(0.001, 100)

predictions2 = gd.predict()

plotter = Plotter()

plotter.scatterplot(predictions2,predictions1)