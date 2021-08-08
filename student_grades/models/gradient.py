import numpy as np


class GradientDescent:
	def __init__(self, X, y):

		self.y_shape = y.shape
		self.X = np.array(X)
		self.x_shape = self.X.shape
		self.X = np.insert(self.X, 0, np.ones((self.x_shape[0])), axis=1)
		self.x_shape = self.X.shape
		self.y = np.array(y).reshape(self.y_shape[0], 1)
		self.theta = np.zeros((self.x_shape[1], 1))
		self.m = self.x_shape[0]

	def cost_function(self, X, y, theta):

		J = np.sum(np.square((np.dot(X, theta) - y))) / (2 * self.x_shape[0])
		return J

	def fit(self, alpha, iterations, cost_map=False):

		J_hist =[]
		it = []
		if cost_map:
			for i in range(iterations):
				self.theta -= alpha * (np.dot(self.X.T, (np.dot(self.X, self.theta) - self.y))) / (self.x_shape[0])
				J_hist.append(self.cost_function(self.X, self.y, self.theta))

			it = [i for i in range(iterations)]

			return [J_hist, it]

		else:
			for i in range(iterations):
				self.theta -= alpha * (np.dot(self.X.T, (np.dot(self.X, self.theta) - self.y))) / (self.x_shape[0])

		return[J_hist, it]

	def predict(self, X_test):

		X_test = np.array(X_test)
		X_test = np.insert(X_test, 0, np.ones((X_test.shape[0])), axis=1)
		return np.dot(X_test, self.theta)

	def get_theta(self):
		return self.theta

	def get_shapes(self):
		return self.x_shape, self.y_shape, self.theta.shape
