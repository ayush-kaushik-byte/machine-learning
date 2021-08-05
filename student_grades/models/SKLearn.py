from sklearn.linear_model import LinearRegression

def get_model(X, y):
	linear_model = LinearRegression()
	linear_model.fit(X, y)
	return linear_model