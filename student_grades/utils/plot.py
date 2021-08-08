import matplotlib.pyplot as plt
import seaborn as sns

class Plotter():
	
	def __init__(self):
		sns.set_palette("GnBu_d")
		sns.set_style('whitegrid')

	def jointplot(self, x_param, y_param, data):
		sns.jointplot(x=x_param, y=y_param, data=data)
		plt.show()

	def scatterplot(self, x_param, y_param):
		plt.scatter(x_param,y_param)
		plt.xlabel('X AXIS')
		plt.ylabel('Y AXIS')
		plt.show()

	def histplot(self, y_test, predictions):
		sns.histplot((y_test-predictions),bins=50)
		plt.show()

	def pairplot(self, data):
		sns.pairplot(data)
		plt.show()

