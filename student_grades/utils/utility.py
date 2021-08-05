import pandas as pd

def read_data(file_name):
	data = pd.read_csv(file_name, sep=';')
	X = data[['traveltime', 'studytime', 'failures', 'freetime',
					'goout', 'Dalc', 'Walc', 'health', 'absences',
					'G1', 'G2']]
	y = data[['G3']]
	return X, y


# def transform_binary(data)

if __name__ == '__main__':
	X, y = read_data('student-mat.csv')
	print(X.head())
	print(X.info())
	print(X.describe())