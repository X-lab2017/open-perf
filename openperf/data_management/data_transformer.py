from sklearn.preprocessing import StandardScaler

class DataTransformer:

    def __init__(self, data):
        self.data = data

    def standardize(self, columns):
        scaler = StandardScaler()
        self.data[columns] = scaler.fit_transform(self.data[columns])

    def transform(self):
        numeric_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.standardize(numeric_columns)
        return self.data
