import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, data_path, test_size=0.3, label_column_name='label'):
        self.data_path = data_path
        self.test_size = test_size
        self.label_column_name = label_column_name
        self.data = None
        self.train_data, self.test_data, self.train_labels, self.test_labels = None, None, None, None

    def load_data(self):
        # Using pandas to load CSV file
        df = pd.read_csv(self.data_path)

        labels = df[self.label_column_name].values
        data = df.drop(columns=[self.label_column_name]).values

        # Split the data
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            data, labels, test_size=self.test_size, random_state=42
        )

    def get_train_data(self):
        if self.train_data is None:
            self.load_data()
        return self.train_data, self.train_labels

    def get_test_data(self):
        if self.test_data is None:
            self.load_data()
        return self.test_data, self.test_labels

    def get_data_splits(self):
        if not self.train_data or not self.test_data:
            self.load_data()
        return self.train_data, self.train_labels, self.test_data, self.test_labels

    def load_data(self):
        # Using pandas to load CSV file
        df = pd.read_csv(self.data_path)
        self.data = df.to_dict(orient='records')

    def get_project_list(self):
        if self.data is None:
            self.load_data()
        return [item['project_name'] for item in self.data]
