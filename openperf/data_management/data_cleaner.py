class DataCleaner:

    def __init__(self, data):
        self.data = data

    def remove_missing_values(self):
        self.data = self.data.dropna()

    def remove_duplicates(self):
        self.data = self.data.drop_duplicates()

    def clean(self):
        self.remove_missing_values()
        self.remove_duplicates()
        return self.data
