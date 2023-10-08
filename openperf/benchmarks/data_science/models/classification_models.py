from .base_model import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        self.model = LogisticRegression()

    def train(self, data, labels):
        self.model.fit(data, labels)

    def predict(self, data):
        return self.model.predict(data)

class RandomForestModel(BaseModel):
    def __init__(self):
        self.model = RandomForestClassifier()

    def train(self, data, labels):
        self.model.fit(data, labels)

    def predict(self, data):
        return self.model.predict(data)
