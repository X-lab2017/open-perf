from .base_model import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class LinearRegressionModel(BaseModel):
    def __init__(self):
        self.model = LinearRegression()

    def train(self, data, labels):
        self.model.fit(data, labels)

    def predict(self, data):
        return self.model.predict(data)

class RandomForestRegressionModel(BaseModel):
    def __init__(self):
        self.model = RandomForestRegressor()

    def train(self, data, labels):
        self.model.fit(data, labels)

    def predict(self, data):
        return self.model.predict(data)
