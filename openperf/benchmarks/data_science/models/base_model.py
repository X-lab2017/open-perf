class BaseModel:
    def train(self, data, labels):
        raise NotImplementedError("Train method not implemented in BaseModel!")

    def predict(self, data):
        raise NotImplementedError("Predict method not implemented in BaseModel!")
