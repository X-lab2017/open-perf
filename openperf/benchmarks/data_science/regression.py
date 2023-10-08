from sklearn.metrics import mean_squared_error
from regression_models import LinearRegressionModel, RandomForestRegressionModel

class RegressionBenchmark:

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.metrics_result = None

    def run(self, train_features, train_labels, test_features, test_labels):
        # Train the model using training data
        self.model.train(train_features, train_labels)

        # Make predictions using test data
        predictions = self.model.predict(test_features)
        self.metrics_result = {
            "mse": mean_squared_error(test_labels, predictions)
        }
        return self.metrics_result

    def get_metrics(self):
        if self.metrics_result is None:
            self.run()
        return self.metrics_result
