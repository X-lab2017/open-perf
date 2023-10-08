from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from classification_models import ModelA, ModelB, ModelC

class ClassificationBenchmark:

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.metrics_result = None

    def run(self, train_features, train_labels, test_features, test_labels):
        # Train the model using training data
        self.model.train(train_features, train_labels)

        # Make predictions using test data
        predictions = self.model.predict(test_features)
        self.metrics_result = {  # 保存运行结果
            "accuracy": accuracy_score(test_labels, predictions),
            "f1_score": f1_score(test_labels, predictions, average='macro'),
            "precision": precision_score(test_labels, predictions, average='macro'),
            "recall": recall_score(test_labels, predictions, average='macro')
        }
        return self.metrics_result

    def get_metrics(self):
        if self.metrics_result is None:  # 如果结果为空，则运行模型
            self.run()
        return self.metrics_result
