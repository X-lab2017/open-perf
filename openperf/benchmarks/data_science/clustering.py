from sklearn.metrics import silhouette_score
class ClusteringBenchmark:

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.metrics_result = None

    def run(self, train_features):
        # Train the model using training data
        self.model.train(train_features)

        # Make predictions using test data
        cluster_labels = self.model.predict(train_features)
        self.metrics_result = {
            "silhouette_score": silhouette_score(train_features, cluster_labels)
        }
        return self.metrics_result

    def get_metrics(self):
        if self.metrics_result is None:
            self.run()
        return self.metrics_result
