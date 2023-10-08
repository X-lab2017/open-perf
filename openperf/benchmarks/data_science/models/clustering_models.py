from .base_model import BaseModel
from sklearn.cluster import KMeans, AgglomerativeClustering

class KMeansModel(BaseModel):
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters)

    def train(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.predict(data)

class AgglomerativeModel(BaseModel):
    def __init__(self, n_clusters=3):
        self.model = AgglomerativeClustering(n_clusters=n_clusters)

    def train(self, data):
        self.model.fit(data)

    def predict(self, data):
        return self.model.labels_
