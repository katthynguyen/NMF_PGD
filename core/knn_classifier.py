# core/knn_classifier.py
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class KNNClassifier:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        probabilities = self.model.predict_proba(X)
        predicted_labels = self.model.predict(X)
        return predicted_labels, probabilities
    
    def get_label_probabilities(self, probabilities):
        return dict(zip(self.model.classes_, probabilities))
    