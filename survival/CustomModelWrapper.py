from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

# Custom model wrapper
class CustomModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, y):
        self.model = model
        self.y = y
        
    
    def fit(self, X, event):
        # Ensure X is a DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)  # reconstruct columns
        # Pass X, event, and y to the model’s fit method
        self.model.fit(X, event, self.y)
        return self
    
    def predict(self, X):
        # Check if X is a numpy ndarray and convert to pandas DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        return self.model.predict(X)

    def score(self, X, event):
        # Check if X is a numpy ndarray and convert to pandas DataFrame if necessary
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        return self.model.score(X, event, self.y)