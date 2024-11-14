from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Custom_Pipeline_Splitter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.X = X
        self.y = y
        return self

    def transform(self, X):
        if self.y is None:
            raise ValueError("The transformer has not been fitted yet.")
        
        # Convert y to a NumPy array and ensure it's 2D
        y = np.array([list(tup) for tup in self.y])
        if y.shape[1] != 2:
            raise ValueError("y should have exactly two columns: one for event and one for time.")
        
        # Separate event and time from y
        event = y[:, 0].astype(int)  # First column for event
        y = y[:, 1]  # Second column for time

        # Restructure output as separate arrays
        return X, y, event  # This format returns X, y (time), and event as needed
