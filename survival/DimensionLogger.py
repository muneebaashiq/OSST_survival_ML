from sklearn.base import BaseEstimator, TransformerMixin

class DimensionLogger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
            # Print the shape of y if it is provided
            if y is not None:
                print("Target variable shape during fit:", y.shape)
            return self

    def transform(self, X, y=None):
        # Print the shape of y if it is provided
        if y is not None:
            print("Target variable shape during transform:", y.shape)
        print("Transforming data with shape:", X.shape)
        return X
        
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
