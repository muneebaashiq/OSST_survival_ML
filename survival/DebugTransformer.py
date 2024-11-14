from sklearn.base import TransformerMixin, BaseEstimator

class DebugTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, step_name=""):
        self.step_name = step_name

    def fit(self, X, y=None):
        print(f"After step '{self.step_name}':")
        print("    X shape:", X.shape if hasattr(X, "shape") else type(X))
        if y is not None:
            print("    y shape:", y.shape if hasattr(y, "shape") else type(y))
        return self  # Fit returns itself to be compatible with pipeline

    def transform(self, X):
        print(f"Transforming at step '{self.step_name}':")
        print("    X shape:", X.shape if hasattr(X, "shape") else type(X))
        return X  # No modification to X, just for debugging
