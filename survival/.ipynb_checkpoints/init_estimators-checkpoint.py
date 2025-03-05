import numpy as np
from functools import partial
from sklearn.feature_selection import SelectKBest, VarianceThreshold, RFE
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    QuantileTransformer,
    Normalizer,
    Binarizer,
    PowerTransformer,
)
import sksurv.metrics as sksurv_metrics
from sklearn.preprocessing import KBinsDiscretizer
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import (
    RandomSurvivalForest,
    ComponentwiseGradientBoostingSurvivalAnalysis,
    GradientBoostingSurvivalAnalysis,
)
from sksurv.svm import FastSurvivalSVM, FastKernelSurvivalSVM
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FastICA
from skopt.space import Real, Categorical, Integer
from osst.model.osst import OSST
from osst.model.metrics import harrell_c_index, uno_c_index, integrated_brier_score, cumulative_dynamic_auc, compute_ibs_per_sample

config = {
    "look_ahead": True,
    "diagnostics": True,
    "verbose": False,

    "regularization": 0.01,
    "uncertainty_tolerance": 0.0,
    "upperbound": 0.0,
    "depth_budget": 5,
    "minimum_captured_points": 7,

    "model_limit": 100
  }

def init_estimators(seed, n_workers, scalers, selectors, bucketizers, models, scoring):
    scalers_dict = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
        'QuantileTransformer': QuantileTransformer(),
        'Normalizer': Normalizer(),
        'Binarizer': Binarizer(),
        'PowerTransformer': PowerTransformer(),
    }
    scalers_dict = {scaler: scalers_dict[scaler] for scaler in scalers if scalers[scaler]}
    selectors_dict = {
        'SelectKBest': SelectKBest(partial(fit_and_score_features, scoring=None)),
        'VarianceThreshold': VarianceThreshold(),
        'FastICA': FastICA(max_iter=10000, random_state=seed),
        'PCA': PCA(n_components=0.9, random_state=seed),
        'KernelPCA': KernelPCA(random_state=seed),
        'TruncatedSVD': TruncatedSVD(random_state=seed),
        'RFE': RFE(CoxPHSurvivalAnalysis(n_iter=1000)),
    }
    selectors_dict = {selector: selectors_dict[selector] for selector in selectors if selectors[selector]}
    bucketizers_dict = {
        '2BinBucketizer': KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform'),
        '3BinBucketizer': KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform'),
        '4BinBucketizer': KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform'),
    }
    bucketizers_dict = {bucketizer: bucketizers_dict[bucketizer] for bucketizer in bucketizers if bucketizers[bucketizer]}

    models_dict = {
        'CoxPH': CoxPHSurvivalAnalysis(n_iter=1000),
        'OSST': OSST(config),
        'RSF': RandomSurvivalForest(random_state=seed, n_jobs=n_workers),

    }
    models_dict = {model: models_dict[model] for model in models if models[model]}
    

    return scalers_dict, selectors_dict, bucketizers_dict, models_dict


def fit_and_score_features(X, y, scoring):
    n_features = X.shape[1]
    scores = np.empty(n_features)
    model = CoxPHSurvivalAnalysis(alpha=0.1)
    if scoring is not None:
        estimator = getattr(sksurv_metrics, scoring)(model)  # attach scoring function
    else:
        estimator = model
    for feature in range(n_features):
        X_feature = X[:, feature : feature + 1]
        estimator.fit(X_feature, y)
        scores[feature] = estimator.score(X_feature, y)

    return scores


def set_params_search_space():
    
    model_params = {
        "OSST": {
            "regularization": [0.0001, 0.0005],
            "depth_budget": [6, 7, 8, 9] ,
            "minimum_captured_points": [5, 6, 7, 8]   
        }
    }

    selector_params = {
        "SelectKBest": {
            "selector__k": Integer(low=5, high=30)
        },
        "VarianceThreshold": {
            "selector__threshold": Real(low=0.01, high=0.2)
        },
        "PCA": {},
        "KernelPCA": {
            "selector__kernel": Categorical(["rbf", "poly"]),
            "selector__n_components": Integer(low=15, high=30)
        },
        "RFE": {
            "selector__n_features_to_select": Real(low=0.5, high=1),
        },
        "FastICA": {},
        "TruncatedSVD": {}
    }

    return selector_params, model_params
