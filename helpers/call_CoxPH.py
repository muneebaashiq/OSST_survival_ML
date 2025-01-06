from sklearn.model_selection import StratifiedKFold

from osst.model.osst import OSST
from osst.model.metrics import harrell_c_index, uno_c_index, integrated_brier_score, cumulative_dynamic_auc, compute_ibs_per_sample

import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
kf = KFold(n_splits=2, shuffle=True)

def call_CoxPH_grid_search(configuration, X, y, event, scaler, bucketizer):
        preprocessor = Pipeline(
                [
                    #("scaler", scaler),
                    ("bucketizer", bucketizer)
                ]
        )


        y = pd.DataFrame(y, columns=['Time'])
        event = pd.DataFrame(event, columns=['Event'])
    
        #skf = StratifiedKFold(n_splits=2)
        skf = KFold(n_splits=2, shuffle=True)
        # Lists to store scores
        train_scores = []
        test_scores = []
    
        fold_number = 1
    
        print(f"The configuration is {configuration} \n")
    
        for train_index, test_index in skf.split(X, event):
            X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
            y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]
            event_train_fold, event_test_fold = event.iloc[train_index], event.iloc[test_index]
    
            print("For fold ", fold_number, " following is the result")

            # Instead of in-place modification, create a new object if you're going to transform or modify data
            X_train_fold_copy = X_train_fold.copy()  # Creating a new copy if required


            #Transforming X_train_fold
            preprocessor.fit(X_train_fold_copy)
            transformed_train = preprocessor.transform(X_train_fold_copy)
            selected_features = X_train_fold_copy.columns
            X_train_fold1 = pd.DataFrame(transformed_train, columns=selected_features)
            #print("X_train_fold1", X_train_fold)
            #print("y_train_fold", y_train_fold)
            #print("event_train_fold", event_train_fold)

                  
            print("X_train_fold shape:", X_train_fold1.shape)

            print(f"X_train_fold type: {type(X_train_fold1)}, shape: {X_train_fold1.shape}")

            t = pd.concat(
                [
                    X_train_fold1.reset_index(drop=True), 
                    y_train_fold.reset_index(drop=True), 
                    event_train_fold.reset_index(drop=True)
                ], 
                axis=1
            )            
            print(t)
                        # Handle missing values
            if t.isna().any().any():
                print("Handling missing values...")



 # Splitting columns again for model training
            X, y, event = df.iloc[:,:-2].values, df.iloc[:,-2].values.astype(int), df.iloc[:,-1].values
            h = df.columns[:-2]
            X = pd.DataFrame(X, columns=h)
            event = pd.DataFrame(event)
            y = pd.DataFrame(y)

            # Training OSST with the resample train data
            model = OSST(configuration)
            model.fit(X, event, y)

            # evaluation
            n_leaves = model.leaves()
            n_nodes = model.nodes()
            time = model.time
            #print("Model training time: {}".format(time))
            #print("# of leaves: {}".format(n_leaves))

            
