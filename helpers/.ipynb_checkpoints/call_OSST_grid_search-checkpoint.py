from sklearn.model_selection import StratifiedKFold

from osst.model.osst import OSST
from osst.model.metrics import harrell_c_index, uno_c_index, integrated_brier_score, cumulative_dynamic_auc, compute_ibs_per_sample

import numpy as np
import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

def call_OSST_grid_search(configuration, X_input, y_input, event_input, scaler, bucketizer):
        preprocessor = Pipeline(
                [
                    ("scaler", scaler),
                    ("bucketizer", bucketizer)
                ]
        )


        y_input = pd.DataFrame(y_input, columns=['Time'])
        event_input = pd.DataFrame(event_input, columns=['Event'])
    
        skf = StratifiedKFold(n_splits=4)
        # Lists to store scores
        train_scores = []
        test_scores = []
        times = []
        leaves = []
        nodes = []
    
        fold_number = 1
    
        #print(f"The configuration is {configuration} \n")
    
        for train_index, test_index in skf.split(X_input, event_input):
            X_train, X_test = X_input.iloc[train_index], X_input.iloc[test_index]
            y_train, y_test = y_input.iloc[train_index], y_input.iloc[test_index]
            event_train, event_test = event_input.iloc[train_index], event_input.iloc[test_index]
    
            #print("For fold ", fold_number, " following is the result")

            # Instead of in-place modification, create a new object if you're going to transform or modify data
            X_train_copy = X_train.copy()  # Creating a new copy if required


            #Transforming X_train
            preprocessor.fit(X_train_copy)
            transformed_train = preprocessor.transform(X_train_copy)
            selected_features = X_train_copy.columns
            X_train_transformed = pd.DataFrame(transformed_train, columns=selected_features)

            #Transforming X_test
            transformed_test = preprocessor.transform(X_test)
            selected_features = X_test.columns
            X_test_transformed = pd.DataFrame(transformed_test, columns=selected_features)


            # Resetting the indices of the df
            df_train = pd.concat(
                [
                    X_train_transformed.reset_index(drop=True), 
                    y_train.reset_index(drop=True), 
                    event_train.reset_index(drop=True)
                ], 
                axis=1
            )   

            df_train.to_excel("df_train_transfored.xlsx", index=False)


            # Resetting the indices of the df
            df_test = pd.concat(
                [
                    X_test_transformed.reset_index(drop=True), 
                    y_test.reset_index(drop=True), 
                    event_test.reset_index(drop=True)
                ], 
                axis=1
            )   

            # Splitting columns again for model training
            X_train_fold, y_train_fold, event_train_fold = df_train.iloc[:,:-2].values, df_train.iloc[:,-2].values.astype(int), df_train.iloc[:,-1].values
            h = df_train.columns[:-2]
            X_train_fold = pd.DataFrame(X_train_fold, columns=h)
            event_train_fold = pd.DataFrame(event_train_fold)
            y_train_fold = pd.DataFrame(y_train_fold)

            # Splitting columns again for model training
            X_test_fold, y_test_fold, event_test_fold = df_test.iloc[:,:-2].values, df_test.iloc[:,-2].values.astype(int), df_test.iloc[:,-1].values
            h = df_test.columns[:-2]
            X_test_fold = pd.DataFrame(X_test_fold, columns=h)
            event_test_fold = pd.DataFrame(event_test_fold)
            y_test_fold = pd.DataFrame(y_test_fold)


            # Training OSST with the resample train data
            model = OSST(configuration)
            model.fit(X_train_fold, event_train_fold, y_train_fold)
    
  
             # evaluation
            n_leaves = model.leaves()
            n_nodes = model.nodes()
            time = model.time
            
            times_train = np.unique(y_train_fold.values.reshape(-1))
            times_test = np.unique(y_test_fold.values.reshape(-1))
    
            S_hat_train = model.predict_survival_function(X_train_fold)
            estimates_train = np.array([f(times_train) for f in S_hat_train])
    
            S_hat_test = model.predict_survival_function(X_test_fold)
            estimates_test = np.array([f(times_test) for f in S_hat_test])
     
            train_score = harrell_c_index(event_train_fold, y_train_fold, estimates_train, times_train)[0]
            test_score = harrell_c_index(event_test_fold, y_test_fold, estimates_test, times_test)[0]
    

            fold_number += 1
    
            train_scores.append(train_score)
            test_scores.append(test_score)
            times.append(time)
            leaves.append(n_leaves)
            nodes.append(n_nodes)
    
    
        # Create a DataFrame
        df_scores = pd.DataFrame({
        'Train Scores': train_scores,
        'Test Scores': test_scores,
        'Times': times,
        'Leaves': leaves,
        'Nodes': nodes
        })

        # Calculate average scores and standard deviations
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)
        mean_time = np.mean(times)
        mean_leaves = np.mean(leaves)
        mean_nodes = np.mean(nodes)
        std_train_score = np.std(train_scores)
        std_test_score = np.std(test_scores)
        std_time = np.std(times)
        std_leaves = np.std(leaves)
        std_nodes = np.std(nodes)
    
        result_df = pd.DataFrame({
            'Mean Train Score': [mean_train_score],
            'Std Train Score': [std_train_score],
            'Mean Test Score': [mean_test_score],
            'Std Test Score': [std_test_score],
            'Mean model Training Time': [mean_time],
            'Std model Training Time': [std_time],
            'Mean model num of leaves': [mean_leaves],
            'Std model num of leaves': [std_leaves],
            'Mean model num of nodes': [mean_nodes],
            'Std model num of nodes': [mean_nodes],
            'Bucketizer': [bucketizer],
            'Scaler': [scaler],
            'Configuration': [configuration],
        })
    
        # Check if the file exists and append results
        file_path = 'scores.csv'
        file_exists = os.path.isfile(file_path)
        result_df.to_csv(file_path, mode='a', header=not file_exists, index=False)
           
    
    