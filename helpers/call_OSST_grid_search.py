from sklearn.model_selection import StratifiedKFold

from osst.model.osst import OSST
from osst.model.metrics import harrell_c_index, uno_c_index, integrated_brier_score, cumulative_dynamic_auc, compute_ibs_per_sample

import numpy as np
import pandas as pd
import os

def call_OSST_grid_search(configuration, X, y, event):

        y = pd.DataFrame(y, columns=['Time'])
        event = pd.DataFrame(event, columns=['Event'])
    
        skf = StratifiedKFold(n_splits=4)
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
    
    
            # Training OSST with the resample train data
            model = OSST(configuration)
            model.fit(X_train_fold, event_train_fold, y_train_fold)
    
            # evaluation
            n_leaves = model.leaves()
            n_nodes = model.nodes()
            time = model.time
            print("Model training time: {}".format(time))
            print("# of leaves: {}".format(n_leaves))
    
            print("Train IBS score: {:.6f} , Test IBS score: {:.6f}".format(\
            model.score(X_train_fold, event_train_fold, y_train_fold), model.score(X_test_fold, event_test_fold, y_test_fold)))
    
            times_train = np.unique(y_train_fold.values.reshape(-1))
            times_test = np.unique(y_test_fold.values.reshape(-1))
    
            S_hat_train = model.predict_survival_function(X_train_fold)
            estimates_train = np.array([f(times_train) for f in S_hat_train])
    
            S_hat_test = model.predict_survival_function(X_test_fold)
            estimates_test = np.array([f(times_test) for f in S_hat_test])
    
            print("Train Harrell's c-index: {:.6f}, Test Harrell's c-index: {:.6f}\n".format(\
                harrell_c_index(event_train_fold, y_train_fold, estimates_train, times_train)[0], \
                harrell_c_index(event_test_fold, y_test_fold, estimates_test, times_test)[0]))
    
            train_score = harrell_c_index(event_train_fold, y_train_fold, estimates_train, times_train)[0]
            test_score = harrell_c_index(event_test_fold, y_test_fold, estimates_test, times_test)[0]
    
            if test_score > 0.7:
                print(f"For fold {fold_number} the test score is {test_score} for configuration {configuration} \n")
    
            fold_number += 1
    
            train_scores.append(train_score)
            test_scores.append(test_score)
    
        # Create a DataFrame
        df_scores = pd.DataFrame({
        'Train Scores': train_scores,
        'Test Scores': test_scores
        })
    
        # Print the DataFrame
        print(df_scores)
    
        # Calculate average scores and standard deviations
        mean_train_score = np.mean(train_scores)
        mean_test_score = np.mean(test_scores)
        std_train_score = np.std(train_scores)
        std_test_score = np.std(test_scores)
    
    
        print(f'Mean Training Score: {mean_train_score}')
        print(f'Mean Test Score: {mean_test_score}\n')
    
        if mean_test_score > 0.6:
                print("The mean_test_score is", mean_test_score, "this configuration", configuration)
    
        result_df = pd.DataFrame({
            'Mean Train Score': [mean_train_score],
            'Std Train Score': [std_train_score],
            'Mean Test Score': [mean_test_score],
            'Std Test Score': [std_test_score],
            'Configuration': [configuration]
        })
    
        # Check if the file exists and append results
        #file_path = 'Nov14_scores.csv'
        #file_exists = os.path.isfile(file_path)
        #result_df.to_csv(file_path, mode='a', header=not file_exists, index=False)

