# import statements 
import os
import sys
import pickle

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from itertools import product
import sksurv.metrics as sksurv_metrics
import random


from survival.init_estimators import init_estimators, set_params_search_space
from survival.CustomModelWrapper import CustomModelWrapper
from helpers.nested_dict import NestedDefaultDict
from helpers.call_OSST_grid_search import call_OSST_grid_search
from helpers.call_simple_OSST import call_simple_OSST
from helpers.call_CoxPH import call_CoxPH_grid_search


from sklearn.feature_selection import SelectKBest,f_classif, VarianceThreshold, RFE

np.random.seed(42)

class Survival:
    def __init__(self, config, progress_manager) -> None:
        self.progress_manager = progress_manager
        self.overwrite = config.meta.overwrite
        self.out_dir = config.meta.out_dir
        self.table_file = os.path.join(self.out_dir, 'results_table.xlsx')
        self.results_file = os.path.join(self.out_dir, 'results.pkl')
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.n_seeds = config.meta.n_seeds
        self.n_workers = config.meta.n_workers
        self.scoring = config.survival.scoring
        self.scalers_dict = config.survival.scalers
        self.selectors_dict = config.survival.feature_selectors
        self.bucketizers_dict = config.survival.bucketizers
        self.models_dict = config.survival.models
        self.n_cv_splits = config.survival.n_cv_splits
        self.n_iter_search = config.survival.n_iter_search
        self.search_type = config.survival.search_type
        self.total_combinations = (
            self.n_seeds
            * sum(self.scalers_dict.values())
            * sum(self.selectors_dict.values())
            * sum(self.bucketizers_dict.values())
        )
        self.result_cols = [
            "Seed",
            "Scaler",
            "Selector",
            "Bucketizer",
            "mean_val_cindex",
            "std_val_cindex",
            "c_index_ipcw",
            "brier_score",
            "auc_mean",
            "auc",
            'evaluation_times',
            'truncation_time'
        ]

        self.selector_params, self.model_params = set_params_search_space()

        
        #print("self.selector_params:", self.selector_params) 
        #print("self.model_params:", self.model_params) 

        try:  # load results if file exists
            if self.overwrite:
                raise FileNotFoundError  # force same behaviour as if file didn't exist

            self.results_table = pd.read_excel(self.table_file)  # human-readable results
            self.row_to_write = self.results_table.shape[0]
            to_concat = pd.DataFrame(
                index=range(self.total_combinations - self.row_to_write),
                columns=self.result_cols,
            )
            self.results_table = pd.concat([self.results_table, to_concat], ignore_index=True)

            with open(self.results_file, 'rb') as file:
                self.results = pickle.load(file)  # results for report
        except FileNotFoundError:
            self.results_table = pd.DataFrame(
                index=range(self.total_combinations),
                columns=self.result_cols,
            )
            self.row_to_write = 0

            self.results = NestedDefaultDict()

    def __call__(self, seed, x_train, y_train, x_test, y_test):
        self.seed = seed
        self.scalers, self.selectors, self.bucketizers, self.models = init_estimators(
            self.seed, self.n_workers, self.scalers_dict, self.selectors_dict, self.bucketizers_dict, self.models_dict,  self.scoring
        )
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.results[self.seed]['x_train'] = x_train
        self.results[self.seed]['y_train'] = y_train
        self.results[self.seed]['x_test'] = x_test
        self.results[self.seed]['y_test'] = y_test
        self.fit_and_evaluate_pipeline()

        return self.results_table

    
    def fit_and_evaluate_pipeline(self):
        pbar = self.progress_manager.counter(
            total=self.total_combinations, desc="Training and evaluating all combinations", unit='it', leave=False
        )
        for scaler_name, scaler in self.scalers.items():
            for selector_name, selector in self.selectors.items():
                for bucketizer_name, bucketizer in self.bucketizers.items():
                        y_col_train = self.y_train[self.time_column]
                        y_col_test = self.y_test[self.time_column]
                        event_col_train = self.y_train[self.event_column]
                        event_col_test = self.y_test[self.event_column]
                       
                        try:
                            if (  # skip if already evaluated
                                (self.results_table["Seed"] == self.seed)
                                & (self.results_table["Scaler"] == scaler_name)
                                & (self.results_table["Selector"] == selector_name)
                                & (self.results_table["Bucketizer"] == bucketizer_name)
                            ).any():
                                logger.info(f"Skipping {scaler_name} - {selector_name} - {bucketizer_name}")
                                pbar.update()
                                continue
                            
                            logger.info(f"Training {scaler_name} - {selector_name} - {bucketizer_name}")
                            row = {"Seed": self.seed, "Scaler": scaler_name, "Selector": selector_name, "Bucketizer": bucketizer_name}
                            # Create pipeline and parameter grid
                            cv = StratifiedKFold(n_splits=self.n_cv_splits, random_state=self.seed, shuffle=True)
                            stratified_folds = [x for x in cv.split(self.x_train, self.y_train[self.event_column])]
                            self.tau = np.min(  # truncation time
                                [np.max(self.y_train[self.time_column][train_idx]) for train_idx, _ in stratified_folds]) - 1

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
  
                            preprocessor = Pipeline(
                                    [
                                        ("selector", selector)                   
                                    ]
                            )

                            ## transforming the test and train as the out of the preprocessor into a format that OSST will take in
                            
                            #print(self.y_train[self.event_column])
                            preprocessor.fit(self.x_train, self.y_train)
                            transformed_train = preprocessor.transform(self.x_train)

                            #print(transformed_X.shape)
                            selected_features = self.x_train.columns[preprocessor.named_steps['selector'].get_support()]  # get selected feature names
                            transformed_train_df = pd.DataFrame(transformed_train, columns=selected_features)
                            print(transformed_train_df.shape) 

                            transformed_test = preprocessor.transform(self.x_test)

                            #print(transformed_X.shape)
                            selected_features = self.x_test.columns[preprocessor.named_steps['selector'].get_support()]  # get selected feature names
                            transformed_test_df = pd.DataFrame(transformed_test, columns=selected_features)
                            print(transformed_test_df.shape) 
                 
                            
                            ## choosing which grid search will be executed based on the config file
                            if self.search_type.get('SimpleExecution', False):
                                print("simple search is true")
                                call_simple_OSST(config, transformed_train_df, y_col_train, event_col_train, transformed_test_df, y_col_test, event_col_test, scaler, bucketizer)
        
                            elif self.search_type.get('GridSearch', False):
                                print("grid search is true")
                                #print("self.model_params:", self.model_params) 

                                params = self.model_params['OSST']

                                keys, values = zip(*params.items())
                                all_combinations = [dict(zip(keys, v)) for v in product (*values)]

                                #print("all combinations", all_combinations)

                                for combination in all_combinations:
                                    config.update(combination)
                                    call_OSST_grid_search(config, transformed_train_df, y_col_train, event_col_train, scaler, bucketizer)
                                    #call_CoxPH_grid_search(config, transformed_train_df, y_col_train, event_col_train, scaler, bucketizer)
                               
                            elif self.search_type.get('RandomSearch', False):
                                print("random search is true")
                                params = self.model_params['OSST']
                                keys, values = zip(*params.items())

                                combination_num = 5
                                
                                
                                random_combinations = [
                                    dict(zip(keys, [random.choice(v) for v in values])) 
                                    for _ in range (combination_num)
                                ]

                                
                                for combination in random_combinations:
                                    config.update(combination)
                                    call_OSST_grid_search(config, transformed_train_df, y_col_train, event_col_train, scaler, bucketizer)

                                                                
                                

                                            
                                                   
                        except Exception as e:
                            print(
                                f"Error encountered for Scaler={scaler_name}, Selector={selector_name}, Bucketizer={bucketizer_name}. Error message: {str(e)}")

        pbar.close()

  
    