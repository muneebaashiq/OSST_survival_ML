from sklearn.pipeline import Pipeline
from osst.model.osst import OSST
from osst.model.metrics import harrell_c_index, uno_c_index, integrated_brier_score, cumulative_dynamic_auc, compute_ibs_per_sample
import pandas as pd
import numpy as np

def call_simple_OSST(config, x_train, y_col_train, event_col_train, x_test, y_col_test, event_col_test, scaler, bucketizer):
    
                            preprocessor = Pipeline(
                                    [
                                        ("scaler", scaler),
                                        ("bucketizer", bucketizer)
                                    ]
                            )

    
                            #Transforming x_train
                            preprocessor.fit(x_train)
                            transformed_train = preprocessor.transform(x_train)
                            selected_features = x_train.columns
                            transformed_train_df = pd.DataFrame(transformed_train, columns=selected_features)
                            #print(transformed_train_df)

                            #Transforming x_train and x_test
                            transformed_test = preprocessor.transform(x_test)
                            selected_features = x_test.columns
                            transformed_test_df = pd.DataFrame(transformed_test, columns=selected_features)
                            #print(transformed_test_df)


                            #Model building
                            model = OSST(config)
                            model.fit(transformed_train_df, event_col_train, y_col_train)
                    
                            # evaluation
                            n_leaves = model.leaves()
                            n_nodes = model.nodes()
                            time = model.time
                            print("Model training time: {}".format(time))
                            print("# of leaves: {}".format(n_leaves))
    
                    
                            print("Train IBS score: {:.6f} , Test IBS score: {:.6f}".format(\
                            model.score(transformed_train_df, event_col_train, y_col_train),  model.score(transformed_test_df, event_col_test, y_col_test)))

    
                            times_train = np.unique(y_col_train.reshape(-1))
                            times_test = np.unique(y_col_test.reshape(-1))
                            
                            S_hat_train = model.predict_survival_function(transformed_train_df)
                            estimates_train = np.array([f(times_train) for f in S_hat_train])
                            
                            S_hat_test = model.predict_survival_function(transformed_test_df)
                            estimates_test = np.array([f(times_test) for f in S_hat_test])
                            
                            print("Train Harrell's c-index: {:.6f}, Test Harrell's c-index: {:.6f}".format(\
                                harrell_c_index(event_col_train, y_col_train, estimates_train, times_train)[0], \
                                harrell_c_index(event_col_test, y_col_test, estimates_test, times_test)[0]))
                            
                            print("Train Uno's c-index: {:.6f}, Test Uno's c-index: {:.6f}".format(\
                                uno_c_index(event_col_train, y_col_train, event_col_train, y_col_train, estimates_train, times_train)[0],\
                                uno_c_index(event_col_test, y_col_test, event_col_test, y_col_test, estimates_test, times_test)[0]))
                            
                            print("Train AUC: {:.6f}, Test AUC: {:.6f}".format(\
                                cumulative_dynamic_auc(event_col_train, y_col_train, event_col_train, y_col_train, estimates_train, times_train)[0],\
                                cumulative_dynamic_auc(event_col_test, y_col_test, event_col_test, y_col_test, estimates_test, times_test)[0]))
                            
                            print(model.tree)