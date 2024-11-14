
from osst.model.osst import OSST
from osst.model.metrics import harrell_c_index, uno_c_index, integrated_brier_score, cumulative_dynamic_auc, compute_ibs_per_sample


def call_simple_OSST(config, x_train, y_col_train, event_col_train, x_test, y_col_test, event_col_test):
                            model = OSST(config)
                            model.fit(x_train, event_col_train, y_col_train)
                    
                            # evaluation
                            n_leaves = model.leaves()
                            n_nodes = model.nodes()
                            time = model.time
                            print("Model training time: {}".format(time))
                            print("# of leaves: {}".format(n_leaves))
    
                    
                            print("Train IBS score: {:.6f} , Test IBS score: {:.6f}".format(\
                            model.score(x_train, event_col_train, y_col_train),  model.score(x_test, event_col_test, y_col_test)))
