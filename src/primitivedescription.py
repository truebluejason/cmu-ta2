import pandas as pd
import numpy as np
import math, sys
from sklearn import metrics
from sklearn import preprocessing
import bo.gp_call
import problem_pb2
import util
import logging
import d3m.index

logging.basicConfig(level=logging.INFO)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.model_selection import GridSearchCV

gridsearch_estimators_parameters = {'d3m.primitives.regression.random_forest.SKlearn': [RandomForestRegressor(), 
                                                                                        {'n_estimators': [100],
                                                                                         'max_depth': [8, 10, 15, None],
                                                                                         'min_samples_split': [2, 5, 10]}],
              'd3m.primitives.classification.random_forest.SKlearn': [RandomForestClassifier(),
                                                                      {'n_estimators': [100],
                                                                       'max_depth': [8, 10, 15, None],
                                                                       'min_samples_split': [2, 5, 10],
                                                                       'class_weight': ['balanced', None]}],
              'd3m.primitives.classification.extra_trees.SKlearn': [ExtraTreesClassifier(),
                                                                      {'n_estimators': [100],
                                                                       'max_depth': [8, 10, 15, None],
                                                                       'min_samples_split': [2, 5, 10],
                                                                       'class_weight': ['balanced', None]}],
              'd3m.primitives.regression.extra_trees.SKlearn': [ExtraTreesRegressor(),
                                                                      {'n_estimators': [100],
                                                                       'max_depth': [8, 10, 15, None],
                                                                       'min_samples_split': [2, 5, 10]}],
              'd3m.primitives.classification.linear_svc.SKlearn': [LinearSVC(),
                                                                   {'C': [0.01, 0.1, 1, 10, 100],
                                                                    'class_weight': ['balanced', None]}],
              'd3m.primitives.regression.linear_svr.SKlearn': [LinearSVR(),
                                                                   {'C': [0.01, 0.1, 1, 10, 100]}],
              'd3m.primitives.classification.svc.SKlearn': [SVC(),
                                                            {'C': [0.01, 0.1, 1, 10, 100],
                                                             'gamma': [0.01, 0.1, 1, 10],
                                                             'class_weight': ['balanced', None]}],
              'd3m.primitives.regression.svr.SKlearn': [SVR(),
                                                        {'C': [0.01, 0.1, 1, 10, 100],
                                                         'gamma': [0.01, 0.1, 1, 10]}],
              'd3m.primitives.classification.logistic_regression.SKlearn': [LogisticRegression(),
                                                                            {'C': [0.1, 1, 10, 100],
                                                                             'class_weight': ['balanced', None]}],
              'd3m.primitives.classification.sgd.SKlearn': [SGDClassifier(),
                                                            {'class_weight': ['balanced', None]}],
              'd3m.primitives.regression.ridge.SKlearn': [Ridge(), 
                                                          {'alpha': [0.001, 0.01, 0.1, 1, 5]}],
              'd3m.primitives.regression.lasso.SKlearn': [Lasso(),
                                                          {'alpha': [0.001, 0.01, 0.1, 1, 5]}]
}
             
def rmse(y_true, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_true, y_pred))

class PrimitiveDescription(object):
    """
    Class representing single primitive.
    Used for optimizing primitive hyper-parameters, doing cross-validation.
    """
    def __init__(self, primitive, primitive_class):
        self.id = primitive_class.id
        self.primitive = primitive
        self.primitive_class = primitive_class

    def get_num_splits(self, length, cols):
        splits = 2
        if length < 500:
            splits = 50
        elif length < 1000:
            splits = 25
        elif length < 2500:
            splits = 20
        elif length < 5000:
            splits = 10
        elif length < 10000:
            splits = 5
        elif length < 20000:
            splits = 3
        else:
            splits = 2
        return splits

        
    def score_primitive(self, X, y, metric_type, posLabel, custom_hyperparams, step_index=0):
        """
        Learns optimal hyperparameters for the primitive
        Returns metric score and optimal parameters.
        """
        python_path = self.primitive.metadata.query()['python_path']

        optimal_params = dict()

        if custom_hyperparams is not None:
            for name, value in custom_hyperparams.items():
                optimal_params[name] = value

        if 'MeanBaseline' in python_path in python_path:
            if util.invert_metric(metric_type) is True:
                return (1000000000.0, optimal_params)
            else:
                return (0.0, optimal_params)

        if y is None or 'graph' in python_path or 'link' in python_path or 'community' in python_path:
            if util.invert_metric(metric_type) is True:
                return (0.0, optimal_params)
            else:
                return (1.0, optimal_params)

        if 'Find_projections' in python_path and 'Numeric' not in python_path:
            rows = len(X)
            min_rows = (int)(rows * 0.8 * 0.5)
            if min_rows < 100:
                optimal_params['support'] = min_rows

        # Do grid-search to learn optimal parameters for the model
        params = None
        #try:
        #    params = self.optimize_primitive_gridsearch(X, np.ravel(y), python_path, metric_type, posLabel)
        #except:
        #    print("optimize_primitive_gridsearch: ", sys.exc_info()[0])
        #    params = None

        if params is not None:
            for name, value in params.items():
                optimal_params[name] = value

        primitive_hyperparams = self.primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        prim_instance = self.primitive(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults(), **optimal_params))
        score = 0.0

        splits = self.get_num_splits(len(X), len(X.columns))
        # Run k-fold CV and compute mean metric score
        (score, metric_scores) = self.k_fold_CV(prim_instance, X, y, metric_type, posLabel, splits)
        mean = np.mean(metric_scores)
        metric_scores.sort()
        median = np.median(metric_scores)
        lb = max((int)(0.025*len(metric_scores) + 0.5)-1,0)
        ub = min((int)(0.975*len(metric_scores) + 0.5)-1, len(metric_scores)-1)
        stderror = np.std(metric_scores)/math.sqrt(len(metric_scores))
        z = 1.96*stderror
        logging.info("CV scores for %s = %s(%s - %s) k = %s", python_path, mean, mean-z, mean+z, len(metric_scores))
        logging.info("CV scores for %s = %s(%s - %s) k = %s", python_path, median, metric_scores[lb], metric_scores[ub], len(metric_scores))
        #print("CV scores for ", python_path, " = ", mean, "(", mean-z, "-", mean+z, ") k = ", len(metric_scores))
        #print("CV scores for ", python_path, " = ", median, "(", metric_scores[lb], "-", metric_scores[ub], ") k = ", len(metric_scores))
        #print("Second last score: ", python_path, " = ", metric_scores[1])
        #text = python_path + "," + str(mean) + "," + str(mean-z) + "," + str(median) + "," + str(metric_scores[lb]) + "," + str(metric_scores[1]) + "\n"
        #filename = "scores.csv"
        #with open(filename, "a") as g:
        #    g.write(text)
        return (score, optimal_params)

    def evaluate_metric(self, predictions, Ytest, metric, posLabel):
        """
        Function to compute metric score for predicted-vs-true output.
        """
        count = len(Ytest)

        if metric is problem_pb2.ACCURACY:
            return metrics.accuracy_score(Ytest, predictions)
        elif metric is problem_pb2.PRECISION:
            return metrics.precision_score(Ytest, predictions)
        elif metric is problem_pb2.RECALL:
            return metrics.recall_score(Ytest, predictions)
        elif metric is problem_pb2.F1:
            return metrics.f1_score(Ytest, predictions, pos_label=posLabel)
        elif metric is problem_pb2.F1_MICRO:
            return metrics.f1_score(Ytest, predictions, average='micro')
        elif metric is problem_pb2.F1_MACRO:
            return metrics.f1_score(Ytest, predictions, average='macro')
        elif metric is problem_pb2.ROC_AUC:
            return metrics.roc_auc_score(Ytest, predictions)
        elif metric is problem_pb2.ROC_AUC_MICRO:
            return metrics.roc_auc_score(Ytest, predictions, average='micro')
        elif metric is problem_pb2.ROC_AUC_MACRO:
            return metrics.roc_auc_score(Ytest, predictions, average='macro')
        elif metric is problem_pb2.MEAN_SQUARED_ERROR:
            return metrics.mean_squared_error(Ytest, predictions)
        elif metric is problem_pb2.ROOT_MEAN_SQUARED_ERROR:
            return math.sqrt(metrics.mean_squared_error(Ytest, predictions))
        elif metric is problem_pb2.ROOT_MEAN_SQUARED_ERROR_AVG:
            return math.sqrt(metrics.mean_squared_error(Ytest, predictions))
        elif metric is problem_pb2.MEAN_ABSOLUTE_ERROR:
            return metrics.mean_absolute_error(Ytest, predictions)
        elif metric is problem_pb2.R_SQUARED:
            return metrics.r2_score(Ytest, predictions)
        elif metric is problem_pb2.NORMALIZED_MUTUAL_INFORMATION:
            return metrics.normalized_mutual_info_score(Ytest, predictions)
        elif metric is problem_pb2.JACCARD_SIMILARITY_SCORE:
            return metrics.jaccard_similarity_score(Ytest, predictions)
        elif metric is problem_pb2.PRECISION_AT_TOP_K:
            return 0.0
        elif metric is problem_pb2.OBJECT_DETECTION_AVERAGE_PRECISION:
            return 0.0
        else:
            return metrics.accuracy_score(Ytest, predictions)

    def optimize_primitive_gridsearch(self, train, output, python_path, metric, posLabel):
        # Do grid-search to learn optimal parameters for the model
        if python_path in gridsearch_estimators_parameters:
            from timeit import default_timer as timer
            start = timer()
            (model, search_grid) = gridsearch_estimators_parameters[python_path]
            splits = self.get_num_splits(len(train), len(train.columns))

            if 'k_neighbors' in python_path:
                training_samples = len(train)*(splits-1)/splits
                import copy
                n_neighbors = copy.deepcopy(search_grid['n_neighbors'])
                for n in n_neighbors:
                    if n >= training_samples:
                        search_grid['n_neighbors'].remove(n)

            from sklearn.metrics import make_scorer
            if metric is problem_pb2.ACCURACY:
                scorer = make_scorer(metrics.accuracy_score)
            elif metric is problem_pb2.F1:
                scorer = make_scorer(metrics.f1_score, pos_label=posLabel)
            elif metric is problem_pb2.F1_MACRO:
                scorer = make_scorer(metrics.f1_score, average='macro')
            elif metric is problem_pb2.MEAN_SQUARED_ERROR:
                scorer = make_scorer(metrics.mean_squared_error, greater_is_better=False)
            elif metric is problem_pb2.ROOT_MEAN_SQUARED_ERROR: 
                scorer = make_scorer(rmse, greater_is_better=False)
            else:
                scorer = None

            rf_random = GridSearchCV(estimator = model, param_grid = search_grid, scoring = scorer, cv = splits, verbose=2, n_jobs = -1)

            # Fit the random search model
            rf_random.fit(train, output)
            print(rf_random.best_params_)
            end = timer() 
            print("Time taken for ", python_path, " = ", end-start, " secs")
            return rf_random.best_params_
        else:
            print("No grid search done for ", python_path)
            return None

    def optimize_corex(self, train, y, python_path, metric_type, posLabel):
        corex_hp = {'n_hidden': [5, 10],
                    'threshold': [0, 500],
                    'n_grams': [1, 3]
                   }

        primitive_hyperparams = self.primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        custom_hyperparams = dict()
        custom_hyperparams['use_semantic_types'] = True
        custom_hyperparams['n_estimators'] = 10
        rf_model = self.primitive(hyperparams=primitive_hyperparams(
                    primitive_hyperparams.defaults(), **custom_hyperparams))

        prim = d3m.index.get_primitive(python_path)
        primitive_hyperparams = prim.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        for i in corex_hp['n_hidden']:
            for j in corex_hp['threshold']:
                for k in corex_hp['n_grams']:
                    custom_hyperparams = dict()
                    custom_hyperparams['n_hidden'] = i
                    custom_hyperparams['threshold'] = j
                    custom_hyperparams['n_grams'] = k
                    model = prim(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults(), **custom_hyperparams))

                    model.set_training_data(inputs=train)
                    model.fit()
                    output = model.produce(inputs=train).value

                    (score, metric_scores) = self.k_fold_CV(rf_model, output, y, metric_type, posLabel, 5)
                    mean = np.mean(metric_scores)
                    print("n_hidden = ", i, " threshold = ", j, " n_grams = ", k, " score = ", mean)
            

    def optimize_primitive(self, train, output, inputs, primitive_hyperparams, optimal_params, hyperparam_types, metric_type, posLabel):
        """
        Function to evaluate each input point in the hyper parameter space.
        This is called for every input sample being evaluated by the bayesian optimization package.
        Return value from this function is used to decide on function optimality.
        """
        custom_hyperparams=dict()
        for index,name in optimal_params.items():
            value = inputs[index]
            if hyperparam_types[name] is int:
                value = (int)(inputs[index]+0.5)
            custom_hyperparams[name] = value

        prim_instance = self.primitive(hyperparams=primitive_hyperparams(primitive_hyperparams.defaults(), **custom_hyperparams))

        metric = self.k_fold_CV(prim_instance, train, output, metric_type, posLabel, 2)
        if util.invert_metric(metric_type) is True:
            metric = metric * (-1)
        print('Metric: %f' %(metric))
        return metric

    def k_fold_CV(self, prim_instance, X, y, metric_type, posLabel, splits):
        """
        Run k-fold CV.
        k = splits
        prim_instance has already been initialized with hyperparameters.
        """

        python_path = self.primitive.metadata.query()['python_path']
        metric_sum = 0

        if isinstance(X, pd.DataFrame):
            Xnew = pd.DataFrame(data=X.values)
        else:
            Xnew = pd.DataFrame(data=X)

        score = 0.0

        # Run k-fold CV and compute mean metric score
        print(Xnew.shape)
        metric_scores = []
        if 'classification' in python_path: # Classification
            frequencies = y.iloc[:,0].value_counts()
            min_freq = frequencies[len(frequencies)-1]
            if min_freq < splits:
                from sklearn.model_selection import KFold as KFold
                kf = KFold(n_splits=splits, shuffle=True, random_state=9001)
                split_indices = kf.split(X)
            else:
                from sklearn.model_selection import StratifiedKFold as KFold
                kf = KFold(n_splits=splits, shuffle=True, random_state=9001)
                split_indices = kf.split(X, y)
        else: # Regression
            from sklearn.model_selection import KFold as KFold
            kf = KFold(n_splits=splits, shuffle=True, random_state=9001)
            split_indices = kf.split(X)

        # Do the actual k-fold CV here
        for train_index, test_index in split_indices:
            X_train, X_test = Xnew.iloc[train_index], Xnew.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train.metadata = X.metadata
            X_test.metadata = X.metadata

            prim_instance.set_training_data(inputs=X_train, outputs=y_train)

            prim_instance.fit()
            predictions = prim_instance.produce(inputs=X_test).value
            if len(predictions.columns) > 1:
                predictions = predictions.iloc[:,len(predictions.columns)-1]
            metric = self.evaluate_metric(predictions, y_test, metric_type, posLabel)
            metric_scores.append(metric)
            metric_sum += metric

        score = metric_sum/splits
        return (score, metric_scores)

    def optimize_hyperparams(self, train, output, lower_bounds, upper_bounds, hyperparam_types, hyperparam_semantic_types,
     metric_type, custom_hyperparams):
        """
        Optimize primitive's hyper parameters using Bayesian Optimization package 'bo'.
        Optimization is done for the numerical parameters with specified range(lower - upper).
        """
        domain_bounds = []
        optimal_params = {}
        index = 0
        optimal_found_params = {}

        # Create parameter ranges in domain_bounds. 
        # Map parameter names to indices in optimal_params
        for name,value in lower_bounds.items():
            if custom_hyperparams is not None and name in custom_hyperparams.keys():
                continue
            sem = hyperparam_semantic_types[name]
            if "https://metadata.datadrivendiscovery.org/types/TuningParameter" not in sem:
                continue
            lower = lower_bounds[name]
            upper = upper_bounds[name]
            if lower is None or upper is None:
                continue
            domain_bounds.append([lower,upper])
            optimal_params[index] = name
            index =index+1

        python_path = self.primitive.metadata.query()['python_path']
        if python_path == 'd3m.primitives.sri.psl.RelationalTimeseries':
            optimal_params[0] = 'period'
            index =index+1
            domain_bounds.append([1,20])

        if index == 0:
            return optimal_found_params

        primitive_hyperparams = self.primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        func = lambda inputs : self.optimize_primitive(train, output, inputs, primitive_hyperparams, optimal_params, hyperparam_types, metric_type)

        try:
            (curr_opt_val, curr_opt_pt) = bo.gp_call.fmax(func, domain_bounds, 50)
        except:
            print("optimize_hyperparams: ", sys.exc_info()[0])
            print(self.primitive)
            optimal_params = None

        # Map optimal parameter values found
        if optimal_params != None:
            for index,name in optimal_params.items():
                value = curr_opt_pt[index]
                if hyperparam_types[name] is int:
                    value = (int)(curr_opt_pt[index]+0.5)
                optimal_found_params[name] = value

        return optimal_found_params

    def find_optimal_hyperparams(self, train, output, hyperparam_spec, metric, custom_hyperparams):
        filter_hyperparam = lambda vl: None if vl == 'None' else vl
        hyperparam_lower_ranges = {name:filter_hyperparam(vl['lower']) for name,vl in hyperparam_spec.items() if 'lower' in vl.keys()}
        hyperparam_upper_ranges = {name:filter_hyperparam(vl['upper']) for name,vl in hyperparam_spec.items() if 'upper' in vl.keys()}
        hyperparam_types = {name:filter_hyperparam(vl['structural_type']) for name,vl in hyperparam_spec.items() if 'structural_type' in vl.keys()}
        hyperparam_semantic_types = {name:filter_hyperparam(vl['semantic_types']) for name,vl in hyperparam_spec.items() if 'semantic_types' in vl.keys()}
        optimal_hyperparams = {}
        if len(hyperparam_lower_ranges) > 0:
            optimal_hyperparams = self.optimize_hyperparams(train, output, hyperparam_lower_ranges, hyperparam_upper_ranges,
             hyperparam_types, hyperparam_semantic_types, metric, custom_hyperparams)
            print("Optimals: ", optimal_hyperparams)

        return optimal_hyperparams

