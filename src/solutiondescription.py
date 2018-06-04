"""
This should be where everything comes together: Problem descriptions get matched up
with the available primitives, and a plan for how to create a solution gets made.

    So it needs to:
    Choose hyperparameters for a primitive
    Run the primitive
    Measure the results
    Feed the results and hyperparameters back into the chooser
"""

import importlib

import logging
import core_pb2
import pandas as pd

from  api_v3 import core

import uuid, sys
from urllib import request as url_request
from urllib import parse as url_parse

class ProblemDescription(object):
    """
    Basically a PipelineCreateRequest; it describes a problem to solve.
    Each PipelineDescription object is then a possible solution for solving this problem.
    """
    def __init__(self, name, dataset_uri, output_dir, task_type, metrics, target_features, predict_features):
        self._name = name
        self._dataset_uri = dataset_uri
        self._task_type = task_type
        self._evaluation_metrics = metrics
        self._target_features = target_features
        self._predict_features = predict_features
        self._output_dir = output_dir

def load_dataset(dataset_spec_uri):
    """Loads a dataset spec URI and does all the annoying
    preprocessing that needs to be done.

    Returns two numpy arrays: (inputs, labels)
    """
    with url_request.urlopen(dataset_spec_uri) as uri:
        res = uri.read()
        # We need to pull the file root path out of the dataset
        # source the TA3 gave us and give it to the DatasetSpec so it
        # knows where to find the actual files
        dataset_root = core.dataset_uri_path(dataset_spec_uri)

        dataset_spec = core.DatasetSpec.from_json_str(res, dataset_root)
        logging.info("Task created, outputting to %s", dataset_root)

    resource_specs = {
        resource.res_id:resource for resource in dataset_spec.resource_specs
    }

    datasets = {
        resource.res_id:resource.load() for resource in dataset_spec.resource_specs
    }

    import numpy as np
    # We have no good way of doing multiple datasets so we just grab the first one
    (resource_name, train_data) = next(iter(datasets.items()))
    logging.info(resource_name)

    # Also filter out NaN's since pandas uses them for missing values.
    # TODO: Right now we replace them with 0, which is not often what we want...
    train_data.fillna(0, inplace=True)

    # Some primitives don't know how to take string type data
    # so we convert categorical to int's
    resource_spec = resource_specs[resource_name]
    # Convert categorical stuff from strings to category's
    for column in resource_spec.columns:
        if column.col_type == 'categorical':
            name = column.col_name
            train_data[name] = train_data[name].astype('category')
            train_data[name] = train_data[name].cat.codes

    label_columns = [
        column.col_name for column in resource_spec.columns
        if 'suggestedTarget' in column.role
    ]

    data_columns = [
        column.col_name for column in resource_spec.columns
        if 'suggestedTarget' not in column.role
    ]

    inputs = train_data[data_columns]
    labels = train_data[label_columns]
    return (inputs, labels)

class SolutionDescription(object):
    """
    A wrapper of a primitive instance and hyperparameters, ready to have inputs
    fed into it.

    The idea is that this can be evaluated, produce a model and performance metrics,
    and the hyperparameter tuning can consume that and choose what to do next.

    Output is fairly basic right now; it writes to a single numpy CSV file with a given name
    based off the results of the primitive (numpy arrays only atm)
    """

    def __init__(self, hyperparam_spec, primitive, hyperparams):
        self.id = str(uuid.uuid4())
        self.hyperparam_spec = hyperparam_spec
        self.primitive = primitive
        self.hyperparams = hyperparams
        self.prim_instance = None

    def train(self, X, y):
        """
        Trains the model.
        """
        if self.hyperparams == None:
            optimal_params = self.find_optimal_hyperparams(train=X, output=y) 
            self.hyperparams = optimal_params

        self.prim_instance = self.primitive(self.hyperparams)
        self.prim_instance.set_training_data(inputs=X.values, outputs=y.values)
        res = self.prim_instance.fit()
        print("TRAINING Done:", res.has_finished)
        print("Iterations:", res.iterations_done)
        print("Value:", res.value)

    def score_solution(self, X, y, mode):
        """
        Learns optimal hyperparameters for the primitive
        Evaluates model on inputs X and outputs y
        Returns metric.
        """
        optimal_params = self.find_optimal_hyperparams(train=X, output=y) 
        self.hyperparams = optimal_params

        from sklearn.model_selection import KFold

        kf = KFold(n_splits=3, shuffle=True, random_state=9001)
      
        splits = 3 
        metric_sum = 0

        prim_instance = self.primitive(hyperparams=optimal_params)
        score = 0.0
        try:
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                prim_instance.set_training_data(inputs=X_train.values, outputs=y_train.values)
                prim_instance.fit()
                predictions = prim_instance.produce(inputs=X_test).value                        
                metric = self.evaluate_metric(predictions, y_test)     
                metric_sum += metric

            score = metric_sum/splits
        except:
             score = 0.0
        return score
 
    def training_finished(self):
        """
        Yields false until the model is finished training, true otherwise.
        """
        yield self.train_result.has_finished

    def produce(self, X):
        """
        Runs the solution.  Returns two things: a model, and a score.
        """
        assert self.train_result.has_finished
        res = self.prim_instance.produce(inputs=X.values, timeout=1000.0, iterations=1)
        print("TESTING: Done:", res.has_finished)
        
        inputs['prediction'] = pd.Series(res.value)
        print(inputs[['d3mIndex', 'prediction']])

    def evaluate_metric(self, predictions, Ytest):
        """
        Function to compute prediction accuracy for classifiers.
        """
        correct = 0
        count = len(Ytest)

        for i in range(count):
            if predictions[i] == Ytest.iloc[i,0]:
                correct=correct+1

        return (correct/count)

    def compute_metric(self, predictions, Ytest, metric):
        count = len(Ytest)
        
        if metric is problem_pb2.PerformanceMetric.ACCURACY:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.PRECISION:
            class_count = 0
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
                if predictions[i] == 1:
                    class_count=class_count+1
            return (correct/class_count)
        elif metric is problem_pb2.PerformanceMetric.RECALL:
            class_count = 0
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
                if Ytest.iloc[i,0] == 1:
                    class_count=class_count+1
            return (correct/class_count)
        elif metric is problem_pb2.PerformanceMetric.F1:
            class_count = 0
            true_class_count = 0
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
                if Ytest.iloc[i,0] == 1:
                    true_class_count=true_class_count+1
                if predictions[i] == 1:
                    class_count=class_count+1
                precision = (correct/class_count)
                recall = (correct/true_class_count)
            return (2.0*precision*recall/(precision+recall))
        elif metric is problem_pb2.PerformanceMetric.F1_MICRO:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.F1_MACRO:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.ROC_AUC:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.ROC_AUC_MICRO:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.ROC_AUC_MACRO:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.MEAN_SQUARED_ERROR:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.ROOT_MEAN_SQUARED_ERROR:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.ROOT_MEAN_SQUARED_ERROR_AVG:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.MEAN_ABSOLUTE_ERROR:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.R_SQUARED:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.NORMALIZED_MUTUAL_INFORMATION:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.JACCARD_SIMILARITY_SCORE:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.PRECISION_AT_TOP_K:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        elif metric is problem_pb2.PerformanceMetric.OBJECT_DETECTION_AVERAGE_PRECISION:
            for i in range(count):
                if predictions[i] == Ytest.iloc[i,0]:
                    correct=correct+1
            return (correct/count)
        else:
            return -1


    def evaluate_regression_metric(self, predictions, Ytest):
        """
        Function to compute R2 for regressors.
        """
        count = len(Ytest)
        sum_pred_error = 0
        sum_true_error = 0
        mean = Ytest.mean()
        for i in range(count):
            pred_error = predictions[i] - Ytest.iloc[i,0]
            pred_error = pred_error * pred_error
            sum_pred_error = sum_pred_error + pred_error

            true_error = Ytest.iloc[i,0] - mean
            true_error = true_error * true_error
            sum_true_error = sum_true_error + true_error

        r2 = 1.0 - (sum_pred_error/sum_true_error)
        return r2 

    def optimize_primitive(self, train, output, inputs, default_params, optimal_params, hyperparam_types):
        """
        Function to evaluate each input point in the hyper parameter space.
        This is called for every input sample being evaluated by the bayesian optimization package.
        Return value from this function is used to decide on function optimality.
        """
        for index,name in optimal_params.items():
            value = inputs[index]
            if hyperparam_types[name] is int:
                value = (int)(inputs[index]+0.5)
            default_params[name] = value

        prim_instance = self.primitive(hyperparams=default_params)

        import random
        random.seed(9001)

        # Run training on 90% and testing on 10% random split of the dataset.
        seq = [i for i in range(len(train))]
        random.shuffle(seq)

        testsize = (int)(0.1 * len(train) + 0.5)

        trainindices = [seq[x] for x in range(len(train)-testsize)]
        testindices = [seq[x] for x in range(len(train)-testsize, len(train))]
        Xtrain = train.iloc[trainindices]
        Ytrain = output.iloc[trainindices]
        Xtest = train.iloc[testindices]
        Ytest = output.iloc[testindices]

        prim_instance.set_training_data(inputs=Xtrain.values, outputs=Ytrain.values)
        prim_instance.fit()
        predictions = prim_instance.produce(inputs=Xtest).value

        metric = self.evaluate_metric(predictions, Ytest)
        print('Metric: %f' %(metric))
        return metric

    def optimize_hyperparams(self, train, output, lower_bounds, upper_bounds, default_params, hyperparam_types):
        """
        Optimize primitive's hyper parameters using Bayesian Optimization package 'bo'.
        Optimization is done for the parameters with specified range(lower - upper).
        """
        import bo
        import bo.gp_call

        domain_bounds = []
        optimal_params = {}
        index = 0

        # Create parameter ranges in domain_bounds. 
        # Map parameter names to indices in optimal_params
        for name,value in lower_bounds.items():
            lower = lower_bounds[name]
            upper = upper_bounds[name]
            domain_bounds.append([lower,upper])
            optimal_params[index] = name
            index =index+1

        func = lambda inputs : self.optimize_primitive(train, output, inputs, default_params, optimal_params, hyperparam_types)
        try:
            (curr_opt_val, curr_opt_pt) = bo.gp_call.fmax(func, domain_bounds, 10)
        except:
            print(sys.exc_info()[0])
            curr_opt_val = 0.0
            curr_opt_pt = []
            print(optimal_params)
            print(lower_bounds)
            for i in range(len(optimal_params)):
                curr_opt_pt.append(lower_bounds[optimal_params[i]])

        # Map optimal parameter values found
        for index,name in optimal_params.items():
            value = curr_opt_pt[index]
            if hyperparam_types[name] is int:
                value = (int)(curr_opt_pt[index]+0.5)
            default_params[name] = value

        return default_params

    def find_optimal_hyperparams(self, train, output):
        filter_hyperparam = lambda vl: None if vl == 'None' else vl
        default_hyperparams = {name:filter_hyperparam(vl['default']) for name,vl in self.hyperparam_spec.items()}
        hyperparam_lower_ranges = {name:filter_hyperparam(vl['lower']) for name,vl in self.hyperparam_spec.items() if 'lower' in vl.keys()}
        hyperparam_upper_ranges = {name:filter_hyperparam(vl['upper']) for name,vl in self.hyperparam_spec.items() if 'upper' in vl.keys()}
        hyperparam_types = {name:filter_hyperparam(vl['structural_type']) for name,vl in self.hyperparam_spec.items() if 'structural_type' in vl.keys()}
        print(default_hyperparams)
        if len(hyperparam_lower_ranges) > 0:
            print(hyperparam_lower_ranges)
            print(hyperparam_upper_ranges)
            print(hyperparam_types)
            self.optimize_hyperparams(train, output, hyperparam_lower_ranges, hyperparam_upper_ranges, default_hyperparams, hyperparam_types)
            print(default_hyperparams)

        return default_hyperparams
