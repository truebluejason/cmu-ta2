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
import core_pb2, problem_pb2, pipeline_pb2, primitive_pb2, value_pb2
import pandas as pd

from  api_v3 import core

import uuid, sys, math
from urllib import request as url_request
from urllib import parse as url_parse

import time
from time import sleep
from google.protobuf.timestamp_pb2 import Timestamp

from sklearn import metrics

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

def compute_timestamp():
    now = time.time()
    seconds = int(now)
    return Timestamp(seconds=seconds)

class SolutionDescription(object):
    """
    A wrapper of a primitive instance and hyperparameters, ready to have inputs
    fed into it.

    The idea is that this can be evaluated, produce a model and performance metrics,
    and the hyperparameter tuning can consume that and choose what to do next.

    Output is fairly basic right now; it writes to a single numpy CSV file with a given name
    based off the results of the primitive (numpy arrays only atm)
    """
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.source = None
        self.created = compute_timestamp()
        self.context = pipeline_pb2.PRETRAINING
        self.name = None
        self.description = None
        self.users = None
        self.inputs = []
        self.outputs = []
        self.steps = []

    def add_step(self, prim):
        self.steps.append(prim)

    def add_inputs(self, inputs):
        for ip in inputs:
            self.inputs.append(ip)

    def score_solution(self, X, y, metric):
        prim = self.steps[0]
        score = prim.score_primitive(X, y, metric)
        return score

    def fit_solution(self, inputs):
        for i in range(len(self.steps)):
            p = self.steps[i]
            p.set_training_data()
            p.fit()
            
    def describe_solution(self, prim_dict):
        inputs = []
        inputs.append(pipeline_pb2.PipelineDescriptionInput(name="dataframe inputs"))
        inputs.append(pipeline_pb2.PipelineDescriptionInput(name="dataframe outputs"))

        outputs=[]
        outputs.append(pipeline_pb2.PipelineDescriptionOutput(name="dataframe predictions", data="steps."+str(len(self.steps)-1)+".produce"))

        steps=[]
        for s in self.steps:
            prim = prim_dict[s.primitive]
            p = primitive_pb2.Primitive(id=prim.id, version=prim.version, python_path=prim.python_path, name=prim.name, digest=prim.digest)
            arguments=[]
            i = 0
            for a in prim.arguments:
                arguments.append({a: {"type": "CONTAINER", "data": "inputs."+str(i)}})
                i=i+1
            step_outputs = []
            for a in prim.produce_methods:
                step_outputs.append(pipeline_pb2.StepOutput(id=a))
            steps.append(pipeline_pb2.PipelineDescriptionStep(primitive=pipeline_pb2.PrimitivePipelineDescriptionStep(primitive=p, arguments=arguments, outputs=step_outputs)))
        return pipeline_pb2.PipelineDescription(id=self.id, source=self.source, created=self.created, context=self.context, name=self.name, description=self.description, inputs=inputs, outputs=outputs, steps=steps)

    def num_steps(self):
        return len(self.steps)

    def get_hyperparams(self, step):
        p = self.steps[step]

        if p.hyperparams is not None:
            hyperparams = p.hyperparams
        else:    
            filter_hyperparam = lambda vl: None if vl == 'None' else vl
            hyperparams = {name:filter_hyperparam(vl['default']) for name,vl in p.hyperparam_spec.items()}

        hyperparam_types = {name:filter_hyperparam(vl['structural_type']) for name,vl in p.hyperparam_spec.items() if 'structural_type' in vl.keys()}
        send_params={}
        for name, value in hyperparams.items():
            tp = hyperparam_types[name]
            if tp is int:
               send_params[name]=value_pb2.Value(int64=value)
            elif tp is float:
                send_params[name]=value_pb2.Value(double=value)
            elif tp is bool:
                send_params[name]=value_pb2.Value(bool=value)
            elif tp is str:
                send_params[name]=value_pb2.Value(string=value)
            else:
                if isinstance(value, int):
                    send_params[name]=value_pb2.Value(int64=value)
                elif isinstance(value, float):
                    send_params[name]=value_pb2.Value(double=value)
                elif isinstance(value, bool):
                    send_params[name]=value_pb2.Value(bool=value)
                elif isinstance(value, str):
                    send_params[name]=value_pb2.Value(string=value)
            
        return core_pb2.PrimitiveStepDescription(hyperparams=send_params)


class PrimitiveDescription(object):
    def __init__(self, id, hyperparam_spec, primitive):
        self.id = id
        self.hyperparam_spec = hyperparam_spec
        self.primitive = primitive
        self.hyperparams = None
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

    def score_primitive(self, X, y, metric):
        """
        Learns optimal hyperparameters for the primitive
        Evaluates model on inputs X and outputs y
        Returns metric.
        """
        optimal_params = self.find_optimal_hyperparams(train=X, output=y, metric=metric) 
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
                metric = self.evaluate_metric(predictions, y_test, metric)     
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

    def evaluate_metric(self, predictions, Ytest, metric):
        """
        Function to compute prediction accuracy for classifiers.
        """
        count = len(Ytest)
       
        if metric is problem_pb2.ACCURACY:
             return metrics.accuracy_score(Ytest, predictions)
        elif metric is problem_pb2.PRECISION:
             return metrics.precision_score(Ytest, predictions)
        elif metric is problem_pb2.RECALL:
             return metrics.recall_score(Ytest, predictions)
        elif metric is problem_pb2.F1:
            return metrics.f1_score(Ytest, predictions)
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

    def optimize_primitive(self, train, output, inputs, default_params, optimal_params, hyperparam_types, metric):
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

        metric = self.evaluate_metric(predictions, Ytest, metric)
        print('Metric: %f' %(metric))
        return metric

    def optimize_hyperparams(self, train, output, lower_bounds, upper_bounds, default_params, hyperparam_types, metric):
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

        func = lambda inputs : self.optimize_primitive(train, output, inputs, default_params, optimal_params, hyperparam_types, metric)
        try:
            (curr_opt_val, curr_opt_pt) = bo.gp_call.fmax(func, domain_bounds, 10)
        except:
            print(sys.exc_info()[0])
            curr_opt_val = 0.0
            curr_opt_pt = []
            for i in range(len(optimal_params)):
                curr_opt_pt.append(lower_bounds[optimal_params[i]])

        # Map optimal parameter values found
        for index,name in optimal_params.items():
            value = curr_opt_pt[index]
            if hyperparam_types[name] is int:
                value = (int)(curr_opt_pt[index]+0.5)
            default_params[name] = value

        return default_params

    def find_optimal_hyperparams(self, train, output, metric):
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
            default_hyperparams = self.optimize_hyperparams(train, output, hyperparam_lower_ranges, hyperparam_upper_ranges, default_hyperparams, hyperparam_types, metric)
            print(default_hyperparams)

        return default_hyperparams
