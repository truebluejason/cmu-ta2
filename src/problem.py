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

import d3m.index
import primitive_lib
import logging
import core_pb2
import pandas as pd

from  api_v1 import core

from urllib import request as url_request
from urllib import parse as url_parse

class ProblemDescription(object):
    """
    Basically a PipelineCreateRequest; it describes a problem to solve.
    Each PipelineDescription object is then a possible solution for solving this problem.
    """
    def __init__(self, name, dataset_uri, output_dir, task_type, metrics, target_features, predict_features):
        """
        This just takes core_pb2 types, which are not necessarily actually convenient,
        especially since that API is going to change.
        But, to get it working, I guess that's something.
        """
        self._name = name
        self._dataset_uri = dataset_uri
        self._task_type = task_type
        # TODO: Currently unused.
        # self._task_subtype = ""
        # Currently undefined, sigh
        # self._output_type
        self._evaluation_metrics = metrics
        self._target_features = target_features
        self._predict_features = predict_features
        self._output_dir = output_dir

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
    
    def optimize_primitive(self, train, output, inputs, prim, default_params, optimal_params, hyperparam_types):
        """
        Function to evaluate each input point in the hyper parameter space.
        This is called for every input sample being evaluated by the bayesian optimization package.
        Return value from this function is used to decide on function optimality.
        """
        for index,name in optimal_params.items():
            value = inputs[index]
            if hyperparam_types[name] == 'int':
                value = (int)(inputs[index]+0.5)
            default_params[name] = value

        prim_instance = prim(hyperparams=default_params)

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
        
        accuracy = self.evaluate_metric(predictions, Ytest)
        print('Accuracy: %f' %(accuracy))
        return accuracy 

    def optimize_hyperparams(self, train, output, lower_bounds, upper_bounds, prim, default_params, hyperparam_types):
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
       
        func = lambda inputs : self.optimize_primitive(train, output, inputs, prim, default_params, optimal_params, hyperparam_types)
        (curr_opt_val, curr_opt_pt) = bo.gp_call.fmax(func, domain_bounds, 10)

        # Map optimal parameter values found
        for index,name in optimal_params.items():
            value = curr_opt_pt[index] 
            if hyperparam_types[name] == 'int':
                value = (int)(curr_opt_pt[index]+0.5)
            default_params[name] = value
        
        return default_params
        

    def find_solutions(self, dataset_spec_uri):
        """
        First pass at just simply finding primitives that match the given problem type.
        """
        logging.info("Listing prims")

        prims = [
            primitive_lib.Primitive(p) for p in primitive_lib.list_primitives()
        ]
        # for p in prims:
        #     print(p._metadata.query()['name'])
        # Find which primitives are applicable to the task.
        # 
        # The prim metadata is so partial (doesn't even say what actual data types
        # it can handle, numerical vs. categorical for example) that we're just going
        # to restrict primitives to ones we've vetted and know work for now.
        good_prims = [
            "sklearn.linear_model.logistic.LogisticRegression",
        ]
        task_name = core_pb2.TaskType.Name(self._task_type)
        valid_prims = [
            p for p in prims
            if p._metadata.query()['primitive_family'] == task_name
                #and p._metadata.query()['name'] in good_prims
        ]
        print("Valid prims are", valid_prims)
        prims = d3m.index.search()
        for p in valid_prims:
            name = p._metadata.query()['name']
            path = p._metadata.query()['python_path']
            print("\n")
            print(name)
            print(path)
            if path in prims:
                print(path, "found")
                # p._metadata.pretty_print()
                prim = prims[path]
                # logging.info("Prim %s can accept: %s", path, prim.can_accept(method_name='fit', arguments={}))
                # Ok, prim is a class constructor.  We need to figure out what its
                # hyperparameters are, then pass them to it as a dict(more or less;
                # d3m.metadata.hyperparams.Hyperparams inherits from a dict and
                # says it should be overridden but nothing seems to do that...)
                # The metadata schema says what hyperparameters should be there and
                # what their valid range is.
                hyperparam_spec = p._metadata.query()['primitive_code']['hyperparams']

                if name == 'sklearn.ensemble.weight_boosting.AdaBoostClassifier':
                    continue
                if name == 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier':
                    continue

                filter_hyperparam = lambda vl: None if vl == 'None' else vl
                default_hyperparams = {name:filter_hyperparam(vl['default']) for name,vl in hyperparam_spec.items()}
                hyperparam_lower_ranges = {name:filter_hyperparam(vl['lower']) for name,vl in hyperparam_spec.items() if 'lower' in vl.keys()}
                hyperparam_upper_ranges = {name:filter_hyperparam(vl['upper']) for name,vl in hyperparam_spec.items() if 'upper' in vl.keys()}
                hyperparam_types = {name:filter_hyperparam(vl['structural_type']) for name,vl in hyperparam_spec.items() if 'structural_type' in vl.keys()} 
                pipe = PipelineDescription(None, None, default_hyperparams)
                (train, output) = pipe._load_dataset(dataset_spec_uri)
                print(default_hyperparams)
                if len(hyperparam_lower_ranges) > 0:
                    print(hyperparam_lower_ranges)
                    print(hyperparam_upper_ranges)
                    print(hyperparam_types)
                    self.optimize_hyperparams(train, output, hyperparam_lower_ranges, hyperparam_upper_ranges, prim, default_hyperparams, hyperparam_types)
                    print(default_hyperparams)
                prim_instance = prim(hyperparams=default_hyperparams)
                # Here we are with our d3m.primitive_interfaces.PrimitiveBase
                # that has its hyperparams in it.
                # Now we just return that, and the server will start training
                # and testing it.
                pipe = PipelineDescription(p._metadata, prim_instance, default_hyperparams)
                yield pipe
            else:
                # Uncomment this to try to just install primitives if they aren't installed.
                # Be prepared to enter your gitlab password many times.
                primitive_lib.install_primitive(p._metadata)
                print("Primitive", path, "should be valid but isn't installed")


class PipelineDescription(object):
    """
    A wrapper of a primitive instance and hyperparameters, ready to have inputs
    fed into it.

    The idea is that this can be evaluated, produce a model and performance metrics,
    and the hyperparameter tuning can consume that and choose what to do next.

    Output is fairly basic right now; it writes to a single numpy CSV file with a given name
    based off the results of the primitive (numpy arrays only atm)
    """

    def __init__(self, metadata, prim_instance, hyperparams):
        self._metadata = metadata
        self.primitive = prim_instance
        self.hyperparams = hyperparams
        self.train_result  = "No result yet, call 'train()'"
        self.eval_result  = "No result yet, call 'train()' followed by 'evaluate()'"


    def _load_dataset(self, dataset_spec_uri):
        """Loads a dataset spec URI and does all the annoying
        preprocessing that needs to be done.

        Returns two numpy arrays: (inputs, labels)
        """

        with url_request.urlopen(dataset_spec_uri) as uri:
            res = uri.read()
            # We need to pull the file root path out of the dataset
            # source the TA3 gave us and give it to the DatasetSpec so it
            # knows where to find the actual files
            self.dataset_root = core.dataset_uri_path(dataset_spec_uri)

            self.dataset_spec = core.DatasetSpec.from_json_str(res, self.dataset_root)
            logging.info("Task created, outputting to %s", self.dataset_root)

        self.resource_specs = {
            resource.res_id:resource for resource in self.dataset_spec.resource_specs
        }

        self.datasets = {
            resource.res_id:resource.load() for resource in self.dataset_spec.resource_specs
        }

        import numpy as np
        # We have no good way of doing multiple datasets so we just grab the first one
        (resource_name, train_data) = next(iter(self.datasets.items()))
        logging.info(resource_name)

        # Also filter out NaN's since pandas uses them for missing values.
        # TODO: Right now we replace them with 0, which is not often what we want...
        train_data.fillna(0, inplace=True)

        # Some primitives don't know how to take string type data
        # so we convert categorical to int's
        resource_spec = self.resource_specs[resource_name]
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

    def train(self, dataset_spec_uri):
        """
        Trains the model.
        """

        (inputs, labels) = self._load_dataset(dataset_spec_uri)
        self.primitive.set_training_data(inputs=inputs.values, outputs=labels.values)
        res = self.primitive.fit()
        print("TRAINING Done:", res.has_finished)
        print("Iterations:", res.iterations_done)
        print("Value:", res.value)
        self.train_result = res


    def training_finished(self):
        """
        Yields false until the model is finished training, true otherwise.
        """
        yield self.train_result.has_finished

    def evaluate(self, dataset_spec_uri, output_file, target_features):
        """
        Runs the solution.  Returns two things: a model, and a score.
        """
        (inputs, _labels) = self._load_dataset(dataset_spec_uri)
        assert self.train_result.has_finished
        res = self.primitive.produce(inputs=inputs.values, timeout=1000.0, iterations=1)
        #print("Result is:", res)
        print("TESTING: Done:", res.has_finished)
        print("Iterations:", res.iterations_done)
        print("Value:", res.value)
        
        self.eval_result = res

        # GREAT now we need to actually measure the results.
        # This means loading the source data and pulling things out of 
        # it by index, I guess.
        # print()
        
        #inputs['prediction'] = pd.Series(res.value)
        #print(inputs[['d3mIndex', 'prediction']])
        print(output_file)
        target = target_features[0]
        predictions_df = pd.DataFrame(res.value, columns=[target.feature_name])
        predictions_df.to_csv(output_file)


