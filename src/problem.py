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

from  api_v1 import core


from urllib import request as url_request
from urllib import parse as url_parse

class ProblemDescription(object):
    """
    Basically a PipelineCreateRequest; it describes a problem to solve.
    Each Pipeline object is then a possible solution for solving this problem.
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
        # TODO
        # self._task_subtype = ""
        # Currently undefined, sigh
        # self._output_type
        self._evaluation_metrics = metrics
        self._target_features = target_features
        self._predict_features = predict_features
        self._output_dir = output_dir


    def find_solutions(self):
        """
        First pass at just simply find primitives that match the given problem type.
        """
        logging.info("Listing prims")
        prims = [primitive_lib.Primitive(p) for p in primitive_lib.list_primitives()]
        task_name = core_pb2.TaskType.Name(self._task_type)
        valid_prims = [
            p for p in prims
            if p._metadata.query()['primitive_family'] == task_name
        ]
        prims = d3m.index.search()
        for p in valid_prims:
            name = p._metadata.query()['name']
            path = p._metadata.query()['python_path']
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
                # aaaaaaa why are Python lambdas so shitty
                # also JSON can't list 'None' properly apparently.
                filter_hyperparam = lambda vl: None if vl == 'None' else vl
                default_hyperparams = {name:filter_hyperparam(vl['default']) for name,vl in hyperparam_spec.items()}
                prim_instance = prim(hyperparams=default_hyperparams)


                print("Hyperparams:", default_hyperparams)
                # print("Params:", prim_instance.get_params())

                pipe = PipelineDescription(p._metadata, prim_instance, default_hyperparams)
                # Here we are with our d3m.primitive_interfaces.PrimitiveBase
                # Now we have to shove the training data into it...
                # import numpy as np
                # # inputs = self._dataset_uri
                # inputs = np.zeros((10, 10))

                yield pipe

                break
            else:
                # install_primitive(p._metadata)
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


    def train(self, dataset_spec_uri):
        """
        Trains the model.
        """
        # set_training_data() apparently sometimes does not take an outputs argument
        # whyyyyyyyyyy is this even an option
        # For non-parametric models? I dunno
        # BUT there is NO gorram way to tell what kind of "outputs" arg it needs; it's an ndarray.
        # Gee how descriptive, how BIG does it have to be?
        # Why does it not return a bloody value?

        # This gets all circular with DatasetSpec stuff in here, but forget it.

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

        # input_spec = self._metadata.query()['primitive_code']['instance_methods']['set_params']
        # outputs = "file:///home/sheath/tmp/output"
        import numpy as np
        # We have no good way of doign multiple datasets so we just grab the first one
        (resource_name, train_data) = next(iter(self.datasets.items()))
        logging.info(resource_name)

        # Okay, you know what?  We're going to throw out any data that isn't numeric.
        resource_spec = self.resource_specs[resource_name]
        valid_column_names = [
            column.col_name for column in resource_spec.columns
            if column.col_type != 'categorical'
        ]
        # logging.info("Resource: %s %s", resource_spec.type, valid_column_names)

        # I guess turn it from a pandas dataframe into a numpy array since that's
        # what most things expect
        train_data = train_data[valid_column_names].values
        print(train_data)
        outputs = np.zeros(train_data.shape[0])
        self.primitive.set_training_data(inputs=train_data, outputs=outputs)
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

    def evaluate(self, dataset_spec_uri):
        """
        Runs the solution.  Returns two things: a model, and a score.
        """

        # <<<<<<<<<<<<<<<<<< identical to train() because I have 30 minutes
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

        # input_spec = self._metadata.query()['primitive_code']['instance_methods']['set_params']
        # outputs = "file:///home/sheath/tmp/output"
        import numpy as np
        # We have no good way of doign multiple datasets so we just grab the first one
        (resource_name, test_data) = next(iter(self.datasets.items()))
        logging.info(resource_name)

        # Okay, you know what?  We're going to throw out any data that isn't numeric.
        resource_spec = self.resource_specs[resource_name]
        valid_column_names = [
            column.col_name for column in resource_spec.columns
            if column.col_type != 'categorical'
        ]
        logging.info("Resource: %s %s", resource_spec.type, valid_column_names)
        # >>>>>>>>>>>>>>>>>>>>>

        # I guess turn it from a pandas dataframe into a numpy array since that's
        # what most things expect
        test_data = test_data[valid_column_names].values
        print(test_data)
        # outputs = np.zeros(test_data.shape[0])

        assert self.train_result.has_finished
        res = self.primitive.produce(inputs=test_data, timeout=1000.0, iterations=1)
        #print("Result is:", res)
        print("TESTING: Done:", res.has_finished)
        print("Iterations:", res.iterations_done)
        print("Value:", res.value)
        self.eval_result = res


def install_primitive(prim_metadata):
    """
    Very ghetto way of downloading and installing primitives, but...

    if there's a nicer method in the d3m package I can't find it.
    d3m.index is useless 'cause none of the primitives are on `pip`.
    And it has no way to install things anyway.
    ...though once things ARE installed, d3m.index.search() Actually Magically Works.

    This will ask for your gitlab password on the console as necessary,
    which isn't really ideal, but oh well.  Only has to be run once per install.
    """
    import subprocess
    m = prim_metadata.query()
    for inst in m['installation']:
        if inst['type'] == 'PIP':
            print("Can install package", m['name'], "from", inst['package_uri'])
            subprocess.run(['pip', 'install', inst['package_uri']], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            break
        print("Can't install package", m['name'])
