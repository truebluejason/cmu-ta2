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

class ProblemDescription(object):
    """
    Basically a PipelineCreateRequest; it describes a problem to solve.
    Each Pipeline object is then a possible solution for solving this problem.
    """
    def __init__(self, name, dataset_uri, task_type, metrics, target_features, predict_features):
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
                p._metadata.pretty_print()
                prim = prims[path]
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

                # Here we are with our d3m.primitive_interfaces.PrimitiveBase
                # Now we have to shove the training data into it...
                import numpy as np
                # inputs = self._dataset_uri
                inputs = np.zeros((10, 10))
                # outputs = "file:///home/sheath/tmp/output"
                outputs = np.zeros((10, 26))
                # whyyyyyyyyyy is this even an option
                input_spec = p._metadata.query()['primitive_code']['instance_methods']['set_params']
                prim_instance.set_training_data(inputs=inputs, outputs=outputs)
                run_inputs = np.zeros((10, 10))
                prim_instance.fit()

                res = prim_instance.produce(inputs=run_inputs, timeout=1000.0, iterations=1)
                print("Result is:", res)
            else:
                print("Primitive", path, "should be valid but isn't installed")

class SolutionDescription(object):
    """
    The counterpart to a `Pipeline` without the protocol detail stuff.
    A description

    The idea is that this can be evaluated, produce a model and performance metrics,
    and the hyperparameter tuning can consume that and choose what to do next.
    """

    def __init__(self, problem_desc):
        self._problem = problem_desc

    def evaluate(self):
        """
        Runs the solution.  Returns two things: a model, and a score.
        """


def install_primitive(prim_metadata):
    """
    Very ghetto way of installing primitives, but...

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