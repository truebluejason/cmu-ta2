"""
This should be where everything comes together: Problem descriptions get matched up
with the available primitives, and a plan for how to create a solution gets made.
"""


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
        for p in valid_prims:
            p._metadata.pretty_print()