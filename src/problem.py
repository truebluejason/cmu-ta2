"""
This should be where everything comes together: Problem descriptions get matched up
with the available primitives, and a plan for how to create a solution gets made.
"""



class ProblemDescription(object):
    """
    Basically a PipelineCreateRequest; it describes a problem to solve.
    Each Pipeline object is then a possible solution for solving this problem.
    """
    def __init__(self, name, dataset_uri, task_type, metrics, target_features, predict_features):
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
