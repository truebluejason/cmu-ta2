"""
Implementation of the ta2ta3 API v1 (preprocessing extensions) -- core.proto
"""

# Dataset solution locations expected by TA3:
# {'dataset_schema': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TRAIN/dataset_TRAIN/datasetDoc.json', 'training_data_root': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TRAIN/dataset_TRAIN', 'problem_schema': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TRAIN/problem_TRAIN/problemDoc.json', 'problem_root': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TRAIN/problem_TRAIN', 'pipeline_logs_root': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/logs', 'executables_root': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/executables', 'results_root': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/results', 'temp_storage_root': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/temp', 'user_problems_root': '/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/user_problems', 'timeout': 60, 'cpus': '16', 'ram': '64Gi'}
# Okay we can just return a file:// URL of predictions, the prediction format 
# being a CSV file loaded into a pandas DF
# Expects a column of target_variable_name


import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc
import logging
import primitive_lib
import json
import os
import pandas as pd
from urllib import request as url_request


logging.basicConfig(level=logging.INFO)

__symbol_idx = 0
def gensym(id="gensym"):
    global __symbol_idx
    s = "{}_{}".format(id, __symbol_idx)
    __symbol_idx += 1
    return s


class TaskClassification(object):
    "Simple random classifier that just does the data loading etc."
    def __init__(self, dataset_uri, target_features, predict_features):
        with url_request.urlopen(dataset_uri) as uri:
            res = uri.read()
            self.dataset_spec = DatasetSpec.from_json_str(res)

        self.resource_specs = {
            resource.res_id:resource for resource in self.dataset_spec.resource_specs
        }

        self.target_features = target_features
        self.predict_features = predict_features

        self.datasets = {
            resource.res_id:resource.load() for resource in self.dataset_spec.resource_specs
        }

        # Resolve our (resource_name, feature_name) pairs
        # to (resource_spec, feature_name)
        # [
        #     (self.data_resources[feature.resource_id],
        #      feature.feature_name)
        #     for feature in predict_features
        # ]

    def to_json_dict(self):
        return self.dataset_spec.to_json_dict()

    def run(self, output_path):
        import random
        for target in self.target_features:
            target_column = self.datasets[target.resource_id][target.feature_name]
            # possible_values = list(set(target_column.values))
            predictions = random.choices(target_column.values, k=len(target_column))
            predictions_df = pd.DataFrame(
                predictions,
                columns=[target.feature_name]
            )
            predictions_df.to_csv(output_path)
            break


class Column(object):
    def __init__(self, colIndex, colName, colType, role):
        self.col_index = colIndex
        self.col_name = colName
        self.col_type  = colType
        self.role = role

    @staticmethod
    def from_json_dict(json_dct):
        return Column(json_dct['colIndex'], json_dct['colName'], json_dct['colType'], json_dct['role'])


    # TODO: Refactor this to a class method, annotation or mixin.
    @staticmethod
    def from_json_str(json_str):
        "Easy way to deserialize JSON to an object, see https://stackoverflow.com/a/16826012"
        return json.loads(json_str, object_hook=Column.from_json_dict)

    def to_json_dict(self):
        return {
            'colIndex':self.col_index,
            'colName':self.col_name,
            'colType':self.col_type,
            'role':self.role,
        }

# TODO: sigh
DATASET_ROOT = "/home/sheath/projects/D3M/cmu-ta3/test-data/185_baseball/TRAIN/dataset_TRAIN/"
OUTPUT_ROOT = "/home/sheath/projects/D3M/cmu-ta2/test_output/"

class DataResource(object):
    def __init__(self, res_id, path, type, format, is_collection, columns):
        self.res_id = res_id
        self.path = path
        self.type = type
        self.format = format
        self.is_collection = is_collection
        self.columns = columns
        self.full_path = os.path.join(DATASET_ROOT, path)

    def load(self):
        """
        Loads the dataset and returns a Pandas dataframe.
        """
        # Just take the first index column, assuming there's only one.
        index_col = next((c for c in self.columns if c.role == "index"), None)
        df = pd.read_csv(self.full_path, index_col=index_col)
        return df

    @staticmethod
    def from_json_dict(json_dct):
        columns = [
            Column.from_json_dict(dct) for dct in json_dct['columns']
        ]
        return DataResource(
            json_dct['resID'], 
            json_dct['resPath'], 
            json_dct['resType'], 
            json_dct['resFormat'],
            json_dct['isCollection'],
            columns)

    @staticmethod
    def from_json_str(json_str):
        return json.loads(json_str, object_hook=DataResource.from_json_dict)

    def to_json_dict(self):
        return {
            'resID': self.res_id,
            'resPath': self.path,
            'resType': self.type,
            'resFormat': self.format,
            'isCollection': self.is_collection,
            'columns': list(map(lambda column: column.to_json_dict(), self.columns))
        }

class DatasetSpec(object):
    def __init__(self, about, resources):
        self.about = about
        self.resource_specs = resources

    @staticmethod
    def from_json_dict(json_dct):
        resources = [
                DataResource.from_json_dict(dct) for dct in json_dct['dataResources']
            ]
        return DatasetSpec(
            json_dct['about'],
            resources
        )
    @staticmethod
    def from_json_str(json_str):
        s = json.loads(json_str)
        return DatasetSpec.from_json_dict(s)

    def to_json_dict(self):
        return {
            'about': self.about,
            'dataResources': list(map(lambda spec: spec.to_json_dict(), self.resource_specs))
        }

class PipelineSpecification(object):
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

class Pipeline(object):
    """
    A single model that is trying to solve a particular problem described
    by the PipelineSpecification
    """
    def __init__(self, name, spec):
        self._name = name
        self._spec = spec

class Session(object):
    """
    A single Session contains one or more PipelineSpecifications,
    and keeps track of the Pipelines currently being trained to solve
    those problems.
    """
    def __init__(self, name):
        self._name = name
        # Dict of pipeline_id : pipeline pairs
        self._pipelines_specifications = {}

    def new_problem(self, pipeline_spec):
        """
        Takes a new PipelineSpecification and starts trying
        to solve the problem it presents.
        """
        specname = gensym(self._name + "_spec")
        self._pipelines_specifications[specname] = pipeline_spec

class Core(core_pb2_grpc.CoreServicer):
    def __init__(self):
        self._sessions = {}

    def _new_session_id(self):
        "Returns an identifier string for a new session."
        return gensym("session")

    def _new_pipeline_id(self):
        "Returns an identifier string for a new pipeline."
        return gensym("pipeline")

    def _response_session_invalid(self, session_id):
        "Returns a message that the given session does not exist"
        pipeline = core_pb2.Pipeline(
            predict_result_uri = "invalid",
            output = core_pb2.OUTPUT_TYPE_UNDEFINED,
            scores = []
        )
        msg = core_pb2.PipelineCreateResult(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.SESSION_UNKNOWN),
            ),
            progress_info=core_pb2.ERRORED,
            pipeline_id="invalid",
            pipeline_info=pipeline
        )
        return msg

    def CreatePipelines(self, request, context):
        logging.info("Message received: CreatePipelines: %s", request)

        session_id = request.context.session_id
        if session_id not in self._sessions:
            logging.warning("Asked to create pipeline for session %s which does not exist", session_id)
            return self._response_session_invalid(session_id)
        session = self._sessions[session_id]

        # Setup pipeline specification
        pipeline_id = self._new_pipeline_id()
        dataset_uri = request.dataset_uri
        task_type = request.task
        # TODO: task_subtype is currently ignored.
        # TODO: task_description is currently ignored.
        metrics = request.metrics
        target_features = request.target_features
        predict_features = request.predict_features
        spec = PipelineSpecification(pipeline_id, dataset_uri, task_type, metrics, target_features, predict_features)
        logging.debug("Starting new problem for session %s", session_id)
        session.new_problem(spec)

        output_file = os.path.join(OUTPUT_ROOT, pipeline_id + ".csv")
        output_uri = "file://" + output_file
        pipeline = core_pb2.Pipeline(
            predict_result_uri = output_uri,
            output = core_pb2.OUTPUT_TYPE_UNDEFINED,
            scores = []
        )
        msg = core_pb2.PipelineCreateResult(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK),
            ),
            progress_info=core_pb2.SUBMITTED,
            pipeline_id=pipeline_id,
            pipeline_info=pipeline
        )
        yield msg

        # Actually do stuff
        msg.progress_info = core_pb2.RUNNING
        yield msg


        classifier = TaskClassification(request.dataset_uri, request.target_features, request.predict_features)
        classifier.run(output_file)
        print(classifier.to_json_dict())

        # Return pipeline results
        msg.progress_info = core_pb2.COMPLETED
        yield msg


    def ExecutePipeline(self, request, context):
        logging.info("Message received: ExecutePipelines")
        yield core_pb2.PipelineExecuteResult(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK),
            ),
            progress_info=core_pb2.COMPLETED,
            pipeline_id=1,
            result_uri=1,
        )
        

    def ListPipelines(self, request, context):
        logging.info("Message received: ListPipelines")
        return core_pb2.PipelineListResult(
            response_info = core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)),
            pipeline_ids = list(self._pipelines)
        )


    def DeletePipelines(self, request, context):
        logging.info("Message received: DeletePipelines")
        return core_pb2.PipelineListResult(
            response_info = core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)),
            pipeline_ids = [
                
            ]
        )

    def GetCreatePipelineResults(self, request, context):
        logging.info("Message received: GetCreatePipelineResults")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))
    def GetExecutePipelineResults(self, request, context):
        logging.info("Message received: GetExecutePipelineResults")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def ExportPipeline(self, request, context):
        logging.info("Message received: ExportPipeline")
        return core_pb2.Response(status=core_pb2.Status(code=core_pb2.OK))

    def UpdateProblemSchema(self, request, context):
        logging.info("Message received: UpdateProblemSchema")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def StartSession(self, request, context):
        logging.info("Message received: StartSession %s", request)
        version = core_pb2.DESCRIPTOR.GetOptions().Extensions[
            core_pb2.protocol_version]
        session_id = self._new_session_id()
        session = Session(session_id)
        self._sessions[session_id] = session
        # TODO: Check duplicates
        # session = "session_%d" % len(self.sessions)
        # self.sessions.add(session)
        logging.info("Session started: %s (protocol version %s)", session_id, version)
        return core_pb2.SessionResponse(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)
            ),
            # Sigh, importing the version string is screwy for no apparent reason
            user_agent="cmu_ta2 0.1",
            version=version,
            context=core_pb2.SessionContext(session_id=session_id),
        )

    def EndSession(self, request, context):
        logging.info("Message received: EndSession")
        if request.session_id in self._sessions.keys():
            _session = self._sessions.pop(request.session_id)
            logging.info("Session terminated: %s", request.session_id)
            return core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK),
            )
        else:
            logging.warn("Client tried to end session %s which does not exist", request.session_id)
            return core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.SESSION_UNKNOWN),
            )

def add_to_server(server):
    core_pb2_grpc.add_CoreServicer_to_server(Core(), server)