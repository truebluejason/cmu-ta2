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
from urllib import request as url_request

logging.basicConfig(level=logging.INFO)

__version__ = "0.1.0"

__symbol_idx = 0
def gensym(id="gensym"):
    global __symbol_idx
    s = "{}_{}".format(id, __symbol_idx)
    __symbol_idx += 1
    return s

def run_pipeline_mockup():
    import time
    yield core_pb2.SUBMITTED
    time.sleep(1)
    yield core_pb2.RUNNING
    time.sleep(1)
    yield core_pb2.COMPLETED

class TaskClassification(object):
    "Simple random classifier that just does the data loading etc."
    def __init__(self, dataset_uri, target_features, predict_features):
        with url_request.urlopen(dataset_uri) as uri:
            res = uri.read()
            self.dataset_spec = DatasetSpec.from_json_str(res)

        self.data_resources = {
            resource['resID']:resource for resource in self.dataset_spec.resource_specs
        }

        self.target_features = target_features
        self.predict_features = predict_features

        # Resolve our (resource_name, feature_name) pairs
        # to (resource_spec, feature_name)
        # [
        #     (self.data_resources[feature.resource_id],
        #      feature.feature_name)
        #     for feature in predict_features
        # ]

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
    def staticmethod(json_str):
        "Easy way to deserialize JSON to an object, see https://stackoverflow.com/a/16826012"
        return json.loads(json_str, object_hook=Column.from_json_dict)


class DataResource(object):
    def __init__(self, res_id, path, type, format, is_collection, columns):
        self.res_id = res_id
        self.path = path
        self.type = type
        self.format = format
        self.is_collection = is_collection
        self.columns = columns


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

class Core(core_pb2_grpc.CoreServicer):
    def __init__(self):
        self._sessions = set()
        self._pipelines = set()

    def _new_session_id(self):
        "Returns an identifier string for a new session."
        return gensym("session")

    def _new_pipeline_id(self):
        "Returns an identifier string for a new pipeline."
        return gensym("pipeline")

    def CreatePipelines(self, request, context):
        logging.info("Message received: CreatePipelines: %s", request)

        # Actually do stuff
        pipeline_id = self._new_pipeline_id()
        self._pipelines.add(pipeline_id)

        classifier = TaskClassification(request.dataset_uri, request.target_features, request.predict_features)

        # Return pipeline results
        pipeline = core_pb2.Pipeline(
            predict_result_uri = "test_uri",
            output = core_pb2.OUTPUT_TYPE_UNDEFINED,
            scores = []
        )
        for status in run_pipeline_mockup():
            msg = core_pb2.PipelineCreateResult(
                response_info=core_pb2.Response(
                    status=core_pb2.Status(code=core_pb2.OK),
                ),
                progress_info=status,
                pipeline_id=pipeline_id,
                pipeline_info=pipeline
            )
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
        self._sessions.add(session_id)
        # session = "session_%d" % len(self.sessions)
        # self.sessions.add(session)
        logging.info("Session started: %s (protocol version %s)", session_id, version)
        return core_pb2.SessionResponse(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)
            ),
            user_agent="cmu_ta2 %s" % __version__,
            version=version,
            context=core_pb2.SessionContext(session_id=session_id),
        )

    def EndSession(self, request, context):
        logging.info("Message received: EndSession")
        if request.session_id in self._sessions:
            self._sessions.remove(request.session_id)
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