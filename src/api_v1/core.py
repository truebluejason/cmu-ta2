"""
Implementation of the ta2ta3 API v1 (preprocessing extensions) -- core.proto
"""



import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc
import logging
import primitive_lib
import json
import os
import os.path
import pandas as pd
from urllib import request as url_request
from urllib import parse as url_parse

import problem
import util


logging.basicConfig(level=logging.INFO)

__symbol_idx = 0
def gensym(id="gensym"):
    global __symbol_idx
    s = "{}_{}".format(id, __symbol_idx)
    __symbol_idx += 1
    return s

def dataset_uri_path(dataset_uri):
    """
    Takes the dataset spec file:// URI passed as a dataset and returns a file 
    path to the directory containing it.
    """
    parsed_url = url_parse.urlparse(dataset_uri)
    assert parsed_url.scheme == 'file'
    dataset_path = parsed_url.path
    # Find the last / and chop off any file after it
    filename_start_loc = dataset_path.rfind('/')
    assert filename_start_loc > 0
    return dataset_path[:filename_start_loc]


# class TaskClassification(object):
#     "Simple random classifier that just does the data loading etc."
#     def __init__(self, dataset_uri, target_features, predict_features):
#         with url_request.urlopen(dataset_uri) as uri:
#             res = uri.read()
#             # We need to pull the file root path out of the dataset
#             # source the TA3 gave us and give it to the DatasetSpec so it
#             # knows where to find the actual files
#             self.dataset_root = dataset_uri_path(dataset_uri)

#             self.dataset_spec = DatasetSpec.from_json_str(res, self.dataset_root)
#             logging.info("Task created, outputting to %s", self.dataset_root)

#         self.resource_specs = {
#             resource.res_id:resource for resource in self.dataset_spec.resource_specs
#         }

#         self.target_features = target_features
#         self.predict_features = predict_features

#         self.datasets = {
#             resource.res_id:resource.load() for resource in self.dataset_spec.resource_specs
#         }

#         # Resolve our (resource_name, feature_name) pairs
#         # to (resource_spec, feature_name)
#         # [
#         #     (self.data_resources[feature.resource_id],
#         #      feature.feature_name)
#         #     for feature in predict_features
#         # ]

#     def to_json_dict(self):
#         return self.dataset_spec.to_json_dict()

#     def run(self, output_path):
#         """
#         Outputs a CSV file relative to self.dataset_root
#         """
#         import random
#         import os
#         output_subdir = os.path.join(self.dataset_root, "output")
#         os.makedirs(output_subdir, exist_ok=True)
#         for target in self.target_features:
#             target_column = self.datasets[target.resource_id][target.feature_name]
#             # possible_values = list(set(target_column.values))
#             predictions = random.choices(target_column.values, k=len(target_column))
#             predictions_df = pd.DataFrame(
#                 predictions,
#                 columns=[target.feature_name]
#             )
#             predictions_df.to_csv(os.path.join(output_subdir, output_path))
#             break


class Column(object):
    """
    Metadata for a data column in a resource specification file.
    """
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

class DataResource(object):
    """
    Metadata for a resource in a resource specification file.
    """
    def __init__(self, res_id, path, type, format, is_collection, columns, dataset_root):
        self.res_id = res_id
        self.path = path
        self.type = type
        self.format = format
        self.is_collection = is_collection
        self.columns = columns
        self.full_path = os.path.join(dataset_root, path)

    def load(self):
        """
        Loads the dataset and returns a Pandas dataframe.
        """
        # Just take the first index column, assuming there's only one.
        index_col = next((c for c in self.columns if c.role == "index"), None)
        df = pd.read_csv(self.full_path, index_col=index_col)
        return df

    @staticmethod
    def from_json_dict(json_dct, dataset_root):
        columns = [
            Column.from_json_dict(dct) for dct in json_dct['columns']
        ]
        return DataResource(
            json_dct['resID'], 
            json_dct['resPath'], 
            json_dct['resType'], 
            json_dct['resFormat'],
            json_dct['isCollection'],
            columns,
            dataset_root)

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
    """
    Parser/shortcut methods for a dataset specification file.

    It is a JSON file that contains one or more DataResource's, each of which
    contains one or more Column's.
    Basically it is a collection of metadata about a dataset, including where to find
    the actual data files (relative to its own location).
    The TA3 client will pass a URI to a dataset spec file as a way of saying which
    dataset it wants to operate on.
    """
    def __init__(self, about, resources, root_path):
        self.about = about
        self.resource_specs = resources
        self.root_path = root_path

    @staticmethod
    def from_json_dict(json_dct, root_path):
        resources = [
                DataResource.from_json_dict(dct, root_path) for dct in json_dct['dataResources']
            ]
        return DatasetSpec(
            json_dct['about'],
            resources,
            root_path
        )
    @staticmethod
    def from_json_str(json_str, root_path):
        s = json.loads(json_str)
        return DatasetSpec.from_json_dict(s, root_path)

    def to_json_dict(self):
        return {
            'about': self.about,
            'dataResources': list(map(lambda spec: spec.to_json_dict(), self.resource_specs))
        }

class Session(object):
    """
    A single Session contains one or more ProblemDescription's,
    and keeps track of the Pipelines currently being trained to solve
    those problems.
    """
    def __init__(self, name):
        self._name = name
        # Dict of identifier : ProblemDescription pairs
        self._problems = {}

    def new_problem(self, problem_desc):
        """
        Takes a new PipelineSpecification and starts trying
        to solve the problem it presents.
        """
        problem_id = gensym(self._name + "_spec")
        self._problems[problem_id] = problem_desc
        return problem_id


    def get_problem(self, name):
        return self._problems[name]

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
        dataset_uri = request.dataset_uri
        task_type = request.task
        # TODO: task_subtype is currently ignored.
        # TODO: task_description is currently ignored.
        metrics = request.metrics
        target_features = request.target_features
        predict_features = request.predict_features

        # We need to tell the TA1 where it can find output,
        # which will be in a created subdirectory of the dataset URI
        # This assumes the URI is always file:// but changing that will
        # be Hard so it's an ok assumption for now.
        dataset_directory = dataset_uri_path(request.dataset_uri)
        output_directory = os.path.join(dataset_directory, "output")

        # We describe a set of related pipelines with a ProblemDescription.
        # This basically is all the parameters for the problem that the client
        # wants us to solve, with the idea that it will do whatever metalearning
        # stuff it wants and then produce a set of PipelineDescription,
        # where each Pipeline is a particular attempt at solving that problem.

        spec = problem.ProblemDescription(session_id, dataset_uri, output_directory, task_type, metrics, target_features, predict_features)

        logging.info("Starting new problem for session %s", session_id)
        problem_id = session.new_problem(spec)

        pipelines = spec.find_solutions(dataset_uri)
        for pipeline in pipelines:
            pipeline_id = self._new_pipeline_id()
            output_file = pipeline_id + ".csv"
            output_uri = "file://" + output_directory + "/" + output_file
            pb2_pipeline = core_pb2.Pipeline(
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
                pipeline_info=pb2_pipeline
            )
            yield msg
    
            # Actually ask the problem description to start finding solutions.
            msg.progress_info = core_pb2.RUNNING

            pipeline.train(dataset_uri)
            yield msg
            pipeline.evaluate(dataset_uri, os.path.join(output_directory, output_file), target_features)
            msg.progress_info = core_pb2.COMPLETED
            yield msg


    def ExecutePipeline(self, request, context):
        "Placeholder."
        logging.info("Message received: ExecutePipelines")
        yield core_pb2.PipelineExecuteResult(
            response_info=core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK),
            ),
            progress_info=core_pb2.COMPLETED,
            pipeline_id=request.pipeline_id,
            result_uri="file://TODO, return valid pipeline URI"
        )
        

    def ListPipelines(self, request, context):
        "Placeholder."
        logging.info("Message received: ListPipelines")
        return core_pb2.PipelineListResult(
            response_info = core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)),
            pipeline_ids = list(self._pipelines)
        )


    def DeletePipelines(self, request, context):
        "Placeholder."
        logging.info("Message received: DeletePipelines")
        return core_pb2.PipelineListResult(
            response_info = core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.OK)),
            pipeline_ids = [
                
            ]
        )

    def GetCreatePipelineResults(self, request, context):
        "Placeholder."
        logging.info("Message received: GetCreatePipelineResults")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def GetExecutePipelineResults(self, request, context):
        "Placeholder."
        logging.info("Message received: GetExecutePipelineResults")
        return core_pb2.Response(core_pb2.Status(code=core_pb2.OK))

    def ExportPipeline(self, request, context):
        "Placeholder."
        logging.info("Message received: ExportPipeline")
        return core_pb2.Response(status=core_pb2.Status(code=core_pb2.OK))

    def UpdateProblemSchema(self, request, context):
        "Placeholder."
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
            user_agent="cmu_ta2 " + util.__version__,
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
            logging.warning("Client tried to end session %s which does not exist", request.session_id)
            return core_pb2.Response(
                status=core_pb2.Status(code=core_pb2.SESSION_UNKNOWN),
            )

def add_to_server(server):
    core_pb2_grpc.add_CoreServicer_to_server(Core(), server)
