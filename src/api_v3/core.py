"""
Implementation of the ta2ta3 API v2 (preprocessing extensions) -- core.proto
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
from multiprocessing import Process, Queue


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
        self._primitives = []
        self._search_solutions = {}
        self._process_handles = {}
        self._solutions = {}
        self._queue = Queue()

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
    
    def evaluate_solution(q, i, sol, X, y, mode):
        score = sol.score_solution(X, y, mode)
        q.put((i, score))

    def SearchSolutions(self, request, context):
        logging.info("Message received: SearchSolutions")
        search_id_str = ""

        if len(self._primitives) == 0:
            for p in primitive_lib.list_primitives():
                self._primitives.append(p)

        problem = request.problem
        template = request.template

        solutions = []
        self._search_solutions[search_id_str] = solutions

        for ip in problem.inputs:
            dataset_id = ip.dataset_id
            dataset_directory = dataset_uri_path(dataset_id)
            task_name = problem.task_type
            (X, y) = problem.load_dataset(dataset_directory)

            for p in self._primitives:
                if p._metadata.query()['primitive_family'] == task_name:
                    pipe = problem.SolutionDescription(p._metadata, p.classname, None) 
                    solutions.append(pipe.id)
                    self._solutions[pipe.id] = pipe
                    #ph.start()

        return core_pb2.SearchSolutionsResponse(search_id = search_id_str)

    def GetSearchSolutionsResults(self, request, context):
        logging.info("Message received: GetSearchSolutionsRequest")
        search_id_str = request.search_id

        #process_handles = self._process_handles[search_id_str]
        scores = []
        count = 0
        solutions = self._search_solutions[search_id_str]
        
        for sol_id in solutions:
            #ph.join()
            #solution_instance = self._solutions[sol_id]
            count = count + 1
            yield core_pb2.GetSearchSolutionsResultsResponse(progress=ProgressState.RUNNING, done_ticks=count, all_ticks=len(solutions), solution_id=sol_id,
             internal_score=NaN, scores=scores)

        yield core_pb2.GetSearchSolutionsResultsResponse(progress=ProgressState.COMPLETED, done_ticks=len(solutions), all_ticks=len(solutions),
                          solution_id="", internal_score=NaN, scores=scores)

    def EndSearchSolutions(self, request, context):
        logging.info("Message received: EndSearchSolutions")
        search_id_str = request.search_id
        self._search_solutions[search_id_str].clear()
        return core_pb2.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        search_id_str = request.search_id
        logging.info("Message received: StopSearchSolutions")
        return core_pb2.StopSearchSolutionsResponse()

    def DescribeSolution(self, request, context):
        logging.info("Message received: DescribeSolution")
        solution_id = request.solution_id
        desc = ""
        return core_pb2.DescribeSolutionResponse(desc)

    def ScoreSolution(self, request, context):
        logging.info("Message received: ScoreSolution")
        solution_id = request.solution_id
        configuration = request.configuration

        request_id = uuid.uuid4()
        self._solution_map[request_id] = request
        q = Queue()
        self._solution_score_map[solution_id] = q

        phmap = []
        self._process_handles[request_id] = phmap
        for i in range(len(request.inputs)):
            ip = request.inputs[i]
            dataset_id = ip.dataset_id
            dataset_directory = dataset_uri_path(dataset_id)
            (X, y) = problem.load_dataset(dataset_directory) 

            ph = Process(self.evaluate_solution, args=(q, i, self._solutions[solution_id], X, y,))
            ph.start()
            phmap.append[ph]

        return core_pb2.ScoreSolutionResponse(request_id = request_id)

    def GetScoreSolutionResults(self, request, context):
        logging.info("Message received: GetScoreSolutionResults")
        request_id = request.request_id
        request_params = self._solution_map[request_id]
        scores = {}
        
        process_handles = self._process_handles[request_id]
        for ph in process_handles:
            ph.join()
            yield core_pb2.GetScoreSolutionResultsResponse(progress=ProgressState.RUNNING, scores=[])

        q = self._solution_score_map[solution_id]
        while q.empty() == False:
            (index, score_value) = q.get()
            scores[index] = value

        q.close()
        q.join_thread()

        send_scores = []
        i = 0
        for key in sorted(scores.iterkeys()):
            send_scores.append(Score(metric=request_params.performance_metrics[i], fold=request_params.configuration.folds, targets=[], value=scores[key]))
        
        # Clean up
        del self._solution_score_map[solution_id]
        del self._solution_map[request_id]
        del self._process_handles[request_id]

        yield core_pb2.GetScoreSolutionResultsResponse(progress=ProgressState.COMPLETED, scores=send_scores)

    def FitSolution(self, request, context):
        logging.info("Message received: FitSolution")
        request_id = uuid.uuid4()
        solution_id = request.solution_id
        solution = self._solutions[solution_id]
        inputs = request.inputs
        solution.train(X, y)
        return core_pb2.FitSolutionResponse(request_id = request_id)

    def GetFitSolutionResults(self, request, context):
        logging.info("Message received: GetFitSolutionResults")
        request_id = request.request_id
        solution_id = self._solution_map[request_id]
        solution = self._solutions[solution_id]
        return core_pb2.GetFitSolutionResultsResponse(progress=ProgressState.COMPLETED)

    def ProduceSolution(self, request, context):
        logging.info("Message received: ProduceSolution")
        solution_id = request.solution_id
        solution = self._solutions[solution_id]
        inputs = request.inputs
        solution.produce(X)
        return core_pb2.ProduceSolutionResponse()

    def GetProduceSolutionResults(self, request, context):
        logging.info("Message received: GetProduceSolutionResults")
        request_id = request.request_id
        solution_id = self._solution_map[request_id]
        solution = self._solutions[solution_id]
        return core_pb2.GetProduceSolutionResultsResponse()

    def SolutionExport(self, request, context):
        logging.info("Message received: SolutionExport")
        solution_id = request.solution_id
        rank = request.rank
        solution = self._solutions[solution_id]
        return core_pb2.SolutionExportResponse()

    def UpdateProblem(self, request, context):
        logging.info("Message received: UpdateProblem")
        return core_pb2.UpdateProblemResponse()

    def ListPrimitives(self, request, context):
        logging.info("Message received: ListPrimitives")
        if len(self._primitives) == 0:
            for p in primitive_lib.list_primitives():
                self._primitives.append(p)

        primitives = []
        for p in self._primitives:
            primtives.append(Primitive(id=p.id, version=p.version, python_path=p.python_module, name=p.name, digest=None))
        return core_pb2.ListPrimitivesResponse(primitives)

    def Hello(self, request, context):
        logging.info("Message received: Hello")
        version = core_pb2.DESCRIPTOR.GetOptions().Extensions[
                    core_pb2.protocol_version]
        return core_pb2.HelloResponse(user_agent="cmu_ta2",
        version=version,
        allowed_value_types = [DATASET_URI],
        supported_extensions = [])
        
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
