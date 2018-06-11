"""
Implementation of the ta2ta3 API v2 (preprocessing extensions) -- core.proto
"""



import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc
import value_pb2 as value_pb2
import primitive_pb2 as primitive_pb2
import problem_pb2 as problem_pb2
import logging
import primitive_lib
import json
import os
import os.path
import pandas as pd
import pickle, copy
from urllib import request as url_request
from urllib import parse as url_parse

import solutiondescription
#from multiprocessing import Pool, cpu_count
import uuid

logging.basicConfig(level=logging.INFO)

from d3m.metadata.pipeline import Pipeline, PrimitiveStep
from d3m.container.dataset import D3MDatasetLoader, Dataset
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Metadata

def load_problem_doc(problem_doc_uri: str):
    """     Load problem_doc from problem_doc_uri     
    Paramters     ---------     problem_doc_uri
         Uri where the problemDoc.json is located
    """     
    with open(problem_doc_uri) as file:         
        problem_doc = json.load(file)     
    problem_doc_metadata = Metadata(problem_doc)     
    return problem_doc_metadata

def add_target_columns_metadata(dataset: 'Dataset', problem_doc_metadata: 'Metadata'):
    
    for data in problem_doc_metadata.query(())['inputs']['data']:
        targets = data['targets']
        for target in targets:
            semantic_types = list(dataset.metadata.query((target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex'])).get('semantic_types', []))
            if 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
                dataset.metadata = dataset.metadata.update((target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex']), {'semantic_types': semantic_types})
            if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in semantic_types:
                semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
                dataset.metadata = dataset.metadata.update((target['resID'], metadata_base.ALL_ELEMENTS, target['colIndex']), {'semantic_types': semantic_types})

    return dataset

def add_target_metadata(dataset, targets):
    for target in targets:
        semantic_types = list(dataset.metadata.query((target.resource_id, metadata_base.ALL_ELEMENTS, target.column_index)).get('semantic_types', []))
        if 'https://metadata.datadrivendiscovery.org/types/Target' not in semantic_types:
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
            dataset.metadata = dataset.metadata.update((target.resource_id, metadata_base.ALL_ELEMENTS, target.column_index), {'semantic_types': semantic_types})
        if 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in semantic_types:
            semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
            dataset.metadata = dataset.metadata.update((target.resource_id, metadata_base.ALL_ELEMENTS, target.column_index), {'semantic_types': semantic_types})

    return dataset

def generate_pipeline(pipeline_uri: str, dataset_uri: str, problem_doc_uri: str):
    # Pipeline description
    pipeline_description = None
    if '.json' in pipeline_uri:
        with open(pipeline_uri) as pipeline_file:
            pipeline_description = Pipeline.from_json_content(string_or_file=pipeline_file)
    else:
        with open(pipeline_uri) as pipeline_file:
            pipeline_description = Pipeline.from_yaml_content(string_or_file=pipeline_file)

    # Problem Doc
    problem_doc = load_problem_doc(problem_doc_uri)

    # Dataset
    if 'file:' not in dataset_uri:
        dataset_uri = 'file://{dataset_uri}'.format(dataset_uri=os.path.abspath(dataset_uri))
    dataset = D3MDatasetLoader().load(dataset_uri)
    # Adding Metadata to Dataset
    dataset = add_target_columns_metadata(dataset, problem_doc)

    # Pipeline
    solution = solutiondescription.SolutionDescription('REGRESSION')
    solution.create_from_pipeline(pipeline_description)
    # Fitting Pipeline
    solution.fit(inputs=[dataset])
    return solution

def test_pipeline(solution: solutiondescription.SolutionDescription, dataset_uri: str, problem_doc_uri: str):
    # Dataset
    if 'file:' not in dataset_uri:
        dataset_uri = 'file://{dataset_uri}'.format(dataset_uri=os.path.abspath(dataset_uri))
    dataset = D3MDatasetLoader().load(dataset_uri)
    return solution.produce(inputs=[dataset])

class Core(core_pb2_grpc.CoreServicer):
    def __init__(self):
        self._sessions = {}
        self._primitives = {}
        self._solutions = {}
        self._solution_score_map = {}
        self._search_solutions = {}
 
        for p in primitive_lib.list_primitives():
            if p.name == 'sklearn.ensemble.weight_boosting.AdaBoostClassifier':
                continue
            if p.name == 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier':
               continue
            self._primitives[p.classname] = solutiondescription.PrimitiveDescription(p.classname, p)

        pipeline_uri = '185_pipe_v3.json'
        dataset_uri = '185_baseball/185_baseball_dataset/datasetDoc.json'
        problem_doc_uri = '185_baseball/185_baseball_problem/problemDoc.json'
        test_dataset_uri = '185_baseball/TEST/dataset_TEST/datasetDoc.json'
        #solution = generate_pipeline(pipeline_uri=pipeline_uri, dataset_uri=dataset_uri, problem_doc_uri=problem_doc_uri)
        #prodop = test_pipeline(solution, test_dataset_uri, problem_doc_uri)

        #if 'file:' not in dataset_uri:
        #    dataset_uri = 'file://{dataset_uri}'.format(dataset_uri=os.path.abspath(dataset_uri))
        #dataset = D3MDatasetLoader().load(dataset_uri)
        # Adding Metadata to Dataset
        #problem_doc = load_problem_doc(problem_doc_uri)
        #dataset = add_target_columns_metadata(dataset, problem_doc)
        #score = solution.score_solution(inputs=[dataset], metric=problem_pb2.ACCURACY, primitive_dict=self._primitives)
        #score = solution.score_solution(inputs=[dataset], metric=problem_pb2.ROOT_MEAN_SQUARED_ERROR, primitive_dict=self._primitives)
        #print("Score = ", score)
        #print(prodop)
        #print("Creating it now")
        #pb2_pd = solution.describe_solution(self._primitives)
        #pb2_pd.steps[3].primitive.hyperparams = {}
        #pb2_pd.steps[3].primitive.hyperparams['min_samples_split'] = pipeline_pb2.PrimitiveStepHyperparameter(value=5)
        #new_sol = solutiondescription.SolutionDescription()
        #print("Descr done")
        #new_sol.create_from_pipelinedescription(pb2_pd)
        #new_sol.score_solution(inputs=[dataset], metric=problem_pb2.ACCURACY, primitive_dict=self._primitives)

    def search_solutions(self, task):
        request = task[0]['request']
        primitives = task[0]['primitives']
        problem = request.problem.problem
        template = request.template
        task_name = problem_pb2.TaskType.Name(problem.task_type)
        print(task_name)

        solutions = []
        basic_sol = solutiondescription.SolutionDescription(request.problem.problem.task_type.name)
        basic_sol.initialize_solution(task_name)

        for classname, p in primitives.items():
            if p.primitive_class.family == task_name:
                pipe = copy.deepcopy(basic_sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step(p.primitive_class.python_path)
                solutions.append(pipe)

        return solutions
    
    def SearchSolutions(self, request, context):
        logging.info("Message received: SearchSolutions")
        search_id_str = str(uuid.uuid4())

        self._solution_score_map[search_id_str] = request
        return core_pb2.SearchSolutionsResponse(search_id = search_id_str)

    def _get_inputs(self, solution, rinputs):
        inputs = []
        for ip in rinputs:
            if ip.HasField("dataset_uri") == True:
                dataset = D3MDatasetLoader().load(ip.dataset_uri)
                targets = solution.problem.inputs[0].targets
                dataset = add_target_metadata(dataset, targets)
                inputs.append(dataset)

        return inputs
        
    def GetSearchSolutionsResults(self, request, context):
        logging.info("Message received: GetSearchSolutionsRequest")
        search_id_str = request.search_id
        
        start=solutiondescription.compute_timestamp()
        msg = core_pb2.Progress(state=core_pb2.PENDING, status="", start=start, end=solutiondescription.compute_timestamp())
        yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=0, all_ticks=0, solution_id="",
                     internal_score=0.0, scores=[])

        request_params = self._solution_score_map[search_id_str]
        task = [{'request': request_params, 'primitives': self._primitives}]
        solutions = self.search_solutions(task)

        msg = core_pb2.Progress(state=core_pb2.RUNNING, status="", start=start, end=solutiondescription.compute_timestamp())
        count = 0
        self._search_solutions[search_id_str] = []

        for sol in solutions:
            count = count + 1
            self._solutions[sol.id] = sol
            self._search_solutions[search_id_str].append(sol.id)
            yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=count, all_ticks=len(solutions), solution_id=sol.id,
             internal_score=0.0, scores=[])

        self._solution_score_map.pop(search_id_str, None)
       
        msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp()) 
        yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=len(solutions), all_ticks=len(solutions),
                          solution_id="", internal_score=0.0, scores=[])

    def EndSearchSolutions(self, request, context):
        logging.info("Message received: EndSearchSolutions")
        search_id_str = request.search_id

        for sol_id in self._search_solutions[search_id_str]:
            self._solutions.pop(sol_id, None)

        self._search_solutions[search_id_str].clear()
        return core_pb2.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        search_id_str = request.search_id
        logging.info("Message received: StopSearchSolutions")
        return core_pb2.StopSearchSolutionsResponse()

    def DescribeSolution(self, request, context):
        logging.info("Message received: DescribeSolution")
        solution_id = request.solution_id
        solution = self._solutions[solution_id]
        desc = solution.describe_solution(self._primitives)

        param_map = []
        num_steps = self._solutions[solution_id].num_steps()
        for j in range(num_steps):
            param_map.append(core_pb2.StepDescription(primitive=self._solutions[solution_id].get_hyperparams(j, self._primitives)))

        return core_pb2.DescribeSolutionResponse(pipeline=desc, steps=param_map)

    def ScoreSolution(self, request, context):
        logging.info("Message received: ScoreSolution")

        request_id = str(uuid.uuid4())
        self._solution_score_map[request_id] = request

        return core_pb2.ScoreSolutionResponse(request_id = request_id)

    def GetScoreSolutionResults(self, request, context):
        logging.info("Message received: GetScoreSolutionResults")
        request_id = request.request_id
        request_params = self._solution_score_map[request_id]
        
        start=solutiondescription.compute_timestamp()
        solution_id = request_params.solution_id
        msg = core_pb2.Progress(state=core_pb2.RUNNING, status="", start=start, end=solutiondescription.compute_timestamp())
        
        send_scores = []

        inputs = self._get_inputs(self._solutions[solution_id], request_params.inputs)
        score = self._solutions[solution_id].score_solution(inputs=inputs, metric=request_params.performance_metrics[0], primitive_dict=self._primitives)
        print(score)
        send_scores.append(core_pb2.Score(metric=request_params.performance_metrics[0],
             fold=request_params.configuration.folds, targets=[], value=value_pb2.Value(double=score)))

        yield core_pb2.GetScoreSolutionResultsResponse(progress=msg, scores=[]) 

        # Clean up
        self._solution_score_map.pop(request_id, None)

        msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())
        yield core_pb2.GetScoreSolutionResultsResponse(progress=msg, scores=send_scores)

    def FitSolution(self, request, context):
        logging.info("Message received: FitSolution")
        request_id = str(uuid.uuid4())
        self._solution_score_map[request_id] = request
        return core_pb2.FitSolutionResponse(request_id = request_id)

    def GetFitSolutionResults(self, request, context):
        logging.info("Message received: GetFitSolutionResults")
        request_id = request.request_id
        request_params = self._solution_score_map[request_id]
        start=solutiondescription.compute_timestamp()

        solution_id = request_params.solution_id
        solution = self._solutions[solution_id]

        msg = core_pb2.Progress(state=core_pb2.RUNNING, status="", start=start, end=solutiondescription.compute_timestamp())
            
        fitted_solution = copy.deepcopy(solution)
        fitted_solution.id = str(uuid.uuid4()) 
        self._solutions[fitted_solution.id] = fitted_solution

        inputs = self._get_inputs(self._solutions[solution_id], request_params.inputs)
        fitted_solution.fit(inputs)
        yield core_pb2.GetFitSolutionResultsResponse(progress=msg, steps=[], exposed_outputs=[], fitted_solution_id=fitted_solution.id)

        self._solution_score_map.pop(request_id, None)

        msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())
        yield core_pb2.GetFitSolutionResultsResponse(progress=msg, steps=[], exposed_outputs=[], fitted_solution_id="")

    def ProduceSolution(self, request, context):
        logging.info("Message received: ProduceSolution")
        request_id = str(uuid.uuid4())
        self._solution_score_map[request_id] = request

        return core_pb2.ProduceSolutionResponse(request_id = request_id)

    def GetProduceSolutionResults(self, request, context):
        logging.info("Message received: GetProduceSolutionResults")
        request_id = request.request_id
        request_params = self._solution_score_map[request_id]
        start=solutiondescription.compute_timestamp()

        solution_id = request_params.fitted_solution_id
        solution = self._solutions[solution_id]

        inputs = self._get_inputs(solution, request_params.inputs)
        solution.produce(inputs)
        
        self._solution_score_map.pop(request_id, None)

        msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())
        return core_pb2.GetProduceSolutionResultsResponse(progress=msg, steps=[], exposed_outputs=[])

    def SolutionExport(self, request, context):
        logging.info("Message received: SolutionExport")
        solution_id = request.fitted_solution_id
        rank = request.rank
        solution = self._solutions[solution_id]
        output = open(solution_id+".dump", "wb")
        pickle.dump(solution, output)
        output.close()
        return core_pb2.SolutionExportResponse()

    def UpdateProblem(self, request, context):
        logging.info("Message received: UpdateProblem")

        return core_pb2.UpdateProblemResponse()

    def ListPrimitives(self, request, context):
        logging.info("Message received: ListPrimitives")

        primitives = []
        for p in self._primitives:
            primitives.append(primitive_pb2.Primitive(id=p.id, version=p.version, python_path=p.python_path, name=p.name, digest=None))
        return core_pb2.ListPrimitivesResponse(primitives=primitives)

    def Hello(self, request, context):
        logging.info("Message received: Hello")
        version = core_pb2.DESCRIPTOR.GetOptions().Extensions[
                    core_pb2.protocol_version]
        return core_pb2.HelloResponse(user_agent="cmu_ta2",
        version=version,
        allowed_value_types = [value_pb2.RAW, value_pb2.DATASET_URI, value_pb2.CSV_URI],
        supported_extensions = [])
        
def add_to_server(server):
    core_pb2_grpc.add_CoreServicer_to_server(Core(), server)
