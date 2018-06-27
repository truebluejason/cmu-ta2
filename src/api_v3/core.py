"""
Implementation of the ta2ta3 API v2 (preprocessing extensions) -- core.proto
"""

import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc
import value_pb2 as value_pb2
import primitive_pb2 as primitive_pb2
import problem_pb2 as problem_pb2
import pipeline_pb2 as pipeline_pb2
import logging
import primitive_lib
import os, sys
import os.path
import pandas as pd
import numpy as np
import pickle, copy
from urllib import request as url_request
from urllib import parse as url_parse

import solutiondescription, util
from multiprocessing import Pool, cpu_count
import uuid

logging.basicConfig(level=logging.INFO)
pd.set_option('display.max_rows', None)

from d3m.container.dataset import D3MDatasetLoader, Dataset

def load_primitives():
    primitives = {}
    for p in primitive_lib.list_primitives():
        #if p.name == 'sklearn.ensemble.weight_boosting.AdaBoostClassifier':
        #    continue
        if p.name == 'common_primitives.BayesianLogisticRegression':
            continue
        if p.python_path == 'd3m.primitives.sklearn_wrap.SKGradientBoostingClassifier':
            continue
        if p.python_path == 'd3m.primitives.common_primitives.ConvolutionalNeuralNet':
            continue
        if p.python_path == 'd3m.primitives.common_primitives.RandomForestClassifier':
            continue
        if p.python_path == 'd3m.primitives.common_primitives.FeedForwardNeuralNet':
            continue
        if p.python_path == 'd3m.primitives.common_primitives.Loss':
            continue
        primitives[p.classname] = solutiondescription.PrimitiveDescription(p.classname, p)

    return primitives

def search_phase():
    """
    TA2 running in stand-alone search phase
    """
    inputDir = os.environ['D3MINPUTDIR']
    outputDir = os.environ['D3MOUTPUTDIR']
    timeout_env = os.environ['D3MTIMEOUT']
    num_cpus = os.environ['D3MCPU']

    logging.info("D3MINPUTDIR = %s", inputDir)
    logging.info("D3MOUTPUTDIR = %s", outputDir)
    logging.info("timeout = %s", timeout_env)
    logging.info("cpus = %s", num_cpus)
    config_file = inputDir + "/search_config.json"
    (dataset, task_name, target, timeout_in_min) = util.load_schema(config_file)

    timeout_in_min = (int)(timeout_env)
    primitives = load_primitives()
    task_name = task_name.upper()
    logging.info(task_name)

    solutions = []

    basic_sol = solutiondescription.SolutionDescription(None)
    basic_sol.initialize_solution(task_name)

    if task_name == 'CLASSIFICATION' or task_name == 'REGRESSION':
        for classname, p in primitives.items():
            if p.primitive_class.family == task_name:
                if 'd3m.primitives.sri.' in p.primitive_class.python_path:
                    continue
                pipe = copy.deepcopy(basic_sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step(p.primitive_class.python_path)
                solutions.append(pipe)
    elif task_name == 'COLLABORATIVEFILTERING' or task_name == 'VERTEXNOMINATION':
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)
    else:
        logging.info("No matching solutions")
        return solutions

    async_message_thread = Pool((int)(num_cpus))
    valid_solutions = {}
    valid_solution_scores = {}

    inputs = []
    inputs.append(dataset) 
    if task_name == 'CLASSIFICATION':
        metric= problem_pb2.ACCURACY
    else:
        metric= problem_pb2.MEAN_SQUARED_ERROR
    results = [async_message_thread.apply_async(evaluate_solution_score, (inputs, sol, primitives, metric,)) for sol in solutions]
    timeout = timeout_in_min * 60
    halftimeout = None
    if timeout <= 0:
        timeout = None
    elif timeout > 60:
        timeout = timeout - 60
        halftimeout = timeout/2

    index = 0
    for r in results:
        try:
            score = r.get(timeout=halftimeout)
            if score >= 0.0:
                id = solutions[index].id
                valid_solutions[id] = solutions[index]
                valid_solution_scores[id] = score
        except:
            logging.info(solutions[index].primitives)
            logging.info(sys.exc_info()[0])
            logging.info("Solution terminated: %s", solutions[index].id)

        index = index + 1

    import operator
    sorted_x = sorted(valid_solution_scores.items(), key=operator.itemgetter(1))
    if metric == problem_pb2.ACCURACY:
        sorted_x.reverse()

    index = 1
    for (sol, score) in sorted_x:
        valid_solutions[sol].rank = index
        index = index + 1

    num = 20
    if len(sorted_x) < 20:
        num = len(sorted_x)

    sorted_x = sorted_x[:num]
    results = [async_message_thread.apply_async(fit_solution, (inputs, valid_solutions[sol], primitives, outputDir,))
     for (sol,score) in sorted_x]

    index = 0
    for r in results:
        try:
            valid=r.get(timeout=halftimeout)
        except:
            logging.info(valid_solutions[sorted_x[index][0]].primitives)
            logging.info(sys.exc_info()[0])
            logging.info("Solution terminated: %s", valid_solutions[sorted_x[index][0]].id)
        index = index + 1

def test_phase():
    """
    TA2 running in stand-alone test phase
    """
    inputDir = os.environ['D3MINPUTDIR']
    outputDir = os.environ['D3MOUTPUTDIR']
    executable = os.environ['D3MTESTOPT']

    config_file = inputDir + "/test_config.json"
    (dataset, task_name, target, timeout_in_min) = util.load_schema(config_file)

    primitives = load_primitives()
    task_name = task_name.upper()
    logging.info(task_name)

    inputs = []
    inputs.append(dataset)

    import ntpath
    pipeline_name = ntpath.basename(executable).split(".")[0]
    solution = util.get_pipeline(outputDir + "/supporting_files", pipeline_name)
    predictions = solution.produce(inputs=inputs)[0]
    if isinstance(predictions, np.ndarray):
        predictions = pd.DataFrame(data=predictions)
    if solution.indices is not None:
        predictions = pd.DataFrame({'d3mIndex': solution.indices['d3mIndex'], target:predictions.iloc[:,0]})
    util.write_predictions(predictions, outputDir + "/predictions", solution)
    

def evaluate_solution_score(inputs, solution, primitives, metric):
    """
    Validate each potential solution
    Runs in a separate process
    """
    logging.info("Evaluating %s", solution.id)

    score = solution.score_solution(inputs=inputs, metric=metric,
                                primitive_dict=primitives, solution_dict=None)

    return score

def fit_solution(inputs, solution, primitives, outputDir):
    """
    Validate each potential solution
    Runs in a separate process
    """
    logging.info("Fitting %s", solution.id)
    solution.fit(inputs=inputs, solution_dict=None)

    util.write_solution(solution, outputDir + "/supporting_files")
    util.write_pipeline_json(solution, primitives, outputDir + "/pipelines")
    util.write_pipeline_executable(solution, outputDir + "/executables")
    return True
 
def evaluate_solution(inputs, solution, solution_dict):
    """
    Validate each potential solution
    Runs in a separate process
    """
    logging.info("Evaluating %s", solution.id)
    score = -1

    try:
        valid = solution.validate_solution(inputs=inputs, solution_dict=solution_dict)
        if valid == True:
            return 0
        else:
            return -1
    except:
        logging.info("evaluate_solution exception: %s", sys.exc_info()[0])
        return -1

    return valid

class Core(core_pb2_grpc.CoreServicer):
    def __init__(self):
        self._sessions = {}
        self._primitives = {}
        self._solutions = {}
        self._solution_score_map = {}
        self._search_solutions = {}
        self.async_message_thread = Pool(cpu_count()) #pool.ThreadPool(processes=1,)
        self._primitives = load_primitives()         

        if 0:
            pipeline_uri = '185_pipe_v3.json'
            #pipeline_uri = '185_withsub.json'
            sub_pipeline_uri = '185_sub.json'
            dataset_uri = '185_baseball/185_baseball_dataset/datasetDoc.json'
            problem_doc_uri = '185_baseball/185_baseball_problem/problemDoc.json'
            test_dataset_uri = '185_baseball/TEST/dataset_TEST/datasetDoc.json'
            solution = util.generate_pipeline(pipeline_uri=pipeline_uri, dataset_uri=dataset_uri, problem_doc_uri=problem_doc_uri)
            #subsolution = util.generate_pipeline(pipeline_uri=sub_pipeline_uri, dataset_uri=dataset_uri, problem_doc_uri=problem_doc_uri)
            #self._solutions[solution.id] = solution
            #self._solutions[subsolution.id] = subsolution

            if 'file:' not in dataset_uri:
                dataset_uri = 'file://{dataset_uri}'.format(dataset_uri=os.path.abspath(dataset_uri))
            dataset = D3MDatasetLoader().load(dataset_uri)
            problem_doc = util.load_problem_doc(problem_doc_uri)
            dataset = util.add_target_columns_metadata(dataset, problem_doc)
            solution.fit(inputs=[dataset], solution_dict=self._solutions)

            if 'file:' not in test_dataset_uri:
                test_dataset_uri = 'file://{dataset_uri}'.format(dataset_uri=os.path.abspath(test_dataset_uri))
            test_dataset = D3MDatasetLoader().load(dataset_uri)
            prodop = solution.produce(inputs=[test_dataset], solution_dict=self._solutions)

    def search_solutions(self, request):
        primitives = self._primitives
        problem = request.problem.problem
        template = request.template
        task_name = problem_pb2.TaskType.Name(problem.task_type)
        logging.info(task_name)

        solutions = []
        if task_name != 'CLASSIFICATION' and task_name != 'REGRESSION':
            logging.info("No matching solutions")
            return solutions

        basic_sol = solutiondescription.SolutionDescription(request.problem)
        if bool(template) and isinstance(template, pipeline_pb2.PipelineDescription) and len(template.steps) > 0:
            print("template:", template)
            basic_sol.create_from_pipelinedescription(pipeline_description=template)
        else:
            template = None

        if bool(template) == False or (basic_sol.num_steps() == 1 and basic_sol.contains_placeholder() == True):
            basic_sol.initialize_solution(task_name)

        if bool(template) == False or basic_sol.contains_placeholder() == True:
            if task_name == 'CLASSIFICATION' or task_name == 'REGRESSION':
                for classname, p in primitives.items():
                    if p.primitive_class.family == task_name:
                        if 'd3m.primitives.sri.' in p.primitive_class.python_path:
                            continue
                        pipe = copy.deepcopy(basic_sol)
                        pipe.id = str(uuid.uuid4())
                        pipe.add_step(p.primitive_class.python_path)
                        solutions.append(pipe)
                    elif task_name == 'COLLABORATIVEFILTERING' or task_name == 'VERTEXNOMINATION':
                        pipe = copy.deepcopy(basic_sol)
                        pipe.id = str(uuid.uuid4())
                        pipe.add_outputs()
                        solutions.append(pipe)

        # Fully defined
        if bool(template) == True and basic_sol.contains_placeholder() == False:
            solutions.append(basic_sol)

        return solutions
    
    def SearchSolutions(self, request, context):
        logging.info("Message received: SearchSolutions")
        search_id_str = str(uuid.uuid4())

        self._solution_score_map[search_id_str] = request
        return core_pb2.SearchSolutionsResponse(search_id = search_id_str)

    def _get_inputs(self, problem, rinputs):
        inputs = []
        for ip in rinputs:
            if ip.HasField("dataset_uri") == True:
                dataset = D3MDatasetLoader().load(ip.dataset_uri)
                targets = problem.inputs[0].targets
                dataset = util.add_target_metadata(dataset, targets)
                inputs.append(dataset)
            elif ip.Hasfield("csv_uri") == True:
                dataset = pd.read_csv(ip.csv_uri)
                targets = problem.inputs[0].targets
                #dataset = add_target_metadata(dataset, targets)
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
        solutions = self.search_solutions(request_params)

        inputs = self._get_inputs(request_params.problem, request_params.inputs)

        count = 0
        index = 0
        self._search_solutions[search_id_str] = []
        msg = core_pb2.Progress(state=core_pb2.RUNNING, status="", start=start, end=solutiondescription.compute_timestamp())

        results = [self.async_message_thread.apply_async(evaluate_solution, (inputs, sol, None,)) for sol in solutions]
        timeout = request_params.time_bound * 60
        if timeout <= 0:
            timeout = None
        elif timeout > 60:
            timeout = timeout - 60

        for r in results:
            try:
                val = r.get(timeout=timeout)
                if val == 0:
                    count = count + 1
                    id = solutions[index].id
                    self._solutions[id] = solutions[index]
                    self._search_solutions[search_id_str].append(id)
                    yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=count, all_ticks=len(solutions), solution_id=id,
                                        internal_score=0.0, scores=[])
            except:
                logging.info(solutions[index].primitives)
                logging.info(sys.exc_info()[0])
                logging.info("Solution terminated: %s", solutions[index].id)

            index = index + 1

        self._solution_score_map.pop(search_id_str, None)
       
        msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp()) 
        yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=count, all_ticks=count,
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

        inputs = self._get_inputs(self._solutions[solution_id].problem, request_params.inputs)
        try:
            score = self._solutions[solution_id].score_solution(inputs=inputs, metric=request_params.performance_metrics[0].metric,
                                primitive_dict=self._primitives, solution_dict=self._solutions)
        except:
            score = 0.0

        logging.info("Score = %f", score)
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

        inputs = self._get_inputs(self._solutions[solution_id].problem, request_params.inputs)
        try:
            output = fitted_solution.fit(inputs=inputs, solution_dict=self._solutions)
        except:
            output = None

        result = None
        outputDir = os.environ['D3MOUTPUTDIR']

        if isinstance(output, np.ndarray):
            output = pd.DataFrame(data=output)

        target = self._solutions[solution_id].problem.inputs[0].targets[0].column_name

        predictions = pd.DataFrame({'d3mIndex': fitted_solution.indices['d3mIndex'], target:output.iloc[:,0]})
        uri = util.write_TA3_predictions(predictions, outputDir + "/predictions", fitted_solution, 'fit') 
        uri = 'file://{uri}'.format(uri=os.path.abspath(uri)) 
        result = value_pb2.Value(csv_uri=uri)

        yield core_pb2.GetFitSolutionResultsResponse(progress=msg, steps=[], exposed_outputs=[], fitted_solution_id=fitted_solution.id)

        self._solution_score_map.pop(request_id, None)

        msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())

        steps = []
        for i in range(fitted_solution.num_steps()):
            steps.append(core_pb2.StepProgress(progress=msg))

        exposed_outputs = {}
        last_step_output = request_params.expose_outputs[len(request_params.expose_outputs)-1]
        exposed_outputs[last_step_output] = result

        yield core_pb2.GetFitSolutionResultsResponse(progress=msg, steps=steps, exposed_outputs=exposed_outputs, fitted_solution_id=fitted_solution.id)

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

        inputs = self._get_inputs(solution.problem, request_params.inputs)
        try:
            output = solution.produce(inputs=inputs, solution_dict=self._solutions)[0]
        except:
            output = None
    
        result = None
        
        outputDir = os.environ['D3MOUTPUTDIR']
        if isinstance(output, np.ndarray):
            output = pd.DataFrame(data=output)

        target = solution.problem.inputs[0].targets[0].column_name
        predictions = pd.DataFrame({'d3mIndex': solution.indices['d3mIndex'], target:output.iloc[:,0]})
        uri = util.write_TA3_predictions(predictions, outputDir + "/predictions", solution, 'produce')
        uri = 'file://{uri}'.format(uri=os.path.abspath(uri))
        result = value_pb2.Value(csv_uri=uri)

        self._solution_score_map.pop(request_id, None)

        msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())

        steps = []
        for i in range(solution.num_steps()):
            steps.append(core_pb2.StepProgress(progress=msg))

        exposed_outputs = {}
        last_step_output = request_params.expose_outputs[len(request_params.expose_outputs)-1]
        exposed_outputs[last_step_output] = result

        yield core_pb2.GetProduceSolutionResultsResponse(progress=msg, steps=steps, exposed_outputs=exposed_outputs)

    def SolutionExport(self, request, context):
        logging.info("Message received: SolutionExport")
        solution_id = request.fitted_solution_id
        rank = request.rank
        solution = self._solutions[solution_id]
        solution.rank = rank

        outputdir = os.environ['D3MOUTPUTDIR'] 
        util.write_solution(solution, outputDir + "/supporting_files")
        util.write_pipeline_json(solution, self.primitives, outputDir + "/pipelines")
        util.write_pipeline_executable(solution, outputDir + "/executables")

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
