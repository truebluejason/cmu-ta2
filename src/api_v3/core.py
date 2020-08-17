"""
Implementation of the ta2ta3 API v2 (preprocessing extensions) -- core.proto
"""

__author__ = "Saswati Ray"
__email__ = "sray@cs.cmu.edu"

import core_pb2 as core_pb2
import core_pb2_grpc as core_pb2_grpc
import value_pb2 as value_pb2
import primitive_pb2 as primitive_pb2
import problem_pb2 as problem_pb2
import pipeline_pb2 as pipeline_pb2
import logging
import primitive_lib
import os, sys, copy
import pandas as pd
import pickle
import numpy as np
import search

import solutiondescription, util, auto_solutions
from multiprocessing import Pool, cpu_count
from multiprocessing.context import TimeoutError
import uuid
from timeit import default_timer as timer

logging.basicConfig(level=logging.ERROR)
pd.set_option('display.max_rows', None)

from d3m.container.dataset import D3MDatasetLoader, Dataset
from d3m.metadata import base as metadata_base
from d3m import container

class Core(core_pb2_grpc.CoreServicer):
    def __init__(self):
        self._solution_to_search = {}
        self._primitives = {}
        self._solutions = {}
        self._solution_score_map = {}
        self._search_solutions = {}
        self.async_message_thread = Pool(cpu_count()) #pool.ThreadPool(processes=1,)
        self._primitives = primitive_lib.load_primitives()         
        outputDir = os.environ['D3MOUTPUTDIR']

    def get_task_list(self, keywords):
        names = []
        for k in keywords:
            name = k #problem_pb2.TaskKeyword.Name(k)
            name = name.replace('_', '')
            names.append(name)
        return names

    def get_task_name(self, keywords):
        names = self.get_task_list(keywords)
        return util.get_task(names)

    def search_solutions(self, request, dataset):
        """
        Populate potential solutions for TA3
        """
        primitives = self._primitives
        problem = request.problem.problem
        template = request.template
        task_name = self.get_task_name(problem.task_keywords)
        logging.critical(task_name)

        solutions = []
        pipeline_placeholder_present = False
        basic_sol = None

        # TA3 has specified a pipeline
        if template != None and isinstance(template, pipeline_pb2.PipelineDescription) and len(template.steps) > 0:
            basic_sol = solutiondescription.SolutionDescription(request.problem)
            basic_sol.create_from_pipelinedescription(self._solutions, pipeline_description=template)
            if basic_sol.contains_placeholder() == False:  # Fully defined
                solutions.append(basic_sol)
                return (solutions, 0)
            else: # Placeholder present
                pipeline_placeholder_present = True    
                inputs = []
                inputs.append(dataset)
                new_dataset = basic_sol.fit(inputs=inputs, solution_dict=self._solutions)      
                dataset = new_dataset      
                logging.critical("New datset from specified pipeline: %s", new_dataset)

        taskname = task_name.replace('_', '')
        logging.critical("taskname = %s", taskname)
        metric = request.problem.problem.performance_metrics[0].metric
        posLabel = request.problem.problem.performance_metrics[0].pos_label
        start = timer()
        automl = auto_solutions.auto_solutions(taskname, request.problem)
        solutions = automl.get_solutions(dataset)
        #solutions = solution_templates.get_solutions(taskname, dataset, primitives, metric, posLabel, request.problem)
        end = timer()
        time_used = end - start
        #try:
        #    keywords = None
        #    if request.problem.data_augmentation is not None and len(request.problem.data_augmentation) > 0:
        #        data_augment = request.problem.data_augmentation
        #        logging.critical("keywords = %s", data_augment[0].keywords)
        #        (augmented_solutions, augmented_time_used) = solution_templates.get_augmented_solutions(taskname, dataset, primitives, metric, posLabel, request.problem, data_augment)
        #        (solutions, time_used) = (augmented_solutions + solutions, augmented_time_used + time_used)
        #except:
        #    logging.critical(sys.exc_info()[0])

        logging.critical("Solutions = %s", solutions)
        if pipeline_placeholder_present is True:
            new_solution_set = []
            for s in solutions:
                try:
                    pipe = copy.deepcopy(basic_sol)
                    pipe.id = str(uuid.uuid4())
                    pipe.add_subpipeline(s)
                    self._solutions[s.id] = s 
                    new_solution_set.append(pipe)
                except:
                    logging.critical(sys.exc_info()[0])            
            logging.critical("%s", new_solution_set)
            return (new_solution_set, time_used)
 
        return (solutions, time_used)
    
    def SearchSolutions(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: SearchSolutions")
        search_id_str = str(uuid.uuid4())

        self._solution_score_map[search_id_str] = request

        outputDir = os.environ['D3MOUTPUTDIR'] + "/" + search_id_str
        util.initialize_for_search(outputDir)
        return core_pb2.SearchSolutionsResponse(search_id = search_id_str)

    def _get_inputs(self, problem, rinputs):
        inputs = []
 
        for ip in rinputs:
            dataset = None
            if ip.HasField("dataset_uri") == True:
                dataset = D3MDatasetLoader().load(ip.dataset_uri)
            elif ip.HasField("csv_uri") == True:
                data = pd.read_csv(ip.csv_uri, dtype=str, header=0, na_filter=False, encoding='utf8', low_memory=False,)
                dataset = container.DataFrame(data)

            logging.critical("Problem %s", problem)
            if len(problem.inputs) > 0:
                targets = problem.inputs[0].targets
                dataset = util.add_target_metadata(dataset, targets)
                dataset = util.add_privileged_metadata(dataset, problem.inputs[0].privileged_data)
            inputs.append(dataset) 

        return inputs
       
    def GetSearchSolutionsResults(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: GetSearchSolutionsRequest")
        search_id_str = request.search_id
        
        start=solutiondescription.compute_timestamp()
        msg = core_pb2.Progress(state=core_pb2.PENDING, status="", start=start, end=solutiondescription.compute_timestamp())
        yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=0, all_ticks=0, solution_id="",
                     internal_score=0.0, scores=[])

        request_params = self._solution_score_map[search_id_str]
        count = 0
        inputs = self._get_inputs(request_params.problem, request_params.inputs)
        (solutions, time_used) = self.search_solutions(request_params, inputs[0])
        self._search_solutions[search_id_str] = []

        # Fully specified solution
        if request_params.template != None and isinstance(request_params.template, pipeline_pb2.PipelineDescription) \
            and len(request_params.template.steps) > 0 and len(solutions) == 1:
            msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())
            count = count + 1
            id = solutions[0].id
            self._solutions[id] = solutions[0]
            self._search_solutions[search_id_str].append(id)
            self._solution_to_search[id] = search_id_str 
            yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=1, all_ticks=1,
                          solution_id=id, internal_score=0.0, scores=[])
                        
        else: # Evaluate potential solutions
            index = 0
            msg = core_pb2.Progress(state=core_pb2.RUNNING, status="", start=start, end=solutiondescription.compute_timestamp())

            metric = request_params.problem.problem.performance_metrics[0].metric
            posLabel = request_params.problem.problem.performance_metrics[0].pos_label
            solutions_dict = copy.deepcopy(self._solutions)
            results =  [self.async_message_thread.apply_async(search.evaluate_solution_score, (inputs, sol, self._primitives, metric, posLabel, solutions_dict, )) for sol in solutions]
            logging.critical("Search timeout = %d", request_params.time_bound_search)
            timeout = request_params.time_bound_search * 60
            if timeout <= 0:
                timeout = None
            elif timeout > 60:
                timeout = timeout - 120
                timeout = timeout - time_used
                if timeout <= 0:
                    timeout = 1

            if timeout is not None:
                logging.critical("Timeout = %d sec", timeout)

            outputDir = os.environ['D3MOUTPUTDIR'] + "/" + search_id_str
            valid_solution_scores = {}

            # Evaluate potential solutions asynchronously and get end-result
            for r in results:
                try:
                    start_solution = timer()
                    (score, optimal_params) = r.get(timeout=timeout)
                    count = count + 1
                    id = solutions[index].id
                    self._solutions[id] = solutions[index]
                    self._search_solutions[search_id_str].append(id)
                    self._solution_to_search[id] = search_id_str
                    valid_solution_scores[index] = score
                    if optimal_params is not None and len(optimal_params) > 0:
                        solutions[index].set_hyperparams(self._solutions, optimal_params)
                    util.write_pipeline_json(solutions[index], self._primitives, self._solutions, outputDir + "/pipelines_searched", outputDir + "/subpipelines")
                    end_solution = timer()
                    time_used = end_solution - start_solution
                    timeout = timeout - time_used
                    if timeout <= 0:
                        timeout = 3
                    yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=count, all_ticks=len(solutions), solution_id=id,
                                        internal_score=0.0, scores=[])
                except TimeoutError:
                    logging.critical(solutions[index].primitives)
                    logging.critical(sys.exc_info()[0])
                    logging.critical("Solution terminated: %s", solutions[index].id)
                    timeout = 3
                except:
                    logging.critical(solutions[index].primitives)
                    logging.critical(sys.exc_info()[0])
                    logging.critical("Solution terminated: %s", solutions[index].id)
                index = index + 1

            # Sort solutions by their scores and rank them
            sorted_x = search.rank_solutions(valid_solution_scores, metric)
            sol_rank = 1
            for (index, score) in sorted_x:
                id = solutions[index].id
                #self._solutions[id] = solutions[index]
                #self._search_solutions[search_id_str].append(id)
                #self._solution_to_search[id] = search_id_str
                self._solutions[id].rank = sol_rank
                logging.critical("Rank %d", sol_rank)
                print("Score ", score)
                rank = core_pb2.Score(metric=problem_pb2.ProblemPerformanceMetric(metric="RANK"), value=value_pb2.Value(raw=value_pb2.ValueRaw(double=sol_rank)))
                search_rank = core_pb2.SolutionSearchScore(scoring_configuration=core_pb2.ScoringConfiguration(), scores=[rank])
                sscore = core_pb2.Score(metric=problem_pb2.ProblemPerformanceMetric(metric=metric), value=value_pb2.Value(raw=value_pb2.ValueRaw(double=score)))
                search_score = core_pb2.SolutionSearchScore(scoring_configuration=core_pb2.ScoringConfiguration(), scores=[sscore])
                yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=count, all_ticks=len(solutions), solution_id=id,
                                        internal_score=0.0, scores=[search_rank, search_score])
                sol_rank = sol_rank + 1  
            msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp()) 
            yield core_pb2.GetSearchSolutionsResultsResponse(progress=msg, done_ticks=count, all_ticks=count, solution_id="", internal_score=0.0, scores=[])
        
        self._solution_score_map.pop(search_id_str, None)
        logging.critical("No. of sol = %d", count)

    def EndSearchSolutions(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: EndSearchSolutions")
        search_id_str = request.search_id

        for sol_id in self._search_solutions[search_id_str]:
            self._solutions.pop(sol_id, None)
            self._solution_to_search.pop(sol_id, None)

        self._search_solutions[search_id_str].clear()
        return core_pb2.EndSearchSolutionsResponse()

    def StopSearchSolutions(self, request, context):
        """
        TA2-3 API call
        """
        search_id_str = request.search_id
        logging.critical("Message received: StopSearchSolutions")
        #self.async_message_thread.terminate()
        #self.async_message_thread.join()
        #self.async_message_thread = Pool(cpu_count())

        return core_pb2.StopSearchSolutionsResponse()

    def SaveSolution(self, request, context):

        logging.critical("Message received: SaveSolution")

        # Get the solution ID from the request object
        solution_id = request.solution_id

        logging.critical("Solution id:", solution_id)

        # Get the output directory env variable
        output_dir = os.environ['D3MOUTPUTDIR']

        logging.critical("D3MOUTPUTDIR", output_dir)

        if solution_id in self._solutions:

            logging.critical("Solution found.")

            solution_object = {
                'solution_id': solution_id,
                'solution': self._solutions[solution_id]
            }

            logging.critical(solution_object)

            with open(os.path.join(output_dir, '/temp/', solution_id+'.pickle'), 'wb') as f:

                logging.critical("File opened")

                # Pickle the solution to the file
                pickle.dump(solution_object, f, pickle.HIGHEST_PROTOCOL)

                logging.critical("Pickle dumped to file")

                # Get real path of the file
                filepath = os.path.realpath(f.name)

                logging.critical("Real file path:", filepath)

                yield core_pb2.SaveSolutionResponse(solution_uri=filepath)

                logging.critical("Yielded", core_pb2.SaveSolutionResponse(solution_uri=filepath))

        else:

            logging.critical("SaveSolution: Solution %s not found!", solution_id)
            yield core_pb2.SaveSolutionResponse(solution_uri="")

    def LoadSolution(self, request, context):

        logging.critical("Message received: LoadSolution")

        filepath = request.solution_uri

        try:

            with open(filepath, 'rb') as f:

                # Unpickle from file
                solution_object = pickle.load(f)

                # Cache the solution in memory
                self._solutions[solution_object['solution_id']] = solution_object['solution']

                yield core_pb2.LoadSolutionResponse(solution_id=solution_object['solution_id'])

        except:

            logging.critical("LoadSolution: File %s not found!", filepath)
            yield core_pb2.LoadSolutionResponse(solution_id="")

    def SaveFittedSolution(self, request, context):

        logging.critical("Message received: SaveFittedSolution")

        # Get the solution ID from the request object
        solution_id = request.solution_id

        # Get the output directory env variable
        output_dir = os.environ['D3MOUTPUTDIR']

        if solution_id in self._solutions:

            solution_object = {
                'solution_id': solution_id,
                'solution': self._solutions[solution_id]
            }

            with open(os.path.join(output_dir, '/temp/', solution_id + '.pickle'), 'wb') as f:

                # Pickle the solution to the file
                pickle.dump(solution_object, f, pickle.HIGHEST_PROTOCOL)

                # Get real path of the file
                filepath = os.path.realpath(f.name)

                yield core_pb2.SaveFittedSolutionResponse(solution_uri=filepath)

        else:

            logging.critical("SaveFittedSolution: Solution %s not found!", solution_id)
            yield core_pb2.SaveFittedSolutionResponse(solution_uri="")

    def LoadFittedSolution(self, request, context):

        logging.critical("Message received: LoadFittedSolution")

        filepath = request.fitted_solution_uri

        try:

            with open(filepath, 'rb') as f:

                # Unpickle from file
                solution_object = pickle.load(f)

                # Cache the solution in memory
                self._solutions[solution_object['solution_id']] = solution_object['solution']

                yield core_pb2.LoadSolutionResponse(fitted_solution_id=solution_object['solution_id'])

        except:

            logging.critical("LoadFittedSolution: File %s not found!", filepath)
            yield core_pb2.LoadSolutionResponse(fitted_solution_id="")

    def SplitData(self, request, context):
        logging.critical("Message received: SplitData")
        pass

    def ScorePredictions(self, request, context):
        logging.critical("Message received: ScorePredictions")
        pass

    def GetStepDescriptions(self, solution_id):
        param_map = []
        solution = self._solutions[solution_id]
        num_steps = solution.num_steps()
        for j in range(num_steps):
            if solution.primitives[j] is not None:
                step = core_pb2.StepDescription(primitive=core_pb2.PrimitiveStepDescription(hyperparams=solution.get_hyperparams(j, self._primitives)))
            else:
                step_array = self.GetStepDescriptions(solution.subpipelines[j])
                step = core_pb2.StepDescription(pipeline=core_pb2.SubpipelineStepDescription(steps=step_array))
            param_map.append(step)
        return param_map

    def DescribeSolution(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: DescribeSolution")
        solution_id = request.solution_id
        solution = self._solutions[solution_id]
        desc = solution.describe_solution(self._primitives, self._solutions)
        param_map = self.GetStepDescriptions(solution_id)

        logging.critical(param_map)
        return core_pb2.DescribeSolutionResponse(pipeline=desc, steps=param_map)

    def ScoreSolution(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: ScoreSolution")

        request_id = str(uuid.uuid4())
        self._solution_score_map[request_id] = request

        return core_pb2.ScoreSolutionResponse(request_id = request_id)

    def GetScoreSolutionResults(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: GetScoreSolutionResults")
        request_id = request.request_id
        request_params = self._solution_score_map[request_id]
        
        start=solutiondescription.compute_timestamp()
        solution_id = request_params.solution_id
        msg = core_pb2.Progress(state=core_pb2.RUNNING, status="", start=start, end=solutiondescription.compute_timestamp())
        
        send_scores = []
        from timeit import default_timer as timer

        if solution_id not in self._solutions:
            logging.critical("GetScoreSolutionResults: Solution %s not found!", solution_id)  
            msg = core_pb2.Progress(state=core_pb2.ERRORED, status="", start=start, end=solutiondescription.compute_timestamp())
            # Clean up
            self._solution_score_map.pop(request_id, None)
            yield core_pb2.GetScoreSolutionResultsResponse(progress=msg, scores=[])
        else:
            inputs = self._get_inputs(self._solutions[solution_id].problem, request_params.inputs)
            score = 0.0
            try:
                s = timer()                
                (score, optimal_params) = self._solutions[solution_id].score_solution(inputs=inputs, metric=request_params.performance_metrics[0].metric,
                                posLabel = request_params.performance_metrics[0].pos_label,
                                primitive_dict=self._primitives, solution_dict=self._solutions)
                if optimal_params is not None and len(optimal_params) > 0:
                    self._solutions[solution_id].set_hyperparams(self._solutions, optimal_params)

                e = timer()
                logging.critical("Time taken = %s sec", e-s) 
            except:
                score = 0.0
                logging.critical("Exception in score: %s", self._solutions[solution_id].primitives)
                logging.critical("Exception in score: %s", sys.exc_info()[0])
            
            search_id_str = self._solution_to_search[solution_id]
            outputDir = os.environ['D3MOUTPUTDIR'] + "/" + search_id_str
            try:
                util.write_pipeline_json(self._solutions[solution_id], self._primitives, self._solutions, outputDir + "/pipelines_scored", outputDir + "/subpipelines")
            except:
                logging.critical(sys.exc_info()[0])
                logging.critical(self._solutions[solution_id].primitives)
            logging.critical("Score = %f", score)
            send_scores.append(core_pb2.Score(metric=request_params.performance_metrics[0],
             fold=0, value=value_pb2.Value(raw=value_pb2.ValueRaw(double=score)), random_seed=0))

            yield core_pb2.GetScoreSolutionResultsResponse(progress=msg, scores=[]) 

            # Clean up
            self._solution_score_map.pop(request_id, None)

            msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())
            yield core_pb2.GetScoreSolutionResultsResponse(progress=msg, scores=send_scores)

    def FitSolution(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: FitSolution")
        request_id = str(uuid.uuid4())
        self._solution_score_map[request_id] = request
        return core_pb2.FitSolutionResponse(request_id = request_id)

    def GetFitSolutionResults(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: GetFitSolutionResults")
        request_id = request.request_id
        request_params = self._solution_score_map[request_id]
        start=solutiondescription.compute_timestamp()

        solution_id = request_params.solution_id

        if solution_id not in self._solutions:
            logging.critical("GetFitSolutionResults: Solution %s not found!", solution_id)
            msg = core_pb2.Progress(state=core_pb2.ERRORED, status="", start=start, end=solutiondescription.compute_timestamp())
            # Clean up
            self._solution_score_map.pop(request_id, None)
            yield core_pb2.GetFitSolutionResultsResponse(progress=msg, steps=[], exposed_outputs=[], fitted_solution_id=None)
        else:
            solution = self._solutions[solution_id]

            msg = core_pb2.Progress(state=core_pb2.RUNNING, status="", start=start, end=solutiondescription.compute_timestamp())
            
            solution.create_pipeline_json(self._primitives, self._solutions) 

            inputs = self._get_inputs(solution.problem, request_params.inputs)
            try:
                output = solution.fit(inputs=inputs, solution_dict=self._solutions)
                logging.critical("Fit predictions with rows = %s", len(output))
            except:
                logging.critical("Exception in fit: %s", solution.primitives)
                logging.critical("Exception in fit: %s", sys.exc_info()[0])
                output = None

            result = None
            search_id_str = self._solution_to_search[solution_id]
            outputDir = os.environ['D3MOUTPUTDIR'] + "/" + search_id_str

            if output is not None:
                uri = util.write_predictions(output, outputDir + "/predictions", request_id)
                uri = 'file://{uri}'.format(uri=os.path.abspath(uri)) 
                result = value_pb2.Value(csv_uri=uri)
            else:
                result = value_pb2.Value(error = value_pb2.ValueError(message="Output is NULL"))

            yield core_pb2.GetFitSolutionResultsResponse(progress=msg, steps=[], exposed_outputs=[], fitted_solution_id=solution.id)

            msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())

            steps = []
            for i in range(solution.num_steps()):
                steps.append(core_pb2.StepProgress(progress=msg))

            exposed_outputs = {}
            if request_params.expose_outputs is not None and len(request_params.expose_outputs) > 0:
                last_step_output = request_params.expose_outputs[len(request_params.expose_outputs)-1]
            else:
                last_step_output = solution.outputs[0][2]

            exposed_outputs[last_step_output] = result

            # Clean up
            self._solution_score_map.pop(request_id, None)

            yield core_pb2.GetFitSolutionResultsResponse(progress=msg, steps=steps, exposed_outputs=exposed_outputs, fitted_solution_id=solution.id)

    def ProduceSolution(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: ProduceSolution")
        request_id = str(uuid.uuid4())
        self._solution_score_map[request_id] = request

        return core_pb2.ProduceSolutionResponse(request_id = request_id)

    def GetProduceSolutionResults(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: GetProduceSolutionResults")
        request_id = request.request_id
        request_params = self._solution_score_map[request_id]
        start=solutiondescription.compute_timestamp()

        solution_id = request_params.fitted_solution_id
        solution = self._solutions[solution_id]

        inputs = self._get_inputs(solution.problem, request_params.inputs)
        try:
            output = solution.produce(inputs=inputs, solution_dict=self._solutions)[0]
            logging.critical("Produce predictions with rows = %s", len(output))
        except:
            logging.critical("Exception in produce: %s", solution.primitives)
            logging.critical("Exception in produce: %s", sys.exc_info()[0])
            output = None
    
        result = None
        search_id_str = self._solution_to_search[solution_id]
        outputDir = os.environ['D3MOUTPUTDIR'] + "/" + search_id_str

        if output is not None:
            uri = util.write_predictions(output, outputDir + "/predictions", request_id)
            uri = 'file://{uri}'.format(uri=os.path.abspath(uri))
            result = value_pb2.Value(csv_uri=uri)
        else:
            result = value_pb2.Value(error = value_pb2.ValueError(message="Output is NULL"))

        self._solution_score_map.pop(request_id, None)

        msg = core_pb2.Progress(state=core_pb2.COMPLETED, status="", start=start, end=solutiondescription.compute_timestamp())

        steps = []
        for i in range(solution.num_steps()):
            steps.append(core_pb2.StepProgress(progress=msg))

        exposed_outputs = {}
        if request_params.expose_outputs is not None and len(request_params.expose_outputs) > 0:
            last_step_output = request_params.expose_outputs[len(request_params.expose_outputs)-1]
        else:
            last_step_output = solution.outputs[0][2]

        exposed_outputs[last_step_output] = result

        yield core_pb2.GetProduceSolutionResultsResponse(progress=msg, steps=steps, exposed_outputs=exposed_outputs)

    def SolutionExport(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: SolutionExport")
        solution_id = request.solution_id
        rank = request.rank
        solution = self._solutions[solution_id]
        solution.rank = rank

        search_id_str = self._solution_to_search[solution_id]
        outputDir = os.environ['D3MOUTPUTDIR'] + "/" + search_id_str
        util.write_pipeline_json(solution, self._primitives, self._solutions, outputDir + "/pipelines_ranked", outputDir + "/subpipelines", rank=solution.rank)
        util.write_rank_file(solution, rank, outputDir + "/pipelines_ranked")

        return core_pb2.SolutionExportResponse()

    def UpdateProblem(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: UpdateProblem")

        return core_pb2.UpdateProblemResponse()

    def ListPrimitives(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: ListPrimitives")

        primitives = []
        for classname, p in self._primitives.items():
            primitives.append(primitive_pb2.Primitive(id=p.id, version=p.primitive_class.version, python_path=p.primitive_class.python_path, name=p.primitive_class.name, digest=None))
        return core_pb2.ListPrimitivesResponse(primitives=primitives)

    def Hello(self, request, context):
        """
        TA2-3 API call
        """
        logging.critical("Message received: Hello")
        version = core_pb2.DESCRIPTOR.GetOptions().Extensions[
                    core_pb2.protocol_version]
        return core_pb2.HelloResponse(user_agent="cmu_ta2",
        version=version,
        allowed_value_types = ['RAW', 'DATASET_URI', 'CSV_URI'],
        supported_extensions = [])
        
def add_to_server(server):
    core_pb2_grpc.add_CoreServicer_to_server(Core(), server)
