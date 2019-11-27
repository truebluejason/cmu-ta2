
import os, sys
import util
import solution_templates
import problem_pb2
import primitive_lib
import logging
from multiprocessing import Pool, cpu_count

def rank_solutions(valid_solution_scores, problem_metric):
    """
    Return sorted list of multiple solutions.
    """
    # Sort solutions by their scores and rank them
    import operator
    sorted_x = sorted(valid_solution_scores.items(), key=operator.itemgetter(1))
    if util.invert_metric(problem_metric) is False:
        sorted_x.reverse()
    return sorted_x

def search_phase():
    """
    TA2 running in stand-alone search phase
    """
    inputDir = os.environ['D3MINPUTDIR']
    outputDir = os.environ['D3MOUTPUTDIR']
    timeout_env = os.environ['D3MTIMEOUT']
    num_cpus = (int)(os.environ['D3MCPU'])
    problemPath = os.environ['D3MPROBLEMPATH']

    logger = logging.getLogger()
    logger.setLevel(level=logging.ERROR)
    print("D3MINPUTDIR = ", inputDir)
    print("D3MOUTPUTDIR = ", outputDir)
    print("timeout = ", timeout_env)
    print("cpus = ", num_cpus)
    (dataset, task_name, problem_desc, metric, posLabel, keywords) = util.load_data_problem(inputDir, problemPath)

    print("Metric = ", metric, " poslabel = ", posLabel)
    timeout_in_min = (int)(timeout_env)
    primitives = primitive_lib.load_primitives()
    task_name = task_name.upper()
    print(task_name)

    if task_name == "TIMESERIES":
        task_name = "TIMESERIESFORECASTING"

    problem_metric = problem_pb2.F1_MACRO
    if metric == 'f1Macro':
        problem_metric = problem_pb2.F1_MACRO
    elif metric == 'f1': 
        problem_metric = problem_pb2.F1
    elif metric == 'accuracy':
        problem_metric = problem_pb2.ACCURACY
    elif metric == 'meanSquaredError':
        problem_metric = problem_pb2.MEAN_SQUARED_ERROR
    elif metric == 'rootMeanSquaredError':
        problem_metric = problem_pb2.ROOT_MEAN_SQUARED_ERROR
    elif metric == 'meanAbsoluteError':
        problem_metric = problem_pb2.MEAN_ABSOLUTE_ERROR

    # Still run the normal pipeline even if augmentation
    (solutions, time_used) = solution_templates.get_solutions(task_name, dataset, primitives, problem_metric, posLabel, None)

    if keywords:
        (augmented_solutions, augmented_time_used) = solution_templates.get_augmented_solutions(task_name, dataset, primitives, problem_metric, posLabel, None, keywords)
        (solutions, time_used) = (augmented_solutions + solutions, augmented_time_used + time_used)

    async_message_thread = Pool((int)(num_cpus))
    valid_solutions = {}
    valid_solution_scores = {}

    inputs = []
    inputs.append(dataset)

    # Score potential solutions
    results = [async_message_thread.apply_async(evaluate_solution_score, (inputs, sol, primitives, problem_metric, posLabel, None, )) for sol in solutions]
    timeout = timeout_in_min * 60
    halftimeout = None
    if timeout <= 0:
        timeout = None
    elif timeout > 60:
        timeout = timeout - 60

    if timeout is not None:
        halftimeout = timeout/2

    index = 0
    for r in results:
        try:
            (score, optimal_params) = r.get(timeout=halftimeout)
            id = solutions[index].id
            valid_solutions[id] = solutions[index]
            valid_solution_scores[id] = score
            if optimal_params is not None and len(optimal_params) > 0:
                valid_solutions[id].set_hyperparams(optimal_params)
        except:
            print(solutions[index].primitives)
            print(sys.exc_info()[0])
            print("Solution terminated: ", solutions[index].id)
        index = index + 1

    # Sort solutions by their scores and rank them
    sorted_x = rank_solutions(valid_solution_scores, problem_metric)

    index = 1
    for (sol, score) in sorted_x:
        valid_solutions[sol].rank = index
        print("Rank ", index)
        print("Score ", score)
        print(valid_solutions[sol].primitives)
        index = index + 1

    num = 20
    if len(sorted_x) < 20:
        num = len(sorted_x)

    util.initialize_for_search(outputDir)

    # Fit solutions and dump out files
    sorted_x = sorted_x[:num]
    results = [async_message_thread.apply_async(fit_solution, (inputs, valid_solutions[sol], primitives, outputDir, problem_desc))
     for (sol,score) in sorted_x]

    index = 0
    for r in results:
        try:
            valid=r.get(timeout=halftimeout)
        except:
            print(valid_solutions[sorted_x[index][0]].primitives)
            print(sys.exc_info()[0])
            print("Solution terminated: ", valid_solutions[sorted_x[index][0]].id)
        index = index + 1

def evaluate_solution_score(inputs, solution, primitives, metric, posLabel, sol_dict):
    """
    Scores each potential solution
    Runs in a separate process
    """
    print("Evaluating ", solution.id)

    (score, optimal_params) = solution.score_solution(inputs=inputs, metric=metric, posLabel=posLabel,
                                primitive_dict=primitives, solution_dict=sol_dict)

    return (score, optimal_params)

def fit_solution(inputs, solution, primitives, outputDir, problem_desc):
    """
    Fits each potential solution
    Runs in a separate process
    """
    print("Fitting ", solution.id)

    #output = solution.fit(inputs=inputs, solution_dict=None)
    #output = solution.produce(inputs=inputs, solution_dict=None)
    util.write_pipeline_json(solution, primitives, outputDir + "/pipelines_ranked", outputDir + "/subpipelines", rank=solution.rank)
    #util.write_pipeline_yaml(solution, outputDir + "/pipeline_runs", inputs, problem_desc)
    return True
