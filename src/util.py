import logging
logging.basicConfig(level=logging.INFO)

__version__ = "0.1.0"

from d3m.container.dataset import D3MDatasetLoader, Dataset
from d3m.metadata import base as metadata_base
from d3m.metadata.base import Metadata
from d3m.metadata.pipeline import Pipeline, PrimitiveStep
import problem_pb2 as problem_pb2
import os, json
import pickle
import solutiondescription
import pandas as pd

def generate_pipeline(pipeline_uri: str, dataset_uri: str, problem_doc_uri: str):
    # Pipeline description
    pipeline_description = None
    if '.json' in pipeline_uri:
        with open(pipeline_uri) as pipeline_file:
            pipeline_description = Pipeline.from_json_content(string_or_file=pipeline_file)
    else:
        with open(pipeline_uri) as pipeline_file:
            pipeline_description = Pipeline.from_yaml_content(string_or_file=pipeline_file)

    # Pipeline
    solution = solutiondescription.SolutionDescription(None)
    solution.create_from_pipeline(pipeline_description)
    return solution

def load_problem_doc(problem_doc_uri: str):
    """     
    Load problem_doc from problem_doc_uri     
    Parameters     ---------     
    problem_doc_uri     Uri where the problemDoc.json is located
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

def get_target_name(problem_doc_metadata: 'Metadata'):
    data = problem_doc_metadata.query(())['inputs']['data'][0]
    target = data['targets'][0]['colName']
    return target

def load_schema(filename):
    print("Reading ",filename)
    with open(filename) as file:
        schema =  json.load(file)

    dataset_schema = schema['dataset_schema']
    problem_schema = schema['problem_schema']

    dataset_uri = 'file://{dataset_uri}'.format(dataset_uri=dataset_schema)
    dataset = D3MDatasetLoader().load(dataset_uri)
    problem_doc = load_problem_doc(problem_schema)

    dataset = add_target_columns_metadata(dataset, problem_doc)

    taskname = problem_doc.query(())['about']['taskType']
    target = get_target_name(problem_doc)

    return (dataset, taskname, target)

def get_pipeline(dirname, pipeline_name):
    newdirname = dirname + "/" + pipeline_name
    filename = pipeline_name.split("_")[0]
    f = newdirname + "/" + filename + ".dump"

    solution = pickle.load(open(f, 'rb'))
    return solution

def write_solution(solution, dirname):
    rank = str(solution.rank)
    supporting_dirname = dirname + "/" + solution.id + "_" + rank
    if not os.path.exists(supporting_dirname):
        os.makedirs(supporting_dirname)
    output = open(supporting_dirname+"/"+solution.id+".dump", "wb")
    pickle.dump(solution, output)
    output.close()

def write_TA3_predictions(predictions, dirname, solution, mode):
    directory = dirname + "/" + solution.id + "_" + mode
    if not os.path.exists(directory):
        os.makedirs(directory)

    outputFilePath = directory + "/predictions.csv"
    with open(outputFilePath, 'w') as outputFile:
        predictions.to_csv(outputFile, header=True, index=False)
    return outputFilePath

def write_predictions(predictions, dirname, solution):
    directory = dirname + "/" + solution.id + "_" + str(solution.rank)
    if not os.path.exists(directory):
        os.makedirs(directory)

    outputFilePath = directory + "/predictions.csv"
    with open(outputFilePath, 'w') as outputFile:
        predictions.to_csv(outputFile, header=True, index=False)
    return outputFilePath
   
def write_pipeline_json(solution, primitives, dirname):
    filename = dirname + "/" + solution.id + "_" + str(solution.rank) + ".json"
    solution.create_pipeline_json(primitives, filename) 

def write_pipeline_executable(solution, dirname):
    shell_script = '#!/bin/bash\n python ./src/main.py test\n'
    filename = dirname + "/" + solution.id + "_" + str(solution.rank) + ".sh"
    with open(filename, 'w') as f:
        f.write(shell_script)
    os.chmod(filename, 0o755)

def invert_metric(metric_type):
    min_metrics = set()
    min_metrics.add(problem_pb2.MEAN_SQUARED_ERROR)
    min_metrics.add(problem_pb2.ROOT_MEAN_SQUARED_ERROR)
    min_metrics.add(problem_pb2.ROOT_MEAN_SQUARED_ERROR_AVG)
    min_metrics.add(problem_pb2.MEAN_ABSOLUTE_ERROR)
    if metric_type in min_metrics:
        return True
    return False

