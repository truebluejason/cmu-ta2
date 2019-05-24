"""
This should be where everything comes together: Problem descriptions get matched up
with the available primitives, and a plan for how to create a solution gets made.

    So it needs to:
    Choose hyperparameters for a primitive
    Run the primitive
    Measure the results
    Feed the results and hyperparameters back into the chooser
"""

import core_pb2, problem_pb2, pipeline_pb2, primitive_pb2, value_pb2
import pandas as pd

from  api_v3 import core

import uuid, sys, copy, math
import time
from enum import Enum
from time import sleep
from google.protobuf.timestamp_pb2 import Timestamp

import json, numpy
import dateutil

from d3m.metadata.pipeline import Pipeline, PrimitiveStep, SubpipelineStep
from d3m.metadata.pipeline_run import PipelineRun, RuntimeEnvironment
from d3m.metadata.base import PrimitiveFamily, Context, ArgumentType
from d3m.metadata import base as metadata_base
from d3m.primitive_interfaces.base import PrimitiveBaseMeta
from d3m.container import DataFrame as d3m_dataframe
from d3m.runtime import Runtime
from sri.psl.link_prediction import LinkPredictionHyperparams
import d3m.index

import networkx as nx
import util
import solution_templates

import logging

logging.basicConfig(level=logging.INFO)

def get_cols_to_encode(df):
    """
    Find categorical attributes which can be one-hot-encoded.
    """
    cols = df.metadata.get_columns_with_semantic_type("https://metadata.datadrivendiscovery.org/types/CategoricalData")
    targets = df.metadata.get_columns_with_semantic_type("https://metadata.datadrivendiscovery.org/types/TrueTarget")

    for t in targets:
        if t in cols:
            cols.remove(t)

    rows = len(df)
    # use rule of thumb to exclude categorical atts with high cardinality for one-hot-encoding
    max_num_cols = math.log(rows, 2)
    tmp_cols = copy.deepcopy(cols)
    for t in tmp_cols:
        if len(df.iloc[:,t].unique()) > max_num_cols:
            cols.remove(t)
    return list(cols)

def column_types_present(dataset):
    """
    Retrieve special data types present: Text, Image, Timeseries, Audio, Categorical
    Returns ([data types], total columns, total rows, [categorical att indices], ok_to_denormalize)
    """
    ok_to_denormalize = True
    ok_to_impute = False
    try:
        primitive = d3m.index.get_primitive('d3m.primitives.data_transformation.denormalize.Common')
        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        model = primitive(hyperparams=primitive_hyperparams.defaults())
        ds = model.produce(inputs=dataset).value
        dataset = ds
    except:
        print("Exception with denormalize!")
        ok_to_denormalize = False

    primitive = d3m.index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common')
    primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    model = primitive(hyperparams=primitive_hyperparams.defaults())
    df = model.produce(inputs=dataset).value

    metadata = df.metadata

    types = []
    cols = len(metadata.get_columns_with_semantic_type("http://schema.org/Text"))
    if cols > 0:
        types.append('TEXT')
    cols = len(metadata.get_columns_with_semantic_type("http://schema.org/ImageObject"))
    if cols > 0:
        types.append('IMAGE')
    cols = len(metadata.get_columns_with_semantic_type("https://metadata.datadrivendiscovery.org/types/Timeseries"))
    if cols > 0:
        types.append('TIMESERIES')
    cols = len(metadata.get_columns_with_semantic_type("http://schema.org/AudioObject"))
    if cols > 0:
        types.append('AUDIO')

    cols = get_cols_to_encode(df)
    if len(cols) > 0:
        types.append('Categorical')
        print("Cats = ", cols)
    
    privileged = metadata.get_columns_with_semantic_type("https://metadata.datadrivendiscovery.org/types/SuggestedPrivilegedData") 
    attcols = metadata.get_columns_with_semantic_type("https://metadata.datadrivendiscovery.org/types/Attribute")
    for t in attcols:
        column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, t))
        semantic_types = column_metadata.get('semantic_types', [])
        if "http://schema.org/Integer" in semantic_types or "http://schema.org/Float" in semantic_types or "http://schema.org/DateTime" in semantic_types:
            ok_to_impute = True
            break

    print("Data types present: ", types)
    return (types, len(attcols), len(df), cols, ok_to_denormalize, ok_to_impute, privileged)

def compute_timestamp():
    now = time.time()
    seconds = int(now)
    return Timestamp(seconds=seconds)

class StepType(Enum):
    PRIMITIVE = 1
    SUBPIPELINE = 2
    PLACEHOLDER = 3

class ActionType(Enum):
    FIT = 1
    SCORE = 2
    VALIDATE = 3

def get_values(arg):
    value = None
    if arg.HasField("double") == True:
        value = arg.double
    elif arg.HasField("int64") == True:
        value = arg.int64
    elif arg.HasField("bool") == True:
        value = arg.bool
    elif arg.HasField("string") == True:
        value = arg.string
    elif arg.HasField("bytes") == True:
        value = arg.bytes
    elif arg.HasField("list") == True:
        value = get_list_items(arg.list.items)
    elif arg.HasField("dict") == True:
        value = get_dict_items(arg.dict.items)

    return value

def get_list_items(values : value_pb2.ValueList):
    items = []
 
    for i in values:
        items.append(get_values(i))

    return items

def get_dict_items(values : value_pb2.ValueDict):
    items = {}

    for name, arg in values.items():
        items[name] = get_values(arg)
    return items

class SolutionDescription(object):
    """
    A wrapper of a pipeline.

    The idea is that this can be evaluated, produce a model and performance metrics,
    and the hyperparameter tuning can consume that and choose what to do next.
    """
    def __init__(self, problem, static_dir):
        self.id = str(uuid.uuid4())
        self.source = None
        self.created = compute_timestamp()
        self.context = pipeline_pb2.PRETRAINING
        self.name = None
        self.description = None
        self.users = None
        self.inputs = []
        self.rank = 1
        
        self.outputs = None
        self.execution_order = None
        self.primitives_arguments = None
        self.primitives = None
        self.subpipelines = None
        self.pipeline = None
        self.produce_order = None
        self.hyperparams = None
        self.problem = problem
        self.steptypes = None
        self.taskname = None
        self.exclude_columns = None
        self.primitives_outputs = None
        self.static_dir = static_dir
        self.pipeline_description = None
        self.total_cols = 0
        self.categorical_atts = None
        self.ok_to_denormalize = True
        self.ok_to_impute = False
        self.privileged = None

    def set_categorical_atts(self, atts):
        self.categorical_atts = atts

    def set_denormalize(self, ok_to_denormalize):
        self.ok_to_denormalize = ok_to_denormalize

    def set_impute(self, ok_to_impute):
        self.ok_to_impute = ok_to_impute

    def set_privileged(self, privileged):
        self.privileged = privileged

    def contains_placeholder(self):
        if self.steptypes is None:
            return False

        for step in self.steptypes:
            if step == StepType.PLACEHOLDER:
                return True
        return False

    def num_steps(self):
        if self.primitives_arguments is not None:
            return len(self.primitives_arguments)
        else:
            return 0

    def create_pipeline_json(self, prim_dict):
        """
        Generate pipeline.json
        """
        name = "Pipeline for evaluation"
        pipeline_id = self.id #+ "_" + str(self.rank)
        pipeline_description = Pipeline(pipeline_id=pipeline_id, context=Context.EVALUATION, name=name)
        for ip in self.inputs:
            pipeline_description.add_input(name=ip['name'])

        num = self.num_steps()
        for i in range(num):
            if self.steptypes[i] == StepType.PRIMITIVE: # Primitive
                p = prim_dict[self.primitives[i]]
                pdesc = {}
                pdesc['id'] = p.id
                pdesc['version'] = p.primitive_class.version
                pdesc['python_path'] = p.primitive_class.python_path
                pdesc['name'] = p.primitive_class.name
                pdesc['digest'] = p.primitive_class.digest
                step = PrimitiveStep(primitive_description=pdesc)
                step.add_output(output_id=p.primitive_class.produce_methods[0])
                for name, value in self.primitives_arguments[i].items():
                    origin = value['origin']
                    argument_type = ArgumentType.CONTAINER
                    step.add_argument(name=name, argument_type=argument_type, data_reference=value['data'])
                if self.hyperparams[i] is not None:
                    for name, value in self.hyperparams[i].items():
                        step.add_hyperparameter(name=name, argument_type=ArgumentType.VALUE, data=value)
            else: # Subpipeline
                pdesc = self.subpipelines[i].pipeline_description
                if pdesc is None:
                    self.subpipelines[i].create_pipeline_json(prim_dict)
                    pdesc = self.subpipelines[i].pipeline_description 
                step = SubpipelineStep(pipeline=pdesc)
                ipname = 'steps.' + str(i-1) + '.produce'
                step.add_input(ipname)
                for output in self.subpipelines[i].outputs:
                    step.add_output(output_id=output[2])

            pipeline_description.add_step(step)

        logging.info("DR = %s", pipeline_description.get_available_data_references())
        for op in self.outputs:
            pipeline_description.add_output(data_reference=op[2], name=op[3])

        self.pipeline_description = pipeline_description
   
    def write_pipeline_json(self, prim_dict, dirName, rank=None):
        """
        Output pipeline to JSON file
        """
        filename = dirName + "/" + self.id + ".json"
        if self.pipeline_description is None:
            self.create_pipeline_json(prim_dict)

        with open(filename, "w") as out:
            parsed = json.loads(self.pipeline_description.to_json())
            if rank is not None:
                parsed['pipeline_rank'] = str(rank)
            json.dump(parsed, out, indent=4)    

    def write_pipeline_run(self, problem_description, dataset, filename_yaml): 
        runtime = Runtime(pipeline=self.pipeline_description, problem_description=problem_description, context=Context.TESTING, is_standard_pipeline=True)
        output = runtime.fit(inputs=dataset)
        pipeline_run = output.pipeline_run
        with open(filename_yaml, "w") as out:
            pipeline_run.to_yaml(file=filename_yaml)

    def create_from_pipelinedescription(self, pipeline_description: pipeline_pb2.PipelineDescription) -> None:
        """
        Initialize a solution object from a pipeline_pb2.PipelineDescription object passed by TA3.
        """
        n_steps = len(pipeline_description.steps)

        logging.info("Steps = %d", n_steps)
        logging.info("Running %s", pipeline_description)
       
        self.inputs = []
        for ip in pipeline_description.inputs:
             self.inputs.append({"name": ip.name})
        self.id = pipeline_description.id
        self.source = {}
        self.source['name'] = pipeline_description.source.name
        self.source['contact'] = pipeline_description.source.contact
        self.source['pipelines'] = []
        for id in pipeline_description.source.pipelines:
            self.source['pipelines'].append(id)

        self.name = pipeline_description.name
        self.description = pipeline_description.description
        self.users = []
        for user in pipeline_description.users:
            self.users.append({"id": user.id, "reason": user.reason, "rationale": user.rationale})

        self.subpipelines = {}
        self.primitives_arguments = {}
        self.primitives = {}
        self.hyperparams = {}
        self.steptypes = []
        for i in range(0, n_steps):
            self.primitives_arguments[i] = {}
            self.hyperparams[i] = None

        self.execution_order = None

        self.pipeline = [None] * n_steps
        self.outputs = []

        # Constructing DAG to determine the execution order
        execution_graph = nx.DiGraph()
        for i in range(0, n_steps):
            # PrimitivePipelineDescriptionStep
            if pipeline_description.steps[i].HasField("primitive") == True:
                s = pipeline_description.steps[i].primitive
                python_path = s.primitive.python_path
                prim = d3m.index.get_primitive(python_path)
                self.primitives[i] = prim
                self.subpipelines[i] = None
                arguments = s.arguments
                self.steptypes.append(StepType.PRIMITIVE)

                for name,argument in arguments.items():
                    if argument.HasField("container") == True:
                        data = argument.container.data
                    else:
                        data = argument.data.data
                    origin = data.split('.')[0]
                    source = data.split('.')[1]
                    self.primitives_arguments[i][name] = {'origin': origin, 'source': int(source), 'data': data}
                    
                    if origin == 'steps':
                        execution_graph.add_edge(str(source), str(i))
                    else:
                        execution_graph.add_edge(origin, str(i))

                hyperparams = s.hyperparams
                if hyperparams is not None:
                    self.hyperparams[i] = {}
                    for name,argument in hyperparams.items():
                        value = None
                        if argument.HasField("value") == True:
                            arg = argument.value.data.raw
                            value = get_values(arg)
                        self.hyperparams[i][name] = value

            # SubpipelinePipelineDescriptionStep
            elif pipeline_description.steps[i].HasField("pipeline") == True:
                s = pipeline_description.steps[i].pipeline
                self.primitives[i] = None
                self.subpipelines[i] = s
                self.steptypes.append(StepType.SUBPIPELINE)
                for j in range(len(s.inputs)):
                    argument_edge = s.inputs[j].data
                    origin = argument_edge.split('.')[0]
                    source = argument_edge.split('.')[1]
                    self.primitives_arguments[i]['inputs'] = {'origin': origin, 'source': int(source), 'data': argument_edge}

                    if origin == 'steps':
                        execution_graph.add_edge(str(source), str(i))
                    else:
                        execution_graph.add_edge(origin, str(i))

            else: # PlaceholderPipelineDescriptionStep
                s = pipeline_description.steps[i].placeholder
                self.steptypes.append(StepType.PLACEHOLDER)
                self.primitives[i] = None
                self.subpipelines[i] = None
                for j in range(len(s.inputs)):
                    argument_edge = s.inputs[j].data
                    origin = argument_edge.split('.')[0]
                    source = argument_edge.split('.')[1]
                    self.primitives_arguments[i]['inputs'] = {'origin': origin, 'source': int(source), 'data': argument_edge}

                    if origin == 'steps':
                        execution_graph.add_edge(str(source), str(i))
                    else:
                        execution_graph.add_edge(origin, str(i))
            
        execution_order = list(nx.topological_sort(execution_graph))

        # Removing non-step inputs from the order
        execution_order = list(filter(lambda x: x.isdigit(), execution_order))
        self.execution_order = [int(x) for x in execution_order]

        # Creating set of steps to be call in produce
        self.produce_order = set()
        for i in range(len(pipeline_description.outputs)):
            output = pipeline_description.outputs[i]
            origin = output.data.split('.')[0]
            source = output.data.split('.')[1]
            self.outputs.append((origin, int(source), output.data, output.name))

            if origin != 'steps':
                continue
            else:
                current_step = int(source)
                self.produce_order.add(current_step)
                for i in range(0, len(execution_order)):
                    step_origin = self.primitives_arguments[current_step]['inputs']['origin']
                    step_source = self.primitives_arguments[current_step]['inputs']['source']
                    if step_origin != 'steps':
                        break
                    else:
                        self.produce_order.add(step_source)
                        current_step = step_source
        logging.info("Outputs = %s", self.outputs)

    def isDataFrameStep(self, n_step):
        if self.steptypes[n_step] is StepType.PRIMITIVE and \
        self.primitives[n_step].metadata.query()['python_path'] == 'd3m.primitives.data_transformation.dataset_to_dataframe.Common':
            return True
        return False

    def exclude(self, df):
        """
        Exclude columns which cannot/need not be processed.
        """
        self.exclude_columns = set()
        metadata = df.metadata
        rows = len(df)
        attributes = metadata.get_columns_with_semantic_type("https://metadata.datadrivendiscovery.org/types/Attribute")

        # Retain max 100 attributes
        max_num_atts = 100
        if len(attributes) > max_num_atts:
            for i in range(len(attributes)-max_num_atts):
                col = (int)(numpy.random.choice(attributes, replace=False))
                self.exclude_columns.add(col)

        #if rows > 1000000: # > 1M rows
        cols = metadata.get_columns_with_semantic_type("https://metadata.datadrivendiscovery.org/types/FloatVector") # Found to be very expensive in ColumnParser!
        for col in cols:
            self.exclude_columns.add(col)

        targets = metadata.get_columns_with_semantic_type("https://metadata.datadrivendiscovery.org/types/TrueTarget")
        for t in targets:
            if t in self.exclude_columns:
                self.exclude_columns.remove(t)

    def fit(self, **arguments):
        """
        Train all steps in the solution.

        Paramters
        ---------
        arguments
            Arguments required to train the solution
        """
        primitives_outputs = [None] * len(self.execution_order)
   
        if self.primitives_outputs is None: 
            for i in range(0, len(self.execution_order)):
                n_step = self.execution_order[i]
                primitives_outputs[n_step] = self.process_step(n_step, primitives_outputs, ActionType.FIT, arguments)

                if self.isDataFrameStep(n_step) == True:
                    self.exclude(primitives_outputs[n_step])
        else:
            last_step = self.get_last_step()
            primitives_outputs[last_step] = self.process_step(last_step, self.primitives_outputs, ActionType.FIT, arguments)
            primitives_outputs[1] = self.primitives_outputs[1]
            if last_step == len(self.execution_order) - 2:
                primitives_outputs[last_step+1] = self.process_step(last_step+1, primitives_outputs, ActionType.FIT, arguments)
      
        v = primitives_outputs[len(self.execution_order)-1]
        return v

    def _pipeline_step_fit(self, n_step: int, pipeline_id: str, primitive_arguments, arguments, action):
        """
        Execute a subpipeline step
        
        Paramters
        ---------
        n_step: int
            An integer of the actual step.
        pipeline_id: str
            
        primitive_arguments
            Arguments for the solution
        """
        solution = arguments['solution_dict'][pipeline_id]

        inputs = []
        inputs.append(primitive_arguments['inputs'])

        if action is ActionType.FIT: 
            return solution.fit(inputs=inputs, solution_dict=arguments['solution_dict'])
        elif action is ActionType.SCORE:
            return solution.score_solution(inputs=inputs, solution_dict=arguments['solution_dict'], primitive_dict=arguments['primitive_dict'],
                   posLabel=arguments['posLabel'], metric=arguments['metric'])
        else:
            return solution.validate_solution(inputs=inputs, solution_dict=arguments['solution_dict'])
 
    def fit_step(self, n_step: int, primitive: PrimitiveBaseMeta, primitive_arguments):
        """
        Execute a primitive step

        Paramters
        ---------
        n_step: int
            An integer of the actual step.
        primitive: PrimitiveBaseMeta
            A primitive class
        primitive_arguments
            Arguments for set_training_data, fit, produce of the primitive for this step.
        """
        model = self.pipeline[n_step]
        produce_params_primitive = self._primitive_arguments(primitive, 'produce')
        produce_params = {}

        for param, value in primitive_arguments.items():
            if param in produce_params_primitive:
                produce_params[param] = value

        python_path = primitive.metadata.query()['python_path']

        if model is not None:  # Use pre-learnt model
            return model.produce(**produce_params).value
        
        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        custom_hyperparams = dict()
 
        hyperparams = self.hyperparams[n_step]
        if hyperparams is not None:
            for hyperparam, value in hyperparams.items():
                custom_hyperparams[hyperparam] = value

        training_arguments_primitive = self._primitive_arguments(primitive, 'set_training_data')
        training_arguments = {}

        for param, value in primitive_arguments.items():
            if param in training_arguments_primitive:
                training_arguments[param] = value

        if 'ImageFeature.DSBOX' in python_path:
            volumes = self.static_dir + '/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
            model = primitive(hyperparams=primitive_hyperparams(
                    primitive_hyperparams.defaults(), **custom_hyperparams), volumes=volumes)
        else:
            model = primitive(hyperparams=primitive_hyperparams(
                    primitive_hyperparams.defaults(), **custom_hyperparams))

        model.set_training_data(**training_arguments)
        model.fit()
        self.pipeline[n_step] = model

        return model.produce(**produce_params).value

    def _primitive_arguments(self, primitive, method: str) -> set:
        """
        Get the arguments of a primitive given a function.

        Parameters
        ---------
        primitive
            A primitive.
        method
            A method of the primitive.
        """
        return set(primitive.metadata.query()['primitive_code']['instance_methods'][method]['arguments'])

    def produce(self, **arguments):
        """
        Run produce on the solution.

        Paramters
        ---------
        arguments
            Arguments required to execute the solution
        """
        steps_outputs = [None] * len(self.execution_order)

        logging.info("produce %s", self.primitives_arguments)
        logging.info("produce args %s", arguments)
        for i in range(0, len(self.execution_order)):
            n_step = self.execution_order[i]
            produce_arguments = {}

            if self.steptypes[n_step] is StepType.SUBPIPELINE:
                produce_arguments = {}
                for argument, value in self.primitives_arguments[n_step].items():
                    if value['origin'] == 'steps':
                        produce_arguments[argument] = steps_outputs[value['source']]
                    else:
                        produce_arguments[argument] = arguments['inputs'][value['source']]
                    if produce_arguments[argument] is None:
                        continue

            if self.steptypes[n_step] is StepType.PRIMITIVE:
                primitive = self.primitives[n_step]
                produce_arguments_primitive = self._primitive_arguments(primitive, 'produce')
                for argument, value in self.primitives_arguments[n_step].items():
                    if argument in produce_arguments_primitive:
                        if value['origin'] == 'steps':
                            produce_arguments[argument] = steps_outputs[value['source']]
                        else:
                            produce_arguments[argument] = arguments['inputs'][value['source']]
                        if produce_arguments[argument] is None:
                            continue

            if self.steptypes[n_step] is StepType.PRIMITIVE: # Primitive
                if n_step in self.produce_order:
                    v = self.pipeline[n_step].produce(**produce_arguments).value
                    steps_outputs[n_step] = v
                else:
                    steps_outputs[n_step] = None
            else: # Subpipeline
                solution_dict = arguments['solution_dict']
                solution = solution_dict[self.subpipelines[n_step].id]
                inputs = []
                inputs.append(produce_arguments['inputs'])
                steps_outputs[n_step] = solution.produce(inputs=inputs, solution_dict=solution_dict)

        # Create output
        pipeline_output = []
        for output in self.outputs:
            if output[0] == 'steps':
                pipeline_output.append(steps_outputs[output[1]])
            else:
                pipeline_output.append(arguments[output[0][output[1]]])
        return pipeline_output

    def initialize_solution(self, taskname):
        """
        Initialize a solution from scratch consisting of predefined steps
        Leave last step for filling in primitive (for classification/regression problems)
        """
        python_paths = copy.deepcopy(solution_templates.task_paths[taskname])

        if 'denormalize' in python_paths[0] and self.ok_to_denormalize == False:
            python_paths.remove('d3m.primitives.data_transformation.denormalize.Common')

        if len(python_paths) > 5 and 'imputer' in python_paths[5] and self.ok_to_impute == False:
            python_paths.remove('d3m.primitives.data_cleaning.imputer.SKlearn')
            if 'imputer' in python_paths[4] and self.ok_to_impute == False:
                python_paths.remove('d3m.primitives.data_cleaning.imputer.SKlearn')

        if (taskname == 'CLASSIFICATION' or taskname == 'REGRESSION' or taskname == 'TEXT' or taskname == 'IMAGE' or taskname == 'TIMESERIES'):
            if self.categorical_atts is not None and len(self.categorical_atts) > 0:
                index = len(python_paths)-1
                python_paths.insert(index, 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn')

            if taskname is not 'IMAGE':  # Image data frame has too many dimensions (in thousands)! This step is extremely slow! 
                index = len(python_paths)-1
                python_paths.insert(index, 'd3m.primitives.data_preprocessing.robust_scaler.SKlearn')

        num = len(python_paths)
        self.taskname = taskname
        self.primitives_arguments = {}
        self.subpipelines = {}
        self.primitives = {}
        self.hyperparams = {}
        self.steptypes = []
        self.pipeline = []
        self.execution_order = None

        self.inputs = []
        self.inputs.append({"name": "dataset inputs"})

        # Constructing DAG to determine the execution order
        execution_graph = nx.DiGraph()

        # Iterate through steps 
        for i in range(num):
            self.add_primitive(python_paths[i], i)

            if python_paths[i] == 'd3m.primitives.data_cleaning.imputer.SKlearn':
                self.hyperparams[i] = {}
                self.hyperparams[i]['use_semantic_types'] = True
                self.hyperparams[i]['return_result'] = 'replace'

            if python_paths[i] == 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn':
                self.hyperparams[i] = {}
                self.hyperparams[i]['use_semantic_types'] = True
                self.hyperparams[i]['return_result'] = 'replace'
                self.hyperparams[i]['handle_unknown'] = 'ignore'

            if python_paths[i] == 'd3m.primitives.data_preprocessing.robust_scaler.SKlearn':
                self.hyperparams[i] = {}
                self.hyperparams[i]['return_result'] = 'replace'

            if python_paths[i] == 'd3m.primitives.feature_construction.corex_text.CorexText':
                self.hyperparams[i] = {}
                self.hyperparams[i]['threshold'] = 500
           
            if python_paths[i] == 'd3m.primitives.link_prediction.graph_matching_link_prediction.GraphMatchingLinkPrediction':
                self.hyperparams[i] = {}
                custom_hyperparams = {}
                custom_hyperparams['prediction_column'] = 'match'
                self.hyperparams[i]['link_prediction_hyperparams'] = LinkPredictionHyperparams(LinkPredictionHyperparams.defaults(), **custom_hyperparams)

            if python_paths[i] == 'd3m.primitives.natural_language_processing.lda.Fastlvm':
                self.hyperparams[i] = {}
                self.hyperparams[i]['k'] = 200

            if self.privileged is not None and len(self.privileged) > 0 and python_paths[i] == 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon':
                self.hyperparams[i] = {}
                self.hyperparams[i]['exclude_columns'] = self.privileged

            if taskname == 'CLASSIFICATION' or \
                 taskname == 'REGRESSION' or \
                 taskname == 'TEXT' or \
                 taskname == 'IMAGE' or \
                 taskname == 'TIMESERIES':
                if i == 0: # denormalize
                    data = 'inputs.0'
                elif i == num-1: # extract_columns_by_semantic_types (targets)
                    data = 'steps.1.produce'
                    self.hyperparams[i] = {}
                    self.hyperparams[i]['semantic_types'] = ['https://metadata.datadrivendiscovery.org/types/TrueTarget']
                else: # other steps
                    data = 'steps.' + str(i-1) + '.produce'

            elif taskname == 'AUDIO':
                if i == 0 or i == 2:# or i == num-1: # denormalize or AudioReader or TargetsReader
                    data = 'inputs.0'
                elif i == num-1: # extract_columns_by_semantic_types (targets)
                    data = 'steps.1.produce'
                    self.hyperparams[i] = {}
                    self.hyperparams[i]['semantic_types'] = ['https://metadata.datadrivendiscovery.org/types/TrueTarget']
                else: # other steps
                    data = 'steps.' + str(i-1) + '.produce'
            elif taskname == 'CLUSTERING':
                if i == 0: # dataset_to_dataframe
                    data = 'inputs.0'
                elif i == num-1: # construct_predictions
                    data = 'steps.0.produce'
                    origin = data.split('.')[0]
                    source = data.split('.')[1]
                    self.primitives_arguments[i]['reference'] = {'origin': origin, 'source': int(source), 'data': data}
                    data = 'steps.' + str(i-1) + '.produce'
                else: # other steps
                    data = 'steps.' + str(i-1) + '.produce'
            else:
                if i == 0:
                    data = 'inputs.0'
                else:
                    data = 'steps.' + str(i-1) + '.produce'

            origin = data.split('.')[0]
            source = data.split('.')[1]
            self.primitives_arguments[i]['inputs'] = {'origin': origin, 'source': int(source), 'data': data}

            if i == 0:
                execution_graph.add_edge(origin, str(i))
            else:
                execution_graph.add_edge(str(source), str(i))

        execution_order = list(nx.topological_sort(execution_graph))

        # Removing non-step inputs from the order
        execution_order = list(filter(lambda x: x.isdigit(), execution_order))
        self.execution_order = [int(x) for x in execution_order]

    def add_primitive(self, python_path, i):
        """
        Helper function to add a primitive in the pipeline with basic initialization only.
        """
        self.primitives_arguments[i] = {}
        self.hyperparams[i] = None
        self.pipeline.append(None)
        prim = d3m.index.get_primitive(python_path)
        self.primitives[i] = prim
        self.steptypes.append(StepType.PRIMITIVE)
        self.subpipelines[i] = None

    def add_subpipeline(self, pipeline):
        """
        Helper function to add a subpipeline(replace placeholder step) in the pipeline with basic initialization only.
        """
        i = len(self.primitives_arguments)-1

        self.primitives_arguments[i] = {}
        self.hyperparams[i] = None
        self.pipeline.append(None)
        self.primitives[i] = None
        self.steptypes[i] = StepType.SUBPIPELINE
        self.subpipelines[i] = pipeline
       
        data = 'steps.' + str(i-1) + '.produce'
        origin = data.split('.')[0]
        source = data.split('.')[1] 
        self.primitives_arguments[i]['inputs'] = {'origin': origin, 'source': int(source), 'data': data}

        data = 'steps.' + str(i) + '.' + pipeline.outputs[0][2]
        origin = data.split('.')[0]
        source = data.split('.')[1]
        self.outputs = []
        self.outputs.append((origin, int(source), data, "output predictions"))
 
    def add_step(self, python_path):
        """
        Add new primitive
        This is currently for classification/regression pipelines
        """
        n_steps = len(self.primitives_arguments) + 1
        i = n_steps-1

        self.add_primitive(python_path, i)

        data = 'steps.' + str(i-2) + str('.produce')
        origin = data.split('.')[0]
        source = data.split('.')[1]
        self.primitives_arguments[i]['inputs'] = {'origin': origin, 'source': int(source), 'data': data}
        
        data = 'steps.' + str(i-1) + str('.produce')
        origin = data.split('.')[0]
        source = data.split('.')[1]
        self.primitives_arguments[i]['outputs'] = {'origin': origin, 'source': int(source), 'data': data}

        if 'SKlearn' in python_path:
            self.hyperparams[i] = {}
            if self.taskname is not 'IMAGE' and self.taskname is not 'TIMESERIES' and self.taskname is not 'AUDIO' and self.taskname is not 'TEXT': 
                self.hyperparams[i]['use_semantic_types'] = True
            hyperparam_spec = self.primitives[i].metadata.query()['primitive_code']['hyperparams']
            if 'n_estimators' in hyperparam_spec:
                self.hyperparams[i]['n_estimators'] = 100
            
        self.execution_order.append(i)

        i = i + 1
        self.add_primitive('d3m.primitives.data_transformation.construct_predictions.DataFrameCommon', i)

        data = 'steps.' + str(i-1) + str('.produce')
        origin = data.split('.')[0]
        source = data.split('.')[1]
        self.primitives_arguments[i]['inputs'] = {'origin': origin, 'source': int(source), 'data': data}
        
        data = 'steps.' + str(1) + str('.produce')
        origin = data.split('.')[0]
        source = data.split('.')[1]
        self.primitives_arguments[i]['reference'] = {'origin': origin, 'source': int(source), 'data': data}

        self.execution_order.append(i)

        self.add_outputs()

    def add_outputs(self): 
        """
        Add outputs as last step for pipeline
        Also compute produce order.
        """
        n_steps = len(self.execution_order)
        
        self.outputs = []

        data = 'steps.' + str(n_steps-1) + '.produce'

        origin = data.split('.')[0]
        source = data.split('.')[1]
        self.outputs.append((origin, int(source), data, "output predictions"))

        # Creating set of steps to be call in produce
        self.produce_order = set()

        current_step = int(source)
        self.produce_order.add(current_step)
        for i in range(0, len(self.execution_order)):
            step_origin = self.primitives_arguments[current_step]['inputs']['origin']
            step_source = self.primitives_arguments[current_step]['inputs']['source']
            if step_origin != 'steps':
                 break
            else:
                self.produce_order.add(step_source)
                current_step = step_source

    def process_step(self, n_step, primitives_outputs, action, arguments):
        """
        Process each step of a pipeline
        This could be used while scoring, validating or fitting a pipeline
        """
        # Subpipeline step
        if self.steptypes[n_step] is StepType.SUBPIPELINE:
            primitive_arguments = {}
            for argument, value in self.primitives_arguments[n_step].items():
                logging.info("Arg = %s", argument)
                logging.info("Value = %s", value)
                if value['origin'] == 'steps':
                    primitive_arguments[argument] = primitives_outputs[value['source']]
                else:
                    primitive_arguments[argument] = arguments['inputs'][value['source']]
            return self._pipeline_step_fit(n_step, self.subpipelines[n_step].id, primitive_arguments, arguments, action)

        # Primitive step
        if self.steptypes[n_step] is StepType.PRIMITIVE:
            primitive_arguments = {}
            for argument, value in self.primitives_arguments[n_step].items():
                if value['origin'] == 'steps':
                    primitive_arguments[argument] = primitives_outputs[value['source']]
                else:
                    primitive_arguments[argument] = arguments['inputs'][value['source']]
            if action is ActionType.SCORE and self.is_last_step(n_step) == True:
                primitive = self.primitives[n_step]
                primitive_desc = arguments['primitive_dict'][primitive]
                return self.score_step(primitive, primitive_arguments, arguments['metric'], arguments['posLabel'], primitive_desc, self.hyperparams[n_step], n_step)
            elif action is ActionType.VALIDATE and self.is_last_step(n_step) == True:
                return self.validate_step(self.primitives[n_step], primitive_arguments)    
            else:
                v = self.fit_step(n_step, self.primitives[n_step], primitive_arguments)
                return v

        # Placeholder step
        if self.steptypes[n_step] is StepType.PLACEHOLDER:
            return primitives_outputs[n_step-1]
 
    def is_last_step(self, n):
        last_step = self.get_last_step()
        if n == last_step:
            return True
        return False

    def get_last_step(self):
        """
        Return index of step which needs to be optimized/cross-validated.
        Typically this is the step for a classifier/regressor which can be optimized/cross-validated.
        This is the last-1 step of the pipeline, since it is followed by construct_predictions primitive to construct predictions output from d3mIndex and primitive output.
        """
        n_steps = len(self.primitives_arguments)

        if self.steptypes[n_steps-1] is StepType.SUBPIPELINE:
            return n_steps - 1

        if self.primitives[n_steps-1] is None:
            return n_steps - 1

        if 'construct_predictions' in self.primitives[n_steps-1].metadata.query()['python_path']:
            return n_steps - 2
        else:
            return n_steps - 1

    def score_solution(self, **arguments):
        """
        Score a solution 
        """
        score = 0.0
        primitives_outputs = [None] * len(self.execution_order)

        last_step = self.get_last_step()

        if self.primitives_outputs is None:
            for i in range(0, last_step+1): 
                n_step = self.execution_order[i]
                primitives_outputs[n_step] = self.process_step(n_step, primitives_outputs, ActionType.SCORE, arguments)

                if self.isDataFrameStep(n_step) == True:
                    self.exclude(primitives_outputs[n_step])
        else:
            primitives_outputs[last_step] = self.process_step(last_step, self.primitives_outputs, ActionType.SCORE, arguments)

        (score, optimal_params) = primitives_outputs[last_step]
        self.hyperparams[last_step] = optimal_params

        return (score, optimal_params)

    def set_hyperparams(self, hp):
        """
        Set hyperparameters for the primtiive at the "last" step.
        """
        n_step = self.get_last_step()
        self.hyperparams[n_step] = hp
        self.pipeline_description = None #Recreate it again

    def validate_solution(self,**arguments):
        """
        Validate a solution 
        """
 
        valid = False
        primitives_outputs = [None] * len(self.execution_order)
        
        last_step = self.get_last_step()

        if self.primitives_outputs is None:
            for i in range(0, last_step+1):
                n_step = self.execution_order[i]
                primitives_outputs[n_step] = self.process_step(n_step, primitives_outputs, ActionType.VALIDATE, arguments)

                if self.isDataFrameStep(n_step) == True:
                    self.exclude(primitives_outputs[n_step])
        else:
            primitives_outputs[last_step] = self.process_step(last_step, self.primitives_outputs, ActionType.VALIDATE, arguments)

        valid = primitives_outputs[last_step]
        return valid

    def run_basic_solution(self, **arguments):
        """
        Run common parts of a pipeline before adding last step of classifier/regressor
        This saves on data processing, featurizing steps being repeated across multiple pipelines.
        """
        from timeit import default_timer as timer
        self.primitives_outputs = [None] * len(self.execution_order)

        for i in range(0, len(self.execution_order)):
            n_step = self.execution_order[i]
            python_path = self.primitives[n_step].metadata.query()['python_path']
            
            if python_path == 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn':
                cols = get_cols_to_encode(self.primitives_outputs[n_step-1])
                self.hyperparams[n_step]['use_columns'] = list(cols)
                print("Cats = ", cols)

            if self.exclude_columns is not None and len(self.exclude_columns) > 0:
                if python_path == 'd3m.primitives.data_transformation.column_parser.DataFrameCommon' or \
                   python_path == 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon':
                    if self.hyperparams[n_step] is None:
                        self.hyperparams[n_step] = {}
                    self.hyperparams[n_step]['exclude_columns'] = list(self.exclude_columns)

            logging.info("Running %s", python_path) 
            start = timer()
            self.primitives_outputs[n_step] = self.process_step(n_step, self.primitives_outputs, ActionType.FIT, arguments)
            end = timer()
            logging.info("Time taken : %s seconds", end - start)

            if self.isDataFrameStep(n_step) == True:
                self.exclude(self.primitives_outputs[n_step])
        
        # Remove other intermediate step outputs, they are not needed anymore.
        for i in range(0, len(self.execution_order)-2):
            if i == 1:
                continue
            self.primitives_outputs[i] = [None]
        self.total_cols = len(self.primitives_outputs[len(self.execution_order)-2].columns) 

    def get_total_cols(self):
        return self.total_cols

    def score_step(self, primitive: PrimitiveBaseMeta, primitive_arguments, metric, posLabel, primitive_desc, hyperparams, step_index):
        """
        Last step of a solution evaluated for score_solution()
        Does hyperparameters tuning
        """
        training_arguments_primitive = self._primitive_arguments(primitive, 'set_training_data')
        training_arguments = {}

        custom_hyperparams = dict()
        if hyperparams is not None:
            for hyperparam, value in hyperparams.items():
                custom_hyperparams[hyperparam] = value

        for param, value in primitive_arguments.items():
            if param in training_arguments_primitive:
                training_arguments[param] = value

        outputs = None
        if 'outputs' in training_arguments:
            outputs = training_arguments['outputs']

        if len(training_arguments) == 0:
            training_arguments['inputs'] = primitive_arguments['inputs']

        (score, optimal_params) = primitive_desc.score_primitive(training_arguments['inputs'], outputs, metric, posLabel, custom_hyperparams, step_index)
        return (score, optimal_params) 

    def validate_step(self, primitive: PrimitiveBaseMeta, primitive_arguments):
        """
        Last step of a solution evaluated for validate_solution()
        """
        python_path = primitive.metadata.query()['python_path']
        if 'Find_projections' in python_path:
            return True

        family = primitive.metadata.query()['primitive_family']

        training_arguments_primitive = self._primitive_arguments(primitive, 'set_training_data')
        training_arguments = {}

        for param, value in primitive_arguments.items():
            if param in training_arguments_primitive:
                training_arguments[param] = value

        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        model = primitive(hyperparams=primitive_hyperparams(
                            primitive_hyperparams.defaults()))

        if family is not PrimitiveFamily.DATA_TRANSFORMATION and 'd3m.primitives.sri.' not in primitive.metadata.query()['python_path']:
            ip = training_arguments['inputs']
            from sklearn.model_selection import KFold
        
            # Train on just 20% of the data to validate
            kf = KFold(n_splits=5, shuffle=True, random_state=9001)
            newtrain_args = {}
            for train_index, test_index in kf.split(ip):
                for param, value in training_arguments.items():
                    newvalue = pd.DataFrame(data=value.values)
                    newtrain_args[param] = newvalue.iloc[test_index]
                    newtrain_args[param].metadata = value.metadata
                break
            training_arguments = newtrain_args
 
        model.set_training_data(**training_arguments)
        model.fit()

        return True
 
    def describe_solution(self, prim_dict):
        """
        Required for TA2-TA3 API DescribeSolution().
        """
        inputs = []
        for i in range(len(self.inputs)):
            inputs.append(pipeline_pb2.PipelineDescriptionInput(name=self.inputs[i]["name"]))

        outputs=[]
        outputs.append(pipeline_pb2.PipelineDescriptionOutput(name="predictions", data=self.outputs[0][2]))

        steps=[]
        
        for j in range(len(self.primitives_arguments)):
            s = self.primitives[j]
            prim = prim_dict[s]
            p = primitive_pb2.Primitive(id=prim.id, version=prim.primitive_class.version, python_path=prim.primitive_class.python_path,
            name=prim.primitive_class.name, digest=prim.primitive_class.digest)

            arguments={}

            for argument, data in self.primitives_arguments[j].items():
                argument_edge = data['data']
                origin = argument_edge.split('.')[0]
                source = argument_edge.split('.')[1]
                
                sa = pipeline_pb2.PrimitiveStepArgument(container = pipeline_pb2.ContainerArgument(data=argument_edge))
                arguments[argument] = sa

            step_outputs = []
            for a in prim.primitive_class.produce_methods:
                step_outputs.append(pipeline_pb2.StepOutput(id=a))
            steps.append(pipeline_pb2.PipelineDescriptionStep(primitive=pipeline_pb2.PrimitivePipelineDescriptionStep(primitive=p,
             arguments=arguments, outputs=step_outputs)))
           
        return pipeline_pb2.PipelineDescription(id=self.id, source=self.source, created=self.created, context=self.context,
         name=self.name, description=self.description, inputs=inputs, outputs=outputs, steps=steps)

    def get_hyperparams(self, step, prim_dict):
        """
        Required for TA2-TA3 API DescribeSolution().
        """
        p = prim_dict[self.primitives[step]]
        custom_hyperparams = self.hyperparams[step]

        send_params={}
        if 'hyperparams' in p.primitive.metadata.query()['primitive_code']:
            hyperparam_spec = p.primitive.metadata.query()['primitive_code']['hyperparams']
            filter_hyperparam = lambda vl: None if vl == 'None' else vl
            hyperparams = {name:filter_hyperparam(vl['default']) for name,vl in hyperparam_spec.items()}

            if custom_hyperparams is not None:
                for name, value in custom_hyperparams.items():
                    hyperparams[name] = value

            hyperparam_types = {name:filter_hyperparam(vl['structural_type']) for name,vl in hyperparam_spec.items() if 'structural_type' in vl.keys()}
        
            for name, value in hyperparams.items():
                tp = hyperparam_types[name]
                if tp is int:
                    send_params[name]=value_pb2.Value(raw=value_pb2.ValueRaw(int64=value))
                elif tp is float:
                    send_params[name]=value_pb2.Value(raw=value_pb2.ValueRaw(double=value))
                elif tp is bool:
                    send_params[name]=value_pb2.Value(raw=value_pb2.ValueRaw(bool=value))
                elif tp is str:
                    send_params[name]=value_pb2.Value(raw=value_pb2.ValueRaw(string=value))
                else:
                    if isinstance(value, int):
                        send_params[name]=value_pb2.Value(raw=value_pb2.ValueRaw(int64=value))
                    elif isinstance(value, float):
                        send_params[name]=value_pb2.Value(raw=value_pb2.ValueRaw(double=value))
                    elif isinstance(value, bool):
                        send_params[name]=value_pb2.Value(raw=value_pb2.ValueRaw(bool=value))
                    elif isinstance(value, str):
                        send_params[name]=value_pb2.Value(raw=value_pb2.ValueRaw(string=value))
           
        return core_pb2.PrimitiveStepDescription(hyperparams=send_params)
