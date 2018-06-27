"""
This should be where everything comes together: Problem descriptions get matched up
with the available primitives, and a plan for how to create a solution gets made.

    So it needs to:
    Choose hyperparameters for a primitive
    Run the primitive
    Measure the results
    Feed the results and hyperparameters back into the chooser
"""

import importlib

import logging
import core_pb2, problem_pb2, pipeline_pb2, primitive_pb2, value_pb2
import pandas as pd

from  api_v3 import core

import uuid, sys, math
import time
from enum import Enum
from time import sleep
from google.protobuf.timestamp_pb2 import Timestamp

from sklearn import metrics
from sklearn import preprocessing

from d3m.metadata.pipeline import Pipeline, PrimitiveStep, SubpipelineStep, ArgumentType, PipelineContext
from d3m.metadata.base import PrimitiveFamily
from d3m.primitive_interfaces.base import PrimitiveBaseMeta
import d3m.index

import networkx as nx

import bo.gp_call

task_paths = {'CLASSIFICATION': ['d3m.primitives.datasets.DatasetToDataFrame','d3m.primitives.data.ExtractColumnsBySemanticTypes', 'd3m.primitives.data.ExtractColumnsBySemanticTypes'], 
'REGRESSION': ['d3m.primitives.datasets.DatasetToDataFrame','d3m.primitives.data.ExtractColumnsBySemanticTypes', 'd3m.primitives.data.ExtractColumnsBySemanticTypes'],
#'COLLABORATIVEFILTERING': ['d3m.primitives.sri.graph.CollaborativeFilteringParser', 'd3m.primitives.sri.graph.GraphTransformer', 'd3m.primitives.sri.psl.LinkPrediction']}
'COLLABORATIVEFILTERING': ['d3m.primitives.sri.psl.CollaborativeFilteringLinkPrediction'],
'VERTEXNOMINATION': ['d3m.primitives.sri.graph.VertexNominationParser', 'd3m.primitives.sri.psl.VertexNomination']}
 
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

class SolutionDescription(object):
    """
    A wrapper of a primitive instance and hyperparameters, ready to have inputs
    fed into it.

    The idea is that this can be evaluated, produce a model and performance metrics,
    and the hyperparameter tuning can consume that and choose what to do next.

    Output is fairly basic right now; it writes to a single numpy CSV file with a given name
    based off the results of the primitive (numpy arrays only atm)
    """
    def __init__(self, problem):
        self.id = str(uuid.uuid4())
        self.source = None
        self.created = compute_timestamp()
        self.context = pipeline_pb2.PRETRAINING
        self.name = None
        self.description = None
        self.users = None
        self.inputs = []
        self.rank = -1
        self.indices = None
        
        self.outputs = None
        self.execution_order = None
        self.primitives_arguments = None
        self.primitives = None
        self.pipeline = None
        self.produce_order = None
        self.hyperparams = None
        self.problem = problem
        self.steptypes = None
        self.le = None
        self.taskname = None

    def contains_placeholder(self):
        if bool(self.steptypes) == False:
            return False

        for step in self.steptypes:
            if step == StepType.PLACEHOLDER:
                return True
        return False

    def num_steps(self):
        if bool(self.primitives_arguments):
            return len(self.primitives_arguments)
        else:
            return 0

    def create_pipeline_json(self, prim_dict, filename):
        pipeline_description = Pipeline(pipeline_id=self.id, context=PipelineContext.EVALUATION)
        for ip in self.inputs:
            pipeline_description.add_input(name=ip['name'])

        num = self.num_steps()
        for i in range(num):
            p = prim_dict[self.primitives[i]]
            pdesc = {}
            pdesc['id'] = p.id
            pdesc['version'] = p.primitive_class.version
            pdesc['python_path'] = p.primitive_class.python_path
            pdesc['name'] = p.primitive_class.name
            pdesc['digest'] = p.primitive_class.digest
            step = PrimitiveStep(primitive_description=pdesc)

            for name, value in self.primitives_arguments[i].items():
                origin = value['origin']
                if origin == 'steps':
                    argument_type = ArgumentType.DATA
                else:
                    argument_type = ArgumentType.CONTAINER
                step.add_argument(name=name, argument_type=argument_type, data_reference=value['data'])
            step.add_output(output_id=p.primitive_class.produce_methods[0])
            if bool(self.hyperparams[i]):
                for name, value in self.hyperparams[i].items():
                    step.add_hyperparameter(name=name, argument_type=ArgumentType.VALUE, data=value)
            pipeline_description.add_step(step)

        for op in self.outputs:
            pipeline_description.add_output(data_reference=op[2], name=op[3])

        outfile = open(filename, "w")
        pipeline_description.to_json(outfile)
        outfile.close()

    def create_from_pipeline(self, pipeline_description: Pipeline) -> None:
        n_steps = len(pipeline_description.steps)

        self.inputs = pipeline_description.inputs
        self.id = pipeline_description.id
        self.source = pipeline_description.source
        self.name = pipeline_description.name
        self.description = pipeline_description.description
        self.users = pipeline_description.users

        self.primitives_arguments = {}
        self.primitives = {}
        self.hyperparams = {}
        self.steptypes = []
        for i in range(0, n_steps):
            self.primitives_arguments[i] = {}
            self.hyperparams[i] = None

        self.execution_order = None

        self.pipeline = [None] * n_steps

        # Constructing DAG to determine the execution order
        execution_graph = nx.DiGraph()
        for i in range(0, n_steps):
            if isinstance(pipeline_description.steps[i], PrimitiveStep):
                self.steptypes.append(StepType.PRIMITIVE)
                self.primitives[i] = pipeline_description.steps[i].primitive
                for argument, data in pipeline_description.steps[i].arguments.items():
                    argument_edge = data['data']
                    origin = argument_edge.split('.')[0]
                    source = argument_edge.split('.')[1]

                    self.primitives_arguments[i][argument] = {'origin': origin, 'source': int(source), 'data': argument_edge}

                    if origin == 'steps':
                        execution_graph.add_edge(str(source), str(i))
                    else:
                        execution_graph.add_edge(origin, str(i))

                hyperparams = pipeline_description.steps[i].hyperparams
                if bool(hyperparams):
                    self.hyperparams[i] = {}
                    for name,argument in hyperparams.items():
                        self.hyperparams[i][name] = argument['data']
            elif isinstance(pipeline_description.steps[i], SubpipelineStep):  
                self.steptypes.append(StepType.SUBPIPELINE)
                self.primitives[i] = pipeline_description.steps[i].pipeline_id
                self.primitives_arguments[i] = []
                for j in range(len(pipeline_description.steps[i].inputs)):
                    argument_edge = pipeline_description.steps[i].inputs[j]
                    origin = argument_edge.split('.')[0]
                    source = argument_edge.split('.')[1]
                    self.primitives_arguments[i].append({'origin': origin, 'source': int(source), 'data': argument_edge})
                    
                    if origin == 'steps':
                        execution_graph.add_edge(str(source), str(i))
                    else:
                        execution_graph.add_edge(origin, str(i))
            else:
                print("NOT IMPLEMENTED!!!")

        execution_order = list(nx.topological_sort(execution_graph))

        # Removing non-step inputs from the order
        execution_order = list(filter(lambda x: x.isdigit(), execution_order))
        self.execution_order = [int(x) for x in execution_order]

        # Creating set of steps to be call in produce
        self.outputs = []
        self.produce_order = set()
        for output in pipeline_description.outputs:
            origin = output['data'].split('.')[0]
            source = output['data'].split('.')[1]
            self.outputs.append((origin, int(source), output['data'], output['name']))

            if origin != 'steps':
                continue
            else:
                current_step = int(source)
                self.produce_order.add(current_step)
                for i in range(0, len(execution_order)):
                    if self.steptypes[current_step] == StepType.PRIMITIVE:
                        step_origin = self.primitives_arguments[current_step]['inputs']['origin']
                        step_source = self.primitives_arguments[current_step]['inputs']['source']
                    else:
                        step_origin = self.primitives_arguments[current_step][0]['origin']
                        step_source = self.primitives_arguments[current_step][0]['source']
                    if step_origin != 'steps':
                        break
                    else:
                        self.produce_order.add(step_source)
                        current_step = step_source

    def create_from_pipelinedescription(self, pipeline_description: pipeline_pb2.PipelineDescription) -> None:
        """
        Initialize a solution object from a pipeline_pb2.PipelineDescription object passed by TA3.
        """
        n_steps = len(pipeline_description.steps)

        print("Steps = ", n_steps)
        self.inputs = pipeline_description.inputs
        self.source = pipeline_description.source
        self.created = pipeline_description.created
        self.name = pipeline_description.name
        self.description = pipeline_description.description
        self.users = pipeline_description.users

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
                prim = d3m.index.search(primitive_path_prefix=python_path)[python_path]
                self.primitives[i] = prim
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
                if bool(hyperparams):
                    self.hyperparams[i] = {}
                    for name,argument in hyperparams.items():
                        self.hyperparams[i][name] = argument['data'] 

            # SubpipelinePipelineDescriptionStep
            elif pipeline_description.steps[i].HasField("pipeline") == True:
                s = pipeline_description.steps[i].pipeline
                self.primitives[i] = s
                self.steptypes.append(StepType.SUBPIPELINE)
                self.primitives_arguments[i] = []
                for j in range(len(pipeline_description.steps[i].inputs)):
                    argument_edge = pipeline_description.steps[i].inputs[j].data
                    origin = argument_edge.split('.')[0]
                    source = argument_edge.split('.')[1]
                    self.primitives_arguments[i].append({'origin': origin, 'source': int(source), 'data': argument_edge})

                    if origin == 'steps':
                        execution_graph.add_edge(str(source), str(i))
                    else:
                        execution_graph.add_edge(origin, str(i))

            else: # PlaceholderPipelineDescriptionStep
                s = pipeline_description.steps[i].placeholder
                self.steptypes.append(StepType.PLACEHOLDER)
                self.primitives_arguments[i] = []
                for j in range(len(pipeline_description.steps[i].inputs)):
                    argument_edge = pipeline_description.steps[i].inputs[j].data
                    origin = argument_edge.split('.')[0]
                    source = argument_edge.split('.')[1]
                    self.primitives_arguments[i].append({'origin': origin, 'source': int(source), 'data': argument_edge})

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

    def isDataFrameStep(self, n_step):
        if self.steptypes[n_step] is StepType.PRIMITIVE and self.primitives[n_step].metadata.query()['python_path'] == 'd3m.primitives.datasets.DatasetToDataFrame':
            return True
        return False

    def fit(self, **arguments):
        """
        Train all steps in the solution.

        Paramters
        ---------
        arguments
            Arguments required to train the solution
        """
        primitives_outputs = [None] * len(self.execution_order)
        for i in range(0, len(self.execution_order)):
            n_step = self.execution_order[i]
        
            primitives_outputs[n_step] = self.process_step(n_step, primitives_outputs, ActionType.FIT, arguments)
            if self.isDataFrameStep(n_step) == True:
                self.indices = primitives_outputs[n_step][['d3mIndex']] 
        v = primitives_outputs[len(self.execution_order)-1]
        return self.invert_output(v)

    def _pipeline_step_fit(self, n_step: int, pipeline_id: str, primitive_arguments, solution_dict, primitive_dict, action):
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
        solution = solution_dict[pipeline_id]

        inputs = []
        for i in range(len(primitive_arguments)):
            inputs.append(primitive_arguments[i])

        if action is ActionType.FIT: 
            return solution.fit(inputs=inputs, solution_dict=solution_dict)
        elif action is ActionType.SCORE:
            return solution.score_solution(inputs=inputs, solution_dict=solution_dict, primitive_dict=primitive_dict)
        else:
            return solution.validate_solution(inputs=inputs, solution_dict=solution_dict)
 
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
        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        custom_hyperparams = dict()
  
        hyperparams = self.hyperparams[n_step]
        if bool(hyperparams):
            for hyperparam, value in hyperparams.items():
                if isinstance(value, dict):
                    custom_hyperparams[hyperparam] = value['data']
                else:
                    custom_hyperparams[hyperparam] = value

        training_arguments_primitive = self._primitive_arguments(primitive, 'set_training_data')
        training_arguments = {}
        produce_params_primitive = self._primitive_arguments(primitive, 'produce')
        produce_params = {}

        for param, value in primitive_arguments.items():
            if param in produce_params_primitive:
                produce_params[param] = value
            if param in training_arguments_primitive:
                training_arguments[param] = value

        model = primitive(hyperparams=primitive_hyperparams(
                    primitive_hyperparams.defaults(), **custom_hyperparams))

        model.set_training_data(**training_arguments)
        model.fit()
        self.pipeline[n_step] = model
        return model.produce(**produce_params).value

    def _primitive_arguments(self, primitive, method: str) -> set:
        """
        Get the arguments of a primitive given a function.

        Paramters
        ---------
        primitive
            A primitive.
        method
            A method of the primitive.
        """
        return set(primitive.metadata.query()['primitive_code']['instance_methods'][method]['arguments'])

    def transform_data(self, primitive, v):
        path = primitive.metadata.query()['python_path']
        if path == 'd3m.primitives.data.ExtractColumnsBySemanticTypes':
            if len(v.columns) > 1:
                v = v.fillna('0').replace('', '0')
                v = v.apply(pd.to_numeric,errors="ignore")
                v = v.select_dtypes(['number'])
                return v
            else:
                if self.taskname == 'CLASSIFICATION':
                    self.le = preprocessing.LabelEncoder()
                    v = pd.DataFrame(self.le.fit_transform(v.values.ravel()))
                    return v

        return v

    def invert_output(self, v):
        if self.le is not None:
            if isinstance(v, pd.DataFrame):
                values = v.iloc[:,0].tolist()
            else:
                values = v
            inverted_op = self.le.inverse_transform(values)
            return inverted_op
        return v
 
    def produce(self, **arguments):
        """
        Run produce on the solution.

        Paramters
        ---------
        arguments
            Arguments required to execute the solution
        """
        steps_outputs = [None] * len(self.execution_order)

        for i in range(0, len(self.execution_order)):
            n_step = self.execution_order[i]
            produce_arguments = {}

            if self.steptypes[n_step] is StepType.SUBPIPELINE:
                produce_arguments = []
                for j in range(len(self.primitives_arguments[n_step])):
                    value = self.primitives_arguments[n_step][j]
                    if value['origin'] == 'steps':
                        produce_arguments.append(steps_outputs[value['source']])
                    else:
                        produce_arguments.append(arguments['inputs'][value['source']])
                    v = produce_arguments[len(produce_arguments)-1]
                    if v is None:
                        continue
                    produce_arguments[len(produce_arguments)-1] = v

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

            if self.steptypes[n_step] is StepType.PRIMITIVE:
                if n_step in self.produce_order:
                    v = self.pipeline[n_step].produce(**produce_arguments).value
                    v = self.transform_data(primitive, v)
                    steps_outputs[n_step] = v
                    if self.isDataFrameStep(n_step) == True:
                        self.indices = steps_outputs[n_step][['d3mIndex']]
                else:
                    steps_outputs[n_step] = None
            else:
                solution_dict = arguments['solution_dict']
                solution = solution_dict[self.primitives[n_step]]
                steps_outputs[n_step] = solution.produce(inputs=produce_arguments, solution_dict=solution_dict)

        # Create output
        pipeline_output = []
        for output in self.outputs:
            if output[0] == 'steps':
                pipeline_output.append(self.invert_output(steps_outputs[output[1]]))
            else:
                pipeline_output.append(arguments[output[0][output[1]]])
        return pipeline_output

    def initialize_solution(self, taskname):
        """
        Initialize a solution from scratch consisting of predefined steps
        Leave last step for filling in primitive
        """
        python_paths = task_paths[taskname]
        num = len(python_paths)

        self.taskname = taskname
        self.primitives_arguments = {}
        self.primitives = {}
        self.hyperparams = {}
        self.steptypes = []
        for i in range(0, num):
            self.primitives_arguments[i] = {}
            self.hyperparams[i] = None

        self.execution_order = None

        self.pipeline = [None] * num
        self.inputs = []
        self.inputs.append({"name": "dataset inputs"})

        # Constructing DAG to determine the execution order
        execution_graph = nx.DiGraph()
  
        for i in range(num):
            prim = d3m.index.get_primitive(python_paths[i])
            self.primitives[i] = prim          
            self.steptypes.append(StepType.PRIMITIVE)

            if i == 0:
                data = 'inputs.0'
            else:
                data = 'steps.0.produce'
            
            origin = data.split('.')[0]
            source = data.split('.')[1]
            self.primitives_arguments[i]['inputs'] = {'origin': origin, 'source': int(source), 'data': data}

            if taskname == 'VERTEXNOMINATION' and i == 1:
                self.primitives_arguments[i]['outputs'] = {'origin': origin, 'source': int(source), 'data': data}

            if taskname == 'CLASSIFICATION' or taskname == 'REGRESSION':
                if i == num-1:
                    self.hyperparams[i] = {}
                    self.hyperparams[i]['semantic_types'] = ['https://metadata.datadrivendiscovery.org/types/Target']
            
                if i == 0:
                    execution_graph.add_edge(origin, str(i))
                else:
                    execution_graph.add_edge(str(source), str(i))
            else:
                if i == 0:
                    execution_graph.add_edge(origin, str(i))
                else:
                    execution_graph.add_edge(str(i-1), str(i))

        execution_order = list(nx.topological_sort(execution_graph))

        # Removing non-step inputs from the order
        execution_order = list(filter(lambda x: x.isdigit(), execution_order))
        self.execution_order = [int(x) for x in execution_order]

    def add_step(self, python_path):
        """
        Add new primitive (or replace placeholder)
        """
        n_steps = len(self.primitives_arguments) + 1
        i = n_steps-1

        placeholder_present = False
        for j in range(len(self.steptypes)):
            if self.steptypes[j] == StepType.PLACEHOLDER:
                i = j
                placeholder_present = True
                break

        if placeholder_present == False:
            self.steptypes.append(StepType.PRIMITIVE)

        self.primitives_arguments[i] = {}
        self.hyperparams[i] = None

        self.pipeline.append([None])

        prim = d3m.index.get_primitive(python_path)
        self.primitives[i] = prim

        data = 'steps.' + str(1) + str('.produce')
        origin = data.split('.')[0]
        source = data.split('.')[1]
        self.primitives_arguments[i]['inputs'] = {'origin': origin, 'source': int(source), 'data': data}
        if i > 2:
            data = 'steps.' + str(2) + str('.produce')
            origin = data.split('.')[0]
            source = data.split('.')[1]
            self.primitives_arguments[i]['outputs'] = {'origin': origin, 'source': int(source), 'data': data}
            
        self.execution_order.append(i)
        self.add_outputs()

    def add_outputs(self):
        n_steps = len(self.execution_order)
        
        self.outputs = []
        # Creating set of steps to be call in produce
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
        # Subpipeline step
        if self.steptypes[n_step] is StepType.SUBPIPELINE:
            primitive_arguments = []
            for j in range(len(self.primitives_arguments[n_step])):
                value = self.primitives_arguments[n_step][j]
                if value['origin'] == 'steps':
                    primitive_arguments.append(primitives_outputs[value['source']])
                else:
                    primitive_arguments.append(arguments['inputs'][value['source']])
            return self._pipeline_step_fit(n_step, self.primitives[n_step], primitive_arguments,
 arguments['solution_dict'], arguments['primitive_dict'], action)

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
                return self.score_step(primitive, primitive_arguments, arguments['metric'], primitive_desc, self.hyperparams[n_step])
            elif action is ActionType.VALIDATE and self.is_last_step(n_step) == True:
                return self.validate_step(self.primitives[n_step], primitive_arguments)    
            else:
                v = self.fit_step(n_step, self.primitives[n_step], primitive_arguments)
                v = self.transform_data(self.primitives[n_step], v)    
                return v
 
    def is_last_step(self, n):
        if n == len(self.execution_order)-1:
            return True
        return False

    def score_solution(self, **arguments):
        """
        Score a solution 
        """
        score = 0.0
        primitives_outputs = [None] * len(self.execution_order)
     
        for i in range(0, len(self.execution_order)): 
            n_step = self.execution_order[i]
            primitives_outputs[n_step] = self.process_step(n_step, primitives_outputs, ActionType.SCORE, arguments)

        (score, optimal_params) = primitives_outputs[len(self.execution_order)-1]
        self.hyperparams[len(self.execution_order)-1] = optimal_params

        return score

    def validate_solution(self,**arguments):
        """
        Validate a solution 
        """
 
        valid = False
        primitives_outputs = [None] * len(self.execution_order)

        for i in range(0, len(self.execution_order)):
            n_step = self.execution_order[i]
            primitives_outputs[n_step] = self.process_step(n_step, primitives_outputs, ActionType.VALIDATE, arguments)

        valid = primitives_outputs[len(self.execution_order)-1]
        return valid
 
    def score_step(self, primitive: PrimitiveBaseMeta, primitive_arguments, metric, primitive_desc, hyperparams):
        """
        Last step of a solution evaluated for score_solution()
        Does hyperparameters tuning
        """
        training_arguments_primitive = self._primitive_arguments(primitive, 'set_training_data')
        training_arguments = {}

        custom_hyperparams = dict()
        if bool(hyperparams):
            for hyperparam, value in hyperparams.items():
                if isinstance(value, dict):
                    custom_hyperparams[hyperparam] = value['data']
                else:
                    custom_hyperparams[hyperparam] = value

        for param, value in primitive_arguments.items():
            if param in training_arguments_primitive:
                training_arguments[param] = value

        outputs = None
        if 'outputs' in training_arguments:
            outputs = training_arguments['outputs']
        (score, optimal_params) = primitive_desc.score_primitive(training_arguments['inputs'], outputs, metric, custom_hyperparams)
        return (score, optimal_params) 

    def validate_step(self, primitive: PrimitiveBaseMeta, primitive_arguments):
        """
        Last step of a solution evaluated for validate_solution()
        """
        family = primitive.metadata.query()['primitive_family']

        training_arguments_primitive = self._primitive_arguments(primitive, 'set_training_data')
        training_arguments = {}

        for param, value in primitive_arguments.items():
            if param in training_arguments_primitive:
                training_arguments[param] = value

        primitive_hyperparams = primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        model = primitive(hyperparams=primitive_hyperparams(
                            primitive_hyperparams.defaults()))

        if family is not PrimitiveFamily.DATA_TRANSFORMATION:
            ip = training_arguments['inputs']
            from sklearn.model_selection import KFold
            # Train on just 20% of the data to validate
            kf = KFold(n_splits=5, shuffle=True, random_state=9001)
            newtrain_args = {}
            for train_index, test_index in kf.split(ip):
                for param, value in training_arguments.items():
                    newtrain_args[param] = value.iloc[test_index]
                break
            training_arguments = newtrain_args
 
        model.set_training_data(**training_arguments)
        model.fit()

        return True
 
    def describe_solution(self, prim_dict):
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
                
                if origin == 'steps':
                    sa = pipeline_pb2.PrimitiveStepArgument(data = pipeline_pb2.DataArgument(data=argument_edge))
                else:
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
        p = prim_dict[self.primitives[step]]
        custom_hyperparams = self.hyperparams[step]

        send_params={}
        if 'hyperparams' in p.primitive.metadata.query()['primitive_code']:
            hyperparam_spec = p.primitive.metadata.query()['primitive_code']['hyperparams']
            filter_hyperparam = lambda vl: None if vl == 'None' else vl
            hyperparams = {name:filter_hyperparam(vl['default']) for name,vl in hyperparam_spec.items()}

            if bool(custom_hyperparams):
                for name, value in custom_hyperparams.items():
                    hyperparams[name] = value

            hyperparam_types = {name:filter_hyperparam(vl['structural_type']) for name,vl in hyperparam_spec.items() if 'structural_type' in vl.keys()}
        
            for name, value in hyperparams.items():
                tp = hyperparam_types[name]
                if tp is int:
                    send_params[name]=value_pb2.Value(int64=value)
                elif tp is float:
                    send_params[name]=value_pb2.Value(double=value)
                elif tp is bool:
                    send_params[name]=value_pb2.Value(bool=value)
                elif tp is str:
                    send_params[name]=value_pb2.Value(string=value)
                else:
                    if isinstance(value, int):
                        send_params[name]=value_pb2.Value(int64=value)
                    elif isinstance(value, float):
                        send_params[name]=value_pb2.Value(double=value)
                    elif isinstance(value, bool):
                        send_params[name]=value_pb2.Value(bool=value)
                    elif isinstance(value, str):
                        send_params[name]=value_pb2.Value(string=value)
           
        return core_pb2.PrimitiveStepDescription(hyperparams=send_params)


class PrimitiveDescription(object):
    def __init__(self, primitive, primitive_class):
        self.id = primitive_class.id
        self.primitive = primitive
        self.primitive_class = primitive_class

    def score_primitive(self, X, y, metric_type, custom_hyperparams):
        """
        Learns optimal hyperparameters for the primitive
        Evaluates model on inputs X and outputs y
        Returns metric.
        """
        if 'hyperparams' in self.primitive.metadata.query()['primitive_code'] and y is not None:
            hyperparam_spec = self.primitive.metadata.query()['primitive_code']['hyperparams']
            optimal_params = self.find_optimal_hyperparams(train=X, output=y, hyperparam_spec=hyperparam_spec,
          metric=metric_type, custom_hyperparams=custom_hyperparams) 
        else:
            optimal_params = self.primitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams'].defaults()
           
        if y is None or self.primitive.metadata.query()['python_path'] == 'd3m.primitives.sri.psl.VertexNomination':
            return (0.0, optimal_params)
              
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=3, shuffle=True, random_state=9001)
      
        splits = 3 
        metric_sum = 0

        prim_instance = self.primitive(hyperparams=optimal_params)
        score = 0.0
     
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]

            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            prim_instance.set_training_data(inputs=X_train, outputs=y_train)

            prim_instance.fit()
            predictions = prim_instance.produce(inputs=X_test).value                        
            metric = self.evaluate_metric(predictions, y_test, metric_type)     
            metric_sum += metric

        score = metric_sum/splits

        return (score, optimal_params)
 
    def evaluate_metric(self, predictions, Ytest, metric):
        """
        Function to compute prediction accuracy for classifiers.
        """
        count = len(Ytest)
     
        if metric is problem_pb2.ACCURACY:
            return metrics.accuracy_score(Ytest, predictions)
        elif metric is problem_pb2.PRECISION:
            return metrics.precision_score(Ytest, predictions)
        elif metric is problem_pb2.RECALL:
            return metrics.recall_score(Ytest, predictions)
        elif metric is problem_pb2.F1:
            return metrics.f1_score(Ytest, predictions)
        elif metric is problem_pb2.F1_MICRO:
            return metrics.f1_score(Ytest, predictions, average='micro')
        elif metric is problem_pb2.F1_MACRO:
            return metrics.f1_score(Ytest, predictions, average='macro')
        elif metric is problem_pb2.ROC_AUC:
            return metrics.roc_auc_score(Ytest, predictions)
        elif metric is problem_pb2.ROC_AUC_MICRO:
            return metrics.roc_auc_score(Ytest, predictions, average='micro')
        elif metric is problem_pb2.ROC_AUC_MACRO:
            return metrics.roc_auc_score(Ytest, predictions, average='macro')
        elif metric is problem_pb2.MEAN_SQUARED_ERROR:
            return metrics.mean_squared_error(Ytest, predictions)
        elif metric is problem_pb2.ROOT_MEAN_SQUARED_ERROR:
            return math.sqrt(metrics.mean_squared_error(Ytest, predictions))
        elif metric is problem_pb2.ROOT_MEAN_SQUARED_ERROR_AVG:
            return math.sqrt(metrics.mean_squared_error(Ytest, predictions))
        elif metric is problem_pb2.MEAN_ABSOLUTE_ERROR:
            return metrics.mean_absolute_error(Ytest, predictions)
        elif metric is problem_pb2.R_SQUARED:
            return metrics.r2_score(Ytest, predictions)
        elif metric is problem_pb2.NORMALIZED_MUTUAL_INFORMATION:
            return metrics.normalized_mutual_info_score(Ytest, predictions)
        elif metric is problem_pb2.JACCARD_SIMILARITY_SCORE:
            return metrics.jaccard_similarity_score(Ytest, predictions)
        elif metric is problem_pb2.PRECISION_AT_TOP_K:
            return 0.0
        elif metric is problem_pb2.OBJECT_DETECTION_AVERAGE_PRECISION:
            return 0.0
        else:
            return metrics.accuracy_score(Ytest, predictions)

    def optimize_primitive(self, train, output, inputs, default_params, optimal_params, hyperparam_types, metric_type):
        """
        Function to evaluate each input point in the hyper parameter space.
        This is called for every input sample being evaluated by the bayesian optimization package.
        Return value from this function is used to decide on function optimality.
        """
        for index,name in optimal_params.items():
            value = inputs[index]
            if hyperparam_types[name] is int:
                value = (int)(inputs[index]+0.5)
            default_params[name] = value

        prim_instance = self.primitive(hyperparams=default_params)

        import random
        random.seed(9001)

        # Run training on 90% and testing on 10% random split of the dataset.
        seq = [i for i in range(len(train))]
        random.shuffle(seq)

        testsize = (int)(0.1 * len(train) + 0.5)

        trainindices = [seq[x] for x in range(len(train)-testsize)]
        testindices = [seq[x] for x in range(len(train)-testsize, len(train))]
        Xtrain = train.iloc[trainindices]
        Ytrain = output.iloc[trainindices]
        Xtest = train.iloc[testindices]
        Ytest = output.iloc[testindices]

        prim_instance.set_training_data(inputs=Xtrain.values, outputs=Ytrain.values)
        prim_instance.fit()
        predictions = prim_instance.produce(inputs=Xtest).value

        metric = self.evaluate_metric(predictions, Ytest, metric_type)
        
        min_metrics = set()
        min_metrics.add(problem_pb2.MEAN_SQUARED_ERROR)
        min_metrics.add(problem_pb2.ROOT_MEAN_SQUARED_ERROR)
        min_metrics.add(problem_pb2.ROOT_MEAN_SQUARED_ERROR_AVG)
        min_metrics.add(problem_pb2.MEAN_ABSOLUTE_ERROR)
        if metric_type in min_metrics:
            metric = metric * (-1)
        print('Metric: %f' %(metric))
        return metric

    def optimize_hyperparams(self, train, output, lower_bounds, upper_bounds, default_params, hyperparam_types,
     hyperparam_semantic_types, metric_type, custom_hyperparams):
        """
        Optimize primitive's hyper parameters using Bayesian Optimization package 'bo'.
        Optimization is done for the parameters with specified range(lower - upper).
        """
        domain_bounds = []
        optimal_params = {}
        index = 0

        # Create parameter ranges in domain_bounds. 
        # Map parameter names to indices in optimal_params
        for name,value in lower_bounds.items():
            if bool(custom_hyperparams) and name in custom_hyperparams.keys():
                continue
            sem = hyperparam_semantic_types[name]
            if "https://metadata.datadrivendiscovery.org/types/TuningParameter" not in sem:
                continue
            lower = lower_bounds[name]
            upper = upper_bounds[name]
            if lower is None or upper is None:
                continue
            domain_bounds.append([lower,upper])
            optimal_params[index] = name
            index =index+1

        if index == 0:
            return default_params

        func = lambda inputs : self.optimize_primitive(train, output, inputs, default_params, optimal_params, hyperparam_types, metric_type)
       
        try:
            (curr_opt_val, curr_opt_pt) = bo.gp_call.fmax(func, domain_bounds, 10)
        except:
            print("optimize_hyperparams: ", sys.exc_info()[0])
            print(self.primitive)
            optimal_params = None

        optimal_params = None
        # Map optimal parameter values found
        if bool(optimal_params):
            for index,name in optimal_params.items():
                value = curr_opt_pt[index]
                if hyperparam_types[name] is int:
                    value = (int)(curr_opt_pt[index]+0.5)
                default_params[name] = value

        if bool(custom_hyperparams):
            for name,value in custom_hyperparams.items():
                default_params[name] = value

        return default_params

    def find_optimal_hyperparams(self, train, output, hyperparam_spec, metric, custom_hyperparams):
        filter_hyperparam = lambda vl: None if vl == 'None' else vl
        default_hyperparams = {name:filter_hyperparam(vl['default']) for name,vl in hyperparam_spec.items()}
        hyperparam_lower_ranges = {name:filter_hyperparam(vl['lower']) for name,vl in hyperparam_spec.items() if 'lower' in vl.keys()}
        hyperparam_upper_ranges = {name:filter_hyperparam(vl['upper']) for name,vl in hyperparam_spec.items() if 'upper' in vl.keys()}
        hyperparam_types = {name:filter_hyperparam(vl['structural_type']) for name,vl in hyperparam_spec.items() if 'structural_type' in vl.keys()}
        hyperparam_semantic_types = {name:filter_hyperparam(vl['semantic_types']) for name,vl in hyperparam_spec.items() if 'semantic_types' in vl.keys()}
        #print("Defaults: ", default_hyperparams)
        if len(hyperparam_lower_ranges) > 0:
            default_hyperparams = self.optimize_hyperparams(train, output, hyperparam_lower_ranges, hyperparam_upper_ranges,
 default_hyperparams, hyperparam_types, hyperparam_semantic_types, metric, custom_hyperparams)
            #print("Optimals: ", default_hyperparams)

        return default_hyperparams
