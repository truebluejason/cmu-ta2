__author__ = "Saswati Ray"
__email__ = "sray@cs.cmu.edu"

import os, copy, uuid, sys
import solutiondescription, solution_templates
import logging
import util
import numpy as np

def is_PCA_solution(basic_solution):
    length = len(basic_solution.primitives)
    python_path = basic_solution.primitives[length-2].metadata.query()['python_path']
    if 'PCA' in python_path:
        return True
    return False

def isSKFeature(basic_solution):
    length = len(basic_solution.primitives)
    python_path = basic_solution.primitives[length-1].metadata.query()['python_path']
    if 'skfeature.TAMU' in python_path:
        return True
    return False

def is_primitive_reasonable(python_path, rows, total_cols, types_present, task_name):
    if (total_cols > 500 or rows > 100000 or 'TIMESERIES' in types_present) and 'xgboost' in python_path:
        return False

    if ('IMAGE' in types_present or 'AUDIO' in types_present or 'VIDEO' in types_present) and \
       ('ada_boost' in python_path or \
        'passive_aggressive' in python_path or \
        'xgboost' in python_path or \
        'gradient_boosting' in python_path or \
        'bagging' in python_path):
        return False

    # Forecasting
    if task_name == "FORE_REGRESSION" and 'mlp' in python_path:
        return False

    if rows >= 100000 and ('gradient_boosting' in python_path or 'mlp' in python_path):
        return False

    # SVM gets extremely expensive for >10k samples!!!
    if rows > 10000 and 'classification.svc.SKlearn' in python_path:
        return False

    return True

def complex_types_present(types_present):
    if types_present is None:
        return False
    if 'AUDIO' in types_present or \
       'VIDEO' in types_present or \
       'TIMESERIES' in types_present or \
       'IMAGE' in types_present:
        return True
    return False

class auto_solutions(object):
    """
    Main class representing the AutoML(pipeline constructor) for TA2.
    Creates a suite of suitable pipelines for a given dataset/problem type.
    Pipelines are only created using templates and not run here.
    In case of tasks with multiple pipelines (classification/regression/SSL), primary data processing/featurization steps are
    run only once and cached for all the pipelines.
    """
    def __init__(self, task_name, problem = None):
        """
        Constructor
        """
        self.types_present = [] # Data types present in the dataset
        self.task_name = task_name # ML task
        self.solutions = [] # Placeholder to contain list of all pipelines for the specific dataset/problem
        self.basic_sol = None # Primary solution containing all data processing/featurization steps
        self.addn_sol = None # Secondary solution (in case of TEXT/IMAGE etc)
        self.poly_pca = None # PCA-based basic solution
        self.problem = problem
        self.rows = 0
        self.total_cols = 0

    def add_classification_pipelines(self):
        # Add a classification pipeline too
        primitives = ['d3m.primitives.classification.random_forest.SKlearn',
                      'd3m.primitives.classification.extra_trees.SKlearn',
                      'd3m.primitives.classification.gradient_boosting.SKlearn',
                      'd3m.primitives.classification.bernoulli_naive_bayes.SKlearn']
        for p in primitives:
            pipe = self.get_solution_by_task_name('CLASSIFICATION', p)
            pipe.set_classifier_hyperparam()
            self.solutions.append(pipe)

    def add_regression_pipelines(self):
        # Add a regression pipeline too
        primitives = ['d3m.primitives.regression.random_forest.SKlearn',
                      'd3m.primitives.regression.extra_trees.SKlearn',
                      'd3m.primitives.regression.gradient_boosting.SKlearn']

        for p in primitives:
            pipe = self.get_solution_by_task_name('REGRESSION', p)
            self.solutions.append(pipe)

    def get_link_prediction_timeseries_pipelines(self):
        names = ['LINKPREDICTION','LINKPREDICTION2']
        for name in names:
            pipe = self.get_solution_by_task_name(name)
            self.solutions.append(pipe)
        self.add_regression_pipelines()

    def get_vertex_classification_pipelines(self):
        names = ['VERTEXCLASSIFICATION','VERTEXCLASSIFICATION2']
        for name in names:
            pipe = self.get_solution_by_task_name(name)
            self.solutions.append(pipe)
        self.add_classification_pipelines()

    def get_graph_matching_pipelines(self):
        names = ['GRAPHMATCHING','GRAPHMATCHING2']
        for name in names:
            pipe = self.get_solution_by_task_name(name)
            self.solutions.append(pipe)
        self.add_classification_pipelines()

    def get_link_prediction_pipelines(self):
        names = ['LINKPREDICTION','LINKPREDICTION2']
        for name in names:
            pipe = self.get_solution_by_task_name(name)
            self.solutions.append(pipe)
        self.add_classification_pipelines()

    def get_collaborative_filtering_pipelines(self):
        pipe = self.get_solution_by_task_name(self.task_name)
        self.solutions.append(pipe)

        # Add a regression pipeline too
        pipe = self.get_solution_by_task_name('REGRESSION', 'd3m.primitives.regression.extra_trees.SKlearn')
        self.solutions.append(pipe)

    def get_SSL_pipelines(self, rows):
        basics = [self.basic_sol, self.addn_sol]

        # Iterate through variants of possible blackbox hyperparamets.
        total_cols = self.total_cols
        for variant in solution_templates.sslVariants:
            valid = is_primitive_reasonable(variant, rows, total_cols, self.types_present, self.task_name)
            if valid is False:
                continue

            for sol in basics:
                if sol is None:
                    continue

                pipe = copy.deepcopy(sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step('d3m.primitives.semisupervised_classification.iterative_labeling.AutonBox', outputstep = pipe.index_denormalize + 3)
                pipe.add_ssl_variant(variant)
                self.solutions.append(pipe)
        # Add extra pipeline too
        pipe = self.get_solution_by_task_name('SEMISUPERVISED_HDB', 'd3m.primitives.semisupervised_classification.iterative_labeling.AutonBox')
        pipe.add_ssl_variant('d3m.primitives.classification.random_forest.SKlearn')
        self.solutions.append(pipe)

    def get_community_detection_pipelines(self):
        names = ['COMMUNITYDETECTION','COMMUNITYDETECTION2']
        for name in names:
            pipe = self.get_solution_by_task_name(name)
            self.solutions.append(pipe)

    def get_clustering_pipelines(self):
        pipe = self.get_solution_by_task_name(self.task_name)
        self.solutions.append(pipe)
        self.add_classification_pipelines()

    def get_forecasting_pipelines(self, dataset):
        names = ['FORECASTING','FORECASTING2','FORECASTING3']#,'FORECASTING4']
        for name in names:
            pipe = self.get_solution_by_task_name(name)
            self.solutions.append(pipe)
        self.task_name = 'FORE_REGRESSION'
        self.get_solutions(dataset)
 
    def get_object_detection_pipelines(self):
        names = ['OBJECTDETECTION','OBJECTDETECTION2']
        for name in names:
            pipe = self.get_solution_by_task_name(name)
            self.solutions.append(pipe)

    def create_imvadio_solution(self):
        if self.types_present is None:
            return

        if 'AUDIO' in self.types_present or 'VIDEO' in self.types_present:
            if self.task_name == 'CLASSIFICATION':
                prim = 'd3m.primitives.classification.extra_trees.SKlearn'
            else:
                prim = 'd3m.primitives.regression.extra_trees.SKlearn'
            pipe = self.get_solution_by_task_name('IMVADIO', ML_prim=prim)
            self.solutions.append(pipe)
 
    def create_basic_solutions(self, dataset):
        """
        In case of tasks with multiple pipelines (classification/regression/SSL), primary data processing/featurization steps are
        run only once and cached for all the pipelines.
        """
        if self.task_name != 'CLASSIFICATION' and self.task_name != 'REGRESSION' and self.task_name != 'SEMISUPERVISED' and self.task_name != 'FORE_REGRESSION':
            return

        basic_sol = None
        try:
            # Set data types, and meta data information to begin with
            basic_sol = solutiondescription.SolutionDescription(self.problem)
            basic_sol.initialize_solution(self.task_name)
            (self.types_present, self.total_cols, self.rows, categorical_atts, ordinal_atts, ok_to_denormalize, privileged, add_floats, ok_to_augment, profiler_needed) = solutiondescription.column_types_present(dataset)
            logging.critical(self.types_present)
            basic_sol.set_add_floats(add_floats)
            basic_sol.set_categorical_atts(categorical_atts)
            basic_sol.set_ordinal_atts(ordinal_atts)
            basic_sol.set_denormalize(ok_to_denormalize)
            basic_sol.set_privileged(privileged)
            basic_sol.profiler_needed = profiler_needed
            basic_sol.initialize_solution(self.task_name)
        except:
            logging.error(sys.exc_info()[0])
            basic_sol = None
            self.types_present = None
            self.basic_sol = None
            return

        # For file in each data point, we treat as time series for featurization
        if len(self.types_present) == 1 and self.types_present[0] == 'FILES':
            self.types_present[0] = 'TIMESERIES'

        self.basic_sol = basic_sol
        # Initialize basic solutions based on data types
        if 'TIMESERIES' in self.types_present:
            self.basic_sol.initialize_solution('TIMESERIES')
        elif 'IMAGE' in self.types_present:
            if self.rows > 1200:
                self.basic_sol.add_splitter()
                self.rows = 1200
            self.basic_sol.initialize_solution('IMAGE')
            self.addn_sol = copy.deepcopy(basic_sol)
            self.addn_sol.initialize_solution('IMAGE2')
        elif 'TEXT' in self.types_present:
            if self.task_name == "FORE_REGRESSION":
                self.basic_sol.initialize_solution('DISTILTEXT')
                self.basic_sol.set_distil_text_hyperparam()
            else:
                self.basic_sol.initialize_solution('TEXT')
                self.addn_sol = copy.deepcopy(basic_sol)
                self.addn_sol.initialize_solution('DISTILTEXT')
                if self.task_name == 'REGRESSION':
                    self.addn_sol.set_distil_text_hyperparam()
        elif 'AUDIO' in self.types_present:
            if self.rows > 1200:
                self.basic_sol.add_splitter()
                self.rows = 1200
            self.basic_sol.initialize_solution('AUDIO')
            print(self.basic_sol.primitives_arguments)
        elif 'VIDEO' in self.types_present:
            self.basic_sol.initialize_solution('VIDEO')

    def get_solutions(self, dataset):
        """
        Get a list of available solutions(pipelines) for the specified task
        Used by both TA2 in "search" phase and TA2-TA3
        """
        if self.task_name == 'VERTEXNOMINATION' or self.task_name == 'VERTEXCLASSIFICATION':
            self.task_name = 'VERTEXCLASSIFICATION'
            self.get_vertex_classification_pipelines()
        elif self.task_name == 'COMMUNITYDETECTION':
            self.get_community_detection_pipelines()
        elif self.task_name == 'LINKPREDICTION':
            self.get_link_prediction_pipelines()
        elif self.task_name == 'GRAPHMATCHING':
            self.get_graph_matching_pipelines()
        elif self.task_name == 'CLUSTERING':
            self.get_clustering_pipelines()
        elif self.task_name == 'OBJECTDETECTION':
            self.get_object_detection_pipelines()
        elif self.task_name == 'COLLABORATIVEFILTERING':
            self.get_collaborative_filtering_pipelines()
        elif self.task_name == 'FORECASTING':
            self.get_forecasting_pipelines(dataset)
        elif self.task_name == 'LINKPREDICTIONTIMESERIES':
            self.get_link_prediction_timeseries_pipelines()
        else: # CLASSIFICATION / REGRESSION / SEMISUPERVISED / FORE_REGRESSION 
            # Initialize dataset properties, data types etc.  
            self.create_basic_solutions(dataset)
            
            # Run common steps of solutions before forking out processes for classifiers/regressors
            output_step_index = self.basic_sol.index_denormalize + 3
            if 'AUDIO' in self.types_present:
                output_step_index = self.basic_sol.index_denormalize + 2
            if self.basic_sol.splitter_present() == False or \
                'IMAGE' in self.types_present or \
                'AUDIO' in self.types_present:
                try:
                    self.basic_sol.run_basic_solution(inputs=[dataset], output_step = output_step_index)
                    self.total_cols = self.basic_sol.get_total_cols()
                    logging.critical("Total cols = %s", self.total_cols)
                except:
                    logging.error(sys.exc_info()[0])
                    self.basic_sol = None
                  
            # Run additional(secondary solution, if any
            if self.addn_sol is not None and (self.addn_sol.splitter_present() == False or 'IMAGE' in self.types_present):
                try:
                    self.addn_sol.run_basic_solution(inputs=[dataset], output_step = output_step_index)
                except:
                    logging.error(sys.exc_info()[0])
                    self.addn_sol = None

            # Add primitives to basic solutions
            if self.task_name == 'CLASSIFICATION' or self.task_name == 'REGRESSION' or self.task_name == 'FORE_REGRESSION':
                for sol in [self.basic_sol, self.addn_sol]:
                    self.get_primitive_solutions(sol, self.rows)
                # Try RPI solutions
                if self.basic_sol is not None and self.basic_sol.splitter == False:
                    if self.is_multi_column_output(self.basic_sol, output_step_index) is False: # multi-column output?
                        self.get_rpi_solutions(self.basic_sol.add_floats, self.basic_sol.privileged, self.rows, dataset)
                conditioner_prim = 'd3m.primitives.classification.bagging.SKlearn'
                if self.task_name != 'CLASSIFICATION':
                    conditioner_prim = 'd3m.primitives.regression.bagging.SKlearn'
                pipe = self.get_solution_by_task_name('CONDITIONER', ML_prim=conditioner_prim)
                if self.rows < 1000000: 
                    self.solutions.append(pipe)
                self.create_imvadio_solution()
            else:
                # Add primitives for SSL solutions
                self.get_SSL_pipelines(self.rows)
                self.get_ssl_rpi_solutions(self.basic_sol.add_floats, self.basic_sol.privileged, self.rows, dataset)
            
            # Extra pipeline for TS classification
            if self.task_name == 'CLASSIFICATION' and 'TIMESERIES' in self.types_present:
                pipe = solutiondescription.SolutionDescription(self.problem)
                pipe.initialize_solution('TIMESERIES2')
                pipe.id = str(uuid.uuid4())
                pipe.run_basic_solution(inputs=[dataset], input_step=1, output_step = 3, dataframe_step = 2)
                pipe.add_step('d3m.primitives.time_series_classification.k_neighbors.Kanine', inputstep=1, outputstep=3, dataframestep=2)
                self.solutions.append(pipe)
 
        return self.solutions

    def is_multi_column_output(self, basic_sol, outputstep):
        if basic_sol.primitives_outputs != None and len(basic_sol.primitives_outputs[outputstep].columns) > 1:
            return True
        return False

    def get_primitive_solutions(self, basic_sol, rows):
        """
        Create multiple pipelines for a task by iterating through different primitives
        Iterate through different ML models (classifiers/regressors).
        Each one is evaluated in a seperate process. 
        """
        if basic_sol is None:
            return 

        if self.task_name == "REGRESSION" or self.task_name == "FORE_REGRESSION":
            listOfSolutions = solution_templates.regressors
        else:
            listOfSolutions = solution_templates.classifiers

        total_cols = self.total_cols
        dataframestep = 1
        if basic_sol.splitter_present() == True:
            dataframestep = 2

        for python_path in listOfSolutions:
            # Prune out expensive pipelines
            valid = is_primitive_reasonable(python_path, rows, total_cols, self.types_present, self.task_name)
            if valid is False: 
                continue

            if is_PCA_solution(basic_sol) == True and not('random' in python_path):
                continue

            outputstep = basic_sol.index_denormalize + 3
            if 'AUDIO' in self.types_present:
                outputstep = basic_sol.index_denormalize + 2
            if self.is_multi_column_output(basic_sol, outputstep) is True: # multi-column output?
                # These do not work for multi-column output
                if 'linear_sv' in python_path or 'ada_boost' in python_path or 'lasso_cv' in python_path or 'gradient_boosting' in python_path:
                    continue

            pipe = copy.deepcopy(basic_sol)
            pipe.id = str(uuid.uuid4())
            pipe.add_step(python_path, outputstep = outputstep, dataframestep=dataframestep)
            self.solutions.append(pipe)

    def get_solution_by_task_name(self, name, ML_prim=None, outputstep=3, dataframestep=1):
        """
        Get a single complete pipeline by task name.
        Used for less frequent tasks (not Classification or regression). 
        name: Name of pipeline template
        ML_prim: ML primitive to be appended (classifier/regressor mostly)
        outputstep: Step in pipeline producing targets
        dataframestep: Dataframe step in pipeline. Used for constructing predictions.
        """
        pipe = solutiondescription.SolutionDescription(self.problem)
        pipe.initialize_solution(name)
        pipe.id = str(uuid.uuid4())
        if ML_prim is not None:
            step1 = pipe.index_denormalize + outputstep
            step2 = pipe.index_denormalize + dataframestep
            pipe.add_step(ML_prim, outputstep=step1, dataframestep=step2)
        else:
            pipe.add_outputs()
        return pipe    

    def get_ssl_rpi_solutions(self, add_floats, privileged, rows, dataset):
        """
        Get RPI-based pipelines for SSL tasks.
        """
        if self.types_present is None:
            return

        if 'AUDIO' in self.types_present or \
           'VIDEO' in self.types_present or \
           'TIMESERIES' in self.types_present or \
           'IMAGE' in self.types_present or \
           'TEXT' in self.types_present:
            return

        basic_sol = solutiondescription.SolutionDescription(self.problem)
        basic_sol.set_privileged(privileged)
        basic_sol.set_add_floats(add_floats)
        basic_sol.initialize_RPI_solution('NOTUNE_PIPELINE_RPI')
        outputstep = basic_sol.index_denormalize + 4
        if add_floats is not None and len(add_floats) > 0:
            outputstep = basic_sol.index_denormalize + 5

        sslVariants = solution_templates.sslVariants
        total_cols = self.total_cols
        if rows <= 100000:
            try:
                basic_sol.run_basic_solution(inputs=[dataset], output_step=outputstep, dataframe_step=basic_sol.index_denormalize + 1)
                total_cols = basic_sol.get_total_cols()
                logging.info("Total cols = %s", total_cols)
            except:
                logging.error(sys.exc_info()[0])
                basic_sol = None
        else:
            sslVariants = ['d3m.primitives.classification.random_forest.SKlearn',
                           'd3m.primitives.classification.bagging.SKlearn']

        if basic_sol is None or total_cols > 200:
            return

        for python_path in sslVariants:
            valid = is_primitive_reasonable(python_path, rows, total_cols, self.types_present, self.task_name)
            if valid is False:
                continue

            pipe = copy.deepcopy(basic_sol)
            pipe.id = str(uuid.uuid4())
            pipe.add_step('d3m.primitives.semisupervised_classification.iterative_labeling.AutonBox', outputstep = pipe.index_denormalize + 4)
            pipe.add_ssl_variant(python_path)
            self.solutions.append(pipe)

    def get_rpi_solutions(self, add_floats, privileged, rows, dataset):
        """
        Get RPI-based pipelines for classification/regression tasks.
        """
        if self.types_present is None:
            return

        if self.task_name != "REGRESSION" and self.task_name != "CLASSIFICATION":
            return

        if rows >= 100000 and self.total_cols > 12:
            return

        if 'AUDIO' in self.types_present or \
           'VIDEO' in self.types_present or \
           'TIMESERIES' in self.types_present or \
           'IMAGE' in self.types_present or \
           'TEXT' in self.types_present:
            return

        basic_sol = solutiondescription.SolutionDescription(self.problem)
        basic_sol.set_privileged(privileged)
        basic_sol.set_add_floats(add_floats)
        outputstep = basic_sol.index_denormalize + 4
        if add_floats is not None and len(add_floats) > 0:
            outputstep = basic_sol.index_denormalize + 5

        total_cols = self.total_cols
        if self.task_name == "REGRESSION":
            listOfSolutions = solution_templates.regressors_rpi
        else:
            listOfSolutions = solution_templates.classifiers_rpi

        if rows >= 25000:
            # No tuning
            basic_sol.initialize_RPI_solution('NOTUNE_PIPELINE_RPI')
            if self.task_name == "REGRESSION":
                listOfSolutions = ['d3m.primitives.regression.random_forest.SKlearn']
            else:
                listOfSolutions = ['d3m.primitives.classification.random_forest.SKlearn']
        else:
            # Grid-search over RPI's binsize and model's no. of estimators. This can be expensive
            basic_sol.initialize_RPI_solution('PIPELINE_RPI')
            try:
                basic_sol.run_basic_solution(inputs=[dataset], output_step=outputstep, dataframe_step=basic_sol.index_denormalize + 1)
                total_cols = basic_sol.get_total_cols()
                logging.info("Total cols = %s", total_cols)
            except:
                logging.error(sys.exc_info()[0])
                basic_sol = None

        if basic_sol is None or total_cols > 200:
            return

        RPI_steps = ['d3m.primitives.feature_selection.joint_mutual_information.AutoRPI','d3m.primitives.feature_selection.simultaneous_markov_blanket.AutoRPI']

        for python_path in listOfSolutions:
            # Avoid expensive solutions!!!
            if 'gradient_boosting' in python_path and ((rows > 1000 and total_cols > 50) or (rows > 5000) or total_cols > 100):
                continue

            if rows >= 25000:
                pipe = copy.deepcopy(basic_sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step(python_path, outputstep)
                self.solutions.append(pipe)
            else:
                for step in RPI_steps:
                    pipe = copy.deepcopy(basic_sol)
                    pipe.id = str(uuid.uuid4())
                    pipe.add_RPI_step(step, python_path, outputstep)
                    self.solutions.append(pipe)

