import os, copy, uuid, sys
import solutiondescription
import logging
import util, search
import numpy as np
from timeit import default_timer as timer
import signal

from multiprocessing import Pool, cpu_count
from multiprocessing.context import TimeoutError

logging.basicConfig(level=logging.INFO)

task_paths = {
'TEXTCLASSIFICATION': ['d3m.primitives.data_transformation.denormalize.Common',
                       'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                       'd3m.primitives.schema_discovery.profiler.Common',
                       'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                       'd3m.primitives.data_transformation.column_parser.Common',
                       'd3m.primitives.data_preprocessing.text_reader.Common',
                       'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                       'd3m.primitives.data_cleaning.imputer.SKlearn',
                       'd3m.primitives.data_transformation.encoder.DistilTextEncoder'],

'TEXT': ['d3m.primitives.data_transformation.denormalize.Common',
         'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
         'd3m.primitives.schema_discovery.profiler.Common',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
         'd3m.primitives.data_transformation.column_parser.Common',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
         'd3m.primitives.data_cleaning.imputer.SKlearn',
         'd3m.primitives.feature_construction.corex_text.DSBOX'],

'LARGETEXT': ['d3m.primitives.data_preprocessing.splitter.DSBOX',
         'd3m.primitives.data_transformation.denormalize.Common',
         'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
         'd3m.primitives.schema_discovery.profiler.Common',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
         'd3m.primitives.data_transformation.column_parser.Common',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
         'd3m.primitives.data_cleaning.imputer.SKlearn',
         'd3m.primitives.feature_construction.corex_text.DSBOX'],

'TIMESERIES': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.schema_discovery.profiler.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
               'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX',
               'd3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX'],

'TIMESERIES2': ['d3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter',
                'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'],

'TIMESERIES3': ['d3m.primitives.data_transformation.denormalize.Common',
                'd3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN'],

'IMAGE2': ['d3m.primitives.data_transformation.denormalize.Common',
           'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
           'd3m.primitives.schema_discovery.profiler.Common',
           'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
           'd3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX',
           'd3m.primitives.feature_extraction.resnet50_image_feature.DSBOX'],

'IMAGE': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.schema_discovery.profiler.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
          'd3m.primitives.data_preprocessing.image_reader.Common',
          'd3m.primitives.data_transformation.column_parser.Common',
          'd3m.primitives.feature_extraction.image_transfer.DistilImageTransfer'],

'VIDEO': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.schema_discovery.profiler.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
          'd3m.primitives.data_transformation.column_parser.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
          'd3m.primitives.feature_extraction.resnext101_kinetics_video_features.VideoFeaturizer'],

'CLASSIFICATION': ['d3m.primitives.data_transformation.denormalize.Common',
                   'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                   'd3m.primitives.schema_discovery.profiler.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                   'd3m.primitives.data_transformation.column_parser.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                   'd3m.primitives.data_cleaning.imputer.SKlearn'],

'SEMISUPERVISED': ['d3m.primitives.data_transformation.denormalize.Common',
                   'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                   'd3m.primitives.schema_discovery.profiler.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                   'd3m.primitives.data_transformation.column_parser.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                   'd3m.primitives.data_cleaning.imputer.SKlearn'],

'FORECASTING': ['d3m.primitives.data_transformation.denormalize.Common',
                   'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                   'd3m.primitives.schema_discovery.profiler.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                   'd3m.primitives.data_transformation.column_parser.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                   'd3m.primitives.time_series_forecasting.arima.DSBOX'],

'FORECASTING2': ['d3m.primitives.data_transformation.denormalize.Common',
                 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                 'd3m.primitives.schema_discovery.profiler.Common',
                 'd3m.primitives.data_transformation.column_parser.Common',
                 'd3m.primitives.time_series_forecasting.vector_autoregression.VAR'],

'REGRESSION': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.schema_discovery.profiler.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
               'd3m.primitives.data_transformation.column_parser.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
               'd3m.primitives.data_cleaning.imputer.SKlearn'],

'GENERAL_RELATIONAL': ['d3m.primitives.data_preprocessing.splitter.DSBOX',
                       'd3m.primitives.data_transformation.denormalize.Common',
                       'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                       'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                       'd3m.primitives.classification.general_relational_dataset.GeneralRelationalDataset',
                       'd3m.primitives.data_transformation.cast_to_type.Common'],

'PIPELINE_RPI': ['d3m.primitives.data_transformation.denormalize.Common',
                 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                 'd3m.primitives.schema_discovery.profiler.Common',
                 'd3m.primitives.data_transformation.column_parser.Common',
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'],

#'CLUSTERING': ['d3m.primitives.data_transformation.dataset_to_dataframe.Common',
#               'd3m.primitives.data_transformation.column_parser.Common',
#               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
#               'd3m.primitives.clustering.k_means.Fastlvm',
#               'd3m.primitives.data_transformation.construct_predictions.Common'],

'CLUSTERING': ['d3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_transformation.column_parser.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
               'd3m.primitives.data_transformation.dataframe_to_ndarray.Common',
               'd3m.primitives.clustering.ekss.Umich',
               'd3m.primitives.data_transformation.ndarray_to_dataframe.Common',
               'd3m.primitives.data_transformation.construct_predictions.Common'],

'GRAPHMATCHING': ['d3m.primitives.data_transformation.load_graphs.DistilGraphLoader',
                  'd3m.primitives.graph_matching.seeded_graph_matching.DistilSeededGraphMatcher'],

'GRAPHMATCHING2': ['d3m.primitives.graph_matching.seeded_graph_matching.JHU'],

'COLLABORATIVEFILTERING': ['d3m.primitives.data_transformation.denormalize.Common',
                           'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                           'd3m.primitives.schema_discovery.profiler.Common',
                           'd3m.primitives.data_transformation.column_parser.Common',
                           'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                           'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                           'd3m.primitives.collaborative_filtering.collaborative_filtering_link_prediction.DistilCollaborativeFiltering',
                           'd3m.primitives.data_transformation.construct_predictions.Common'],

'VERTEXCLASSIFICATION3': ['d3m.primitives.data_transformation.load_graphs.JHU',
                          'd3m.primitives.data_preprocessing.largest_connected_component.JHU',
                          'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
                          'd3m.primitives.classification.gaussian_classification.JHU'],

'VERTEXCLASSIFICATION4': ['d3m.primitives.data_transformation.vertex_classification_parser.VertexClassificationParser',
                          'd3m.primitives.classification.vertex_nomination.VertexClassification'],

'VERTEXCLASSIFICATION': ['d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                         'd3m.primitives.vertex_nomination.vertex_nomination.DistilVertexNomination'],

'OBJECTDETECTION': ['d3m.primitives.data_transformation.denormalize.Common',
                    'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                    'd3m.primitives.feature_extraction.yolo.DSBOX'],

'LINKPREDICTION2': ['d3m.primitives.link_prediction.data_conversion.JHU', 
                    'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
                    'd3m.primitives.link_prediction.rank_classification.JHU'],

'LINKPREDICTION': ['d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                    'd3m.primitives.link_prediction.link_prediction.DistilLinkPrediction'],

'COMMUNITYDETECTION': ['d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                       'd3m.primitives.community_detection.community_detection.DistilCommunityDetection'],

'IMVADIO': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.schema_discovery.profiler.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
          'd3m.primitives.data_transformation.column_parser.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
          'd3m.primitives.data_transformation.one_hot_encoder.SKlearn'],

'AUDIO': ['d3m.primitives.data_preprocessing.audio_loader.DistilAudioDatasetLoader',
          'd3m.primitives.data_transformation.column_parser.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
          'd3m.primitives.feature_extraction.audio_transfer.DistilAudioTransfer']}

classifiers = ['d3m.primitives.classification.bernoulli_naive_bayes.SKlearn',
               'd3m.primitives.classification.linear_discriminant_analysis.SKlearn',
               'd3m.primitives.classification.logistic_regression.SKlearn',
               'd3m.primitives.classification.ada_boost.SKlearn',
               'd3m.primitives.classification.linear_svc.SKlearn',
               'd3m.primitives.classification.extra_trees.SKlearn',
               'd3m.primitives.classification.random_forest.SKlearn',
               'd3m.primitives.classification.bagging.SKlearn',
               'd3m.primitives.classification.svc.SKlearn',
               'd3m.primitives.classification.passive_aggressive.SKlearn',
               'd3m.primitives.classification.xgboost_gbtree.Common',
               'd3m.primitives.classification.gradient_boosting.SKlearn']

regressors = ['d3m.primitives.regression.ridge.SKlearn',
              'd3m.primitives.regression.lasso.SKlearn',
              'd3m.primitives.regression.elastic_net.SKlearn',
              'd3m.primitives.regression.lasso_cv.SKlearn',
              'd3m.primitives.regression.ada_boost.SKlearn',
              'd3m.primitives.regression.linear_svr.SKlearn',
              'd3m.primitives.regression.random_forest.SKlearn',
              'd3m.primitives.regression.extra_trees.SKlearn',
              'd3m.primitives.regression.xgboost_gbtree.Common',
              'd3m.primitives.regression.gradient_boosting.SKlearn']

regressors_rpi = ['d3m.primitives.regression.random_forest.SKlearn',
                  'd3m.primitives.regression.extra_trees.SKlearn',
                  'd3m.primitives.regression.gradient_boosting.SKlearn']

classifiers_rpi = ['d3m.primitives.classification.random_forest.SKlearn',
                   'd3m.primitives.classification.extra_trees.SKlearn',
                   'd3m.primitives.classification.gradient_boosting.SKlearn']

regressors_general_relational = ['d3m.primitives.regression.random_forest.SKlearn',
                                 'd3m.primitives.regression.extra_trees.SKlearn']

classifiers_general_relational = ['d3m.primitives.classification.random_forest.SKlearn',
                                  'd3m.primitives.classification.extra_trees.SKlearn',
                                  'd3m.primitives.classification.linear_discriminant_analysis.SKlearn']

sslVariants = ['d3m.primitives.classification.gradient_boosting.SKlearn',
               'd3m.primitives.classification.extra_trees.SKlearn',
               'd3m.primitives.classification.random_forest.SKlearn',
               'd3m.primitives.classification.bagging.SKlearn']

def get_augmented_solutions(task_name, dataset, primitives, problem_metric, posLabel, problem, keywords, timeout = 60):
    """
    Get all augmented solution by
        1. Search datasets relevant
        2. Evaluate the one that improves the most the performances
        3. Return the TA2 on this dataset
    
    Arguments:
        task_name {[type]} -- [description]
        dataset {[type]} -- [description]
        primitives {[type]} -- [description]
        problem_metric {[type]} -- [description]
        posLabel {[type]} -- [description]
        keywords {[type]} -- [description]
    """
    print('-' * 100)
    start = timer()
    # Search in datamart
    try:
        datasets = util.search_all_related(dataset, keywords)
    except:
        logging.error("DATAMART NOT AVAILABLE")
        return ([], timer() - start)
    
    if len(datasets) == 0:
        return ([], timer() - start)

    try:
        # Evaluate one model on each
        pool = Pool(cpu_count()) 
        parallel_solutions  = [pool.apply_async(get_solutions, (task_name, dataset, primitives, problem_metric, posLabel, problem, aug_dataset.serialize(), True)) for aug_dataset in datasets]

        # Creates solutions
        solutions, time_spent = {}, 0
        for i, process in enumerate(parallel_solutions):
            try:
                start_loop = timer()
                (solution, _) = process.get(timeout=max(0, timeout - time_spent))
                time_spent += timer() - start_loop
                solutions[i] = solution[0]
            except Exception as e:
                logging.error("Augmentation with: {} => TOO SLOW".format(datasets[i].get_json_metadata()['metadata']['name']), e)

        # Evaluates solutions
        parallel_evaluations  = [pool.apply_async(search.evaluate_solution_score, ([dataset], solutions[i], primitives, problem_metric, posLabel, None)) for i in solutions]
        performances, time_spent = {}, 0
        for i, process in zip(solutions.keys(), parallel_evaluations):
            try:
                start_loop = timer()
                performances[i] = process.get(timeout=max(0, timeout - time_spent))[0]
                time_spent += timer() - start_loop
                logging.info("Augmentation with: {} => {}".format(datasets[i].get_json_metadata()['metadata']['name'], performances[i]))
            except Exception as e:
                logging.error("Augmentation scoring with: {} => TOO SLOW".format(datasets[i].get_json_metadata()['metadata']['name']), e)

        # Rank solutions
        sorted_x = search.rank_solutions(performances, problem_metric)
        best = datasets[sorted_x[0][0]]

        # Get all solution
        logging.info("Best augmentation: {}".format(best.get_json_metadata()['metadata']['name']))
        (solutions, _) = get_solutions(task_name, dataset, primitives, problem_metric, posLabel, problem, augmentation_dataset = best.serialize())
    except Exception as e:
        logging.error("Unexpected error during creation pipeline", e)
        solutions = []

    return (solutions, timer() - start)

def get_solutions(task_name, dataset, primitives, problem_metric, posLabel, problem, augmentation_dataset = None, one_model = False):
    """
    Get a list of available solutions(pipelines) for the specified task
    Used by both TA2 in "search" phase and TA2-TA3

    augmentation_dataset -- Serialized dataset returned by Datamart
    one_model -- Return one extra tree model
    """
    solutions = []
    time_used = 0
    start_solution_search = timer()

    if task_name == 'FORECASTING':
        tasks = ['FORECASTING'] #,'FORECASTING2']
        for name in tasks:
            basic_sol = solutiondescription.SolutionDescription(problem)
            basic_sol.initialize_solution(name, augmentation_dataset)
            basic_sol.add_outputs()
            solutions.append(basic_sol)
        task_name = 'REGRESSION'

    if task_name == 'VERTEXNOMINATION':
        task_name = 'VERTEXCLASSIFICATION'

    if augmentation_dataset:
        basic_sol = solutiondescription.SolutionDescription(problem)
        basic_sol.initialize_solution(task_name, augmentation_dataset)
        basic_sol.clear_model()
    
    types_present = []
    total_cols = 0
    privileged = []
    if task_name == 'CLASSIFICATION' or task_name == 'REGRESSION' or task_name == 'SEMISUPERVISED':
        try:
            basic_sol = solutiondescription.SolutionDescription(problem)
            basic_sol.initialize_solution(task_name, augmentation_dataset)
            (types_present, total_cols, rows, categorical_atts, ordinal_atts, ok_to_denormalize, privileged, add_floats, add_texts, ok_to_augment, profiler_needed) = solutiondescription.column_types_present(dataset, augmentation_dataset)
            logging.critical(types_present)
            basic_sol.set_add_floats(add_floats)
            basic_sol.set_add_texts(add_texts)
            basic_sol.set_categorical_atts(categorical_atts)
            basic_sol.set_ordinal_atts(ordinal_atts)
            basic_sol.set_denormalize(ok_to_denormalize)
            basic_sol.set_privileged(privileged)
            basic_sol.profiler_needed = profiler_needed
            if ok_to_augment == False:
                augmentation_dataset = None
            basic_sol.initialize_solution(task_name, augmentation_dataset)
        except:
            logging.error(sys.exc_info()[0])
            basic_sol = None
            types_present = None
            rows = 0

        if types_present is not None:
            if one_model == True and ('AUDIO' in types_present or \
                'VIDEO' in types_present or \
                'TIMESERIES' in types_present or \
                'IMAGE' in types_present or \
                rows > 100000):
                 return ([], 0)

            if len(types_present) == 1 and types_present[0] == 'FILES':
                types_present[0] = 'TIMESERIES' 
            try:
                largetext = False
                addn_sol = None
                if 'TIMESERIES' in types_present:
                    basic_sol.initialize_solution('TIMESERIES', augmentation_dataset)
                elif 'IMAGE' in types_present:
                    basic_sol.initialize_solution('IMAGE2', augmentation_dataset)
                    #if rows < 2000:
                    #    addn_sol = copy.deepcopy(basic_sol)
                    #    addn_sol.initialize_solution('IMAGE2', augmentation_dataset)
                elif 'TEXT' in types_present:
                    if task_name == 'CLASSIFICATION':
                        basic_sol.initialize_solution('TEXTCLASSIFICATION', augmentation_dataset)
                        addn_sol = copy.deepcopy(basic_sol)
                        addn_sol.initialize_solution('TEXT', augmentation_dataset)
                    else:
                        if rows > 50000:
                            basic_sol.initialize_solution('LARGETEXT', augmentation_dataset)
                            largetext = True
                        else:
                            basic_sol.initialize_solution('TEXT', augmentation_dataset)
                elif 'AUDIO' in types_present:
                    basic_sol.initialize_solution('AUDIO', augmentation_dataset)
                elif 'VIDEO' in types_present:
                    basic_sol.initialize_solution('VIDEO', augmentation_dataset)

                start = timer()
                if largetext == True:
                    basic_sol.index_denormalize = 1 # splitter is present

                if basic_sol.splitter_present() == False: # splitter is too expensive to copy! We will run each solution from scratch
                    output_step_index = basic_sol.index_denormalize + 3
                    basic_sol.run_basic_solution(inputs=[dataset], output_step = output_step_index, dataframe_step = basic_sol.index_denormalize + 1)
                if addn_sol is not None:
                    output_step_index = addn_sol.index_denormalize + 3
                    addn_sol.run_basic_solution(inputs=[dataset], output_step = output_step_index, dataframe_step = addn_sol.index_denormalize + 1)

                end = timer()
                logging.info("Time taken to run basic solution: %s secs", end - start)
                total_cols = basic_sol.get_total_cols()
                logging.info("Total cols = %s", total_cols)
                if addn_sol is not None:
                    logging.info("Total cols = %s", addn_sol.get_total_cols())
            except:
                logging.error(sys.exc_info()[0])
                basic_sol = None

        # Iterate through primitives which match task type for populative pool of solutions
        listOfSolutions = []
        if basic_sol is not None:
            if task_name == "REGRESSION":
                if one_model:
                    listOfSolutions = ['d3m.primitives.regression.extra_trees.SKlearn']
                else:
                    listOfSolutions = regressors
            elif task_name == "CLASSIFICATION":
                if one_model:
                    listOfSolutions = ['d3m.primitives.classification.extra_trees.SKlearn']
                else:
                    listOfSolutions = classifiers
        
        # Iterate through different ML models (classifiers/regressors).
        # Each one is evaluated in a seperate process.
        for python_path in listOfSolutions:
            if (total_cols > 500 or rows > 100000 or 'IMAGE' in types_present) and 'xgboost' in python_path:
                continue

            if rows > 100000 and ('linear_sv' in python_path or 'gradient_boosting' in python_path):
                continue

            if types_present is not None and 'TIMESERIES' in types_present and 'xgboost' in python_path:
                continue

            # SVM gets extremely expensive for >10k samples!!!
            if rows > 10000 and 'classification.svc.SKlearn' in python_path:
                continue 

            outputstep = basic_sol.index_denormalize + 3
            if basic_sol.splitter_present() == True:
                outputstep = basic_sol.index_denormalize + 4

            if basic_sol.primitives_outputs != None and len(basic_sol.primitives_outputs[outputstep].columns) > 1: # multi-column output?
                # These do not work for multi-column output
                if 'linear_sv' in python_path or 'ada_boost' in python_path or 'lasso_cv' in python_path or 'gradient_boosting' in python_path:
                    continue

            pipe = copy.deepcopy(basic_sol)
            pipe.id = str(uuid.uuid4())
            pipe.add_step(python_path, outputstep = outputstep, dataframestep = pipe.index_denormalize + 1)
            solutions.append(pipe)

            if addn_sol is not None:
                pipe = copy.deepcopy(addn_sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step(python_path, outputstep = outputstep, dataframestep = pipe.index_denormalize + 1)
                solutions.append(pipe)

        # Try general relational pipelines
        #if not one_model:
        #    general_solutions = get_general_relational_solutions(task_name, types_present, rows, dataset, primitives, problem_metric, posLabel, problem)
        #    solutions = solutions + general_solutions
           
        #    # Try RPI solutions
        #    rpi_solutions = get_rpi_solutions(task_name, types_present, privileged, rows, dataset, primitives, problem_metric, posLabel, problem)
        #    solutions = solutions + rpi_solutions

        if task_name == 'SEMISUPERVISED':
            # Iterate through variants of possible blackbox hyperparamets.
            for variant in sslVariants:
                if rows > 100000 and 'gradient_boosting' in variant:
                    continue
                
                pipe = copy.deepcopy(basic_sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step('d3m.primitives.semisupervised_classification.iterative_labeling.AutonBox', outputstep = pipe.index_denormalize + 3, dataframestep = pipe.index_denormalize + 1)
                pipe.add_ssl_variant(variant)
                solutions.append(pipe)

    elif task_name == 'VERTEXCLASSIFICATION' or \
         task_name == 'COMMUNITYDETECTION' or \
         task_name == 'GRAPHMATCHING' or \
         task_name == 'LINKPREDICTION' or \
         task_name == 'CLUSTERING':
        basic_sol = solutiondescription.SolutionDescription(problem)
        basic_sol.initialize_solution(task_name)
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)
        
        # Add a classification pipeline too
        primitives = ['d3m.primitives.classification.random_forest.SKlearn', 
                      'd3m.primitives.classification.extra_trees.SKlearn', 
                      'd3m.primitives.classification.gradient_boosting.SKlearn', 
                      'd3m.primitives.classification.bernoulli_naive_bayes.SKlearn']
        for p in primitives:
            if task_name == 'COMMUNITYDETECTION' and 'gradient_boosting' in p:
                continue
            pipe = solutiondescription.SolutionDescription(problem)
            pipe.initialize_solution('CLASSIFICATION')
            pipe.id = str(uuid.uuid4())
            outputstep = pipe.index_denormalize + 3
            pipe.add_step(p, outputstep=outputstep)
            solutions.append(pipe)

        if task_name == 'VERTEXCLASSIFICATION':
            indexes = ['3','4']
            for index in indexes:
                pipe = solutiondescription.SolutionDescription(problem)
                second_name = task_name + index
                pipe.initialize_solution(second_name)
                pipe.id = str(uuid.uuid4())
                pipe.add_outputs()
                solutions.append(pipe)
        if task_name == 'LINKPREDICTION' or task_name == 'GRAPHMATCHING':
            pipe = solutiondescription.SolutionDescription(problem)
            second_name = task_name + '2'
            pipe.initialize_solution(second_name)
            pipe.id = str(uuid.uuid4())
            pipe.add_outputs()
            solutions.append(pipe)
    elif task_name == 'COLLABORATIVEFILTERING':
        basic_sol = solutiondescription.SolutionDescription(problem)
        basic_sol.initialize_solution(task_name)
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)

        # Add a regression pipeline too
        pipe = solutiondescription.SolutionDescription(problem)
        pipe.initialize_solution('REGRESSION')
        pipe.id = str(uuid.uuid4())
        pipe.add_step('d3m.primitives.regression.random_forest.SKlearn', outputstep=pipe.index_denormalize + 3)
        solutions.append(pipe)
    elif task_name == 'LINKPREDICTIONTIMESERIES':
        tasks = ['LINKPREDICTION', 'LINKPREDICTION2']
        for t in tasks:
            basic_sol = solutiondescription.SolutionDescription(problem)
            basic_sol.initialize_solution(t)
            pipe = copy.deepcopy(basic_sol)
            pipe.id = str(uuid.uuid4())
            pipe.add_outputs()
            solutions.append(pipe)
        
        # Add a regression pipeline too
        primitives = ['d3m.primitives.regression.random_forest.SKlearn',
                      'd3m.primitives.regression.extra_trees.SKlearn',
                      'd3m.primitives.regression.gradient_boosting.SKlearn']
        for p in primitives:
            pipe = solutiondescription.SolutionDescription(problem)
            pipe.initialize_solution('REGRESSION')
            pipe.id = str(uuid.uuid4())
            outputstep = pipe.index_denormalize + 3
            pipe.add_step(p, outputstep=outputstep)
            solutions.append(pipe)
    elif task_name == 'OBJECTDETECTION':
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)
    else:
        logging.error("No matching solutions")

    if types_present is not None and task_name == 'CLASSIFICATION' and 'TIMESERIES' in types_present:
        pipe = solutiondescription.SolutionDescription(problem)
        pipe.initialize_solution('TIMESERIES2')
        pipe.id = str(uuid.uuid4())
        pipe.run_basic_solution(inputs=[dataset], input_step=1, output_step = 3, dataframe_step = 2)
        pipe.add_step('d3m.primitives.time_series_classification.k_neighbors.Kanine', inputstep=1, outputstep=3, dataframestep=2)
        solutions.append(pipe)
    if types_present is not None and ('AUDIO' in types_present or 'VIDEO' in types_present):
        pipe = solutiondescription.SolutionDescription(problem)
        pipe.initialize_solution('IMVADIO')
        pipe.id = str(uuid.uuid4())
        if task_name == 'CLASSIFICATION':
            pipe.add_step('d3m.primitives.classification.random_forest.SKlearn', outputstep=pipe.index_denormalize + 3, dataframestep=pipe.index_denormalize + 1)
        else:
            pipe.add_step('d3m.primitives.regression.random_forest.SKlearn', outputstep=pipe.index_denormalize + 3, dataframestep=pipe.index_denormalize + 1)
        solutions.append(pipe)

    end_solution_search = timer()
    time_used = end_solution_search - start_solution_search
    return (solutions, time_used)

def get_general_relational_solutions(task_name, types_present, rows, dataset, primitives, problem_metric, posLabel, problem):
    solutions = []
    if types_present is None:
        return solutions

    if task_name != "REGRESSION" and task_name != "CLASSIFICATION":
        return solutions
    
    if 'AUDIO' in types_present or \
       'VIDEO' in types_present or \
       'TEXT' in types_present or \
       'TIMESERIES' in types_present or \
       'IMAGE' in types_present or \
       rows > 100000:
        return solutions

    basic_sol = solutiondescription.SolutionDescription(problem)
    basic_sol.initialize_solution('GENERAL_RELATIONAL')

    start = timer()    
    try:
        basic_sol.run_basic_solution(inputs=[dataset], output_step=3, dataframe_step=2, primitive_dict=primitives, metric_type=problem_metric, posLabel=posLabel)
        total_cols = basic_sol.get_total_cols()
        logging.info("Total cols = %s", total_cols)

        if basic_sol is not None:
            if task_name == "REGRESSION":
                listOfSolutions = regressors_general_relational
            elif task_name == "CLASSIFICATION":
                listOfSolutions = classifiers_general_relational

            for python_path in listOfSolutions:
                pipe = copy.deepcopy(basic_sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step(python_path, outputstep=3, dataframestep=2)
                solutions.append(pipe)
    except:
        logging.error(sys.exc_info()[0])

    end = timer()
    logging.info("Time taken to run general solution: %s secs", end - start)
    return solutions 

def get_rpi_solutions(task_name, types_present, privileged, rows, dataset, primitives, problem_metric, posLabel, problem):
    solutions = []
    
    if types_present is None:
        return solutions

    if task_name != "REGRESSION" and task_name != "CLASSIFICATION":
        return solutions

    if 'AUDIO' in types_present or \
       'VIDEO' in types_present or \
       'TEXT' in types_present or \
       'TIMESERIES' in types_present or \
       'IMAGE' in types_present:
       return solutions

    basic_sol = solutiondescription.SolutionDescription(problem)
    basic_sol.set_privileged(privileged)
    basic_sol.initialize_RPI_solution(task_name)

    try:
        basic_sol.run_basic_solution(inputs=[dataset], output_step=4, dataframe_step=1, pprimitive_dict=primitives, metric_type=problem_metric, posLabel=posLabel)
        total_cols = basic_sol.get_total_cols()
        logging.info("Total cols = %s", total_cols)
    except:
        logging.error(sys.exc_info()[0])
        basic_sol = None

    if basic_sol is not None:
        if task_name == "REGRESSION":
            listOfSolutions = regressors_rpi
        elif task_name == "CLASSIFICATION":
            listOfSolutions = classifiers_rpi

        for python_path in listOfSolutions:
            if total_cols > 200:
                continue

            # Avoid expensive solutions!!!
            if 'gradient_boosting' in python_path and ((rows > 1000 and total_cols > 50) or (rows > 5000)):
                continue

            pipe = copy.deepcopy(basic_sol)
            pipe.id = str(uuid.uuid4())
            pipe.add_RPI_step(python_path, 4)
            solutions.append(pipe)

    return solutions

