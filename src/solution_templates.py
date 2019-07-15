import os, copy, uuid, sys
import solutiondescription
import logging

logging.basicConfig(level=logging.INFO)

task_paths = {
'TEXTCLASSIFICATION': ['d3m.primitives.data_transformation.denormalize.Common',
                       'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                       'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                       'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                       'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                       'd3m.primitives.data_cleaning.imputer.SKlearn',
                       'd3m.primitives.data_transformation.encoder.DistilTextEncoder'],

'TEXT': ['d3m.primitives.data_transformation.denormalize.Common',
         'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
         'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
         'd3m.primitives.data_cleaning.imputer.SKlearn',
         'd3m.primitives.feature_construction.corex_text.DSBOX'],

'TIMESERIES': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
               'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX',
               'd3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX'],

'TIMESERIES2': ['d3m.primitives.data_transformation.denormalize.Common',
                'd3m.primitives.time_series_classification.k_neighbors.Kanine'],

'TIMESERIES3': ['d3m.primitives.data_transformation.data_cleaning.DistilRaggedDatasetLoader',
                'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                'd3m.primitives.data_transformation.data_cleaning.DistilTimeSeriesReshaper',
                'd3m.primitives.learner.random_forest.DistilTimeSeriesNeighboursPrimitive',
                'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon'],

'IMAGE': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
          'd3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX',
          'd3m.primitives.feature_extraction.resnet50_image_feature.DSBOX'],

'VIDEO': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
          'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
          'd3m.primitives.feature_extraction.resnext101_kinetics_video_features.VideoFeaturizer'],

'CLASSIFICATION': ['d3m.primitives.data_transformation.denormalize.Common',
                   'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                   'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                   'd3m.primitives.data_cleaning.imputer.SKlearn'],

'SEMISUPERVISEDCLASSIFICATION': ['d3m.primitives.data_transformation.denormalize.Common',
                                 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                                 'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                                 'd3m.primitives.data_cleaning.imputer.SKlearn'],

'REGRESSION': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
               'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
               'd3m.primitives.data_cleaning.imputer.SKlearn'],

'GENERAL_RELATIONAL': ['d3m.primitives.data_preprocessing.splitter.DSBOX',
                       'd3m.primitives.data_transformation.denormalize.Common',
                       'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                       'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                       'd3m.primitives.classification.general_relational_dataset.GeneralRelationalDataset',
                       'd3m.primitives.data_transformation.cast_to_type.Common'],

'PIPELINE_RPI': ['d3m.primitives.data_transformation.denormalize.Common',
                 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                 'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'CLUSTERING': ['d3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
               'd3m.primitives.clustering.k_means.Fastlvm',
               'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon'],

'GRAPHMATCHING': ['d3m.primitives.link_prediction.graph_matching_link_prediction.GraphMatchingLinkPrediction'],

'GRAPHMATCHING2': ['d3m.primitives.graph_matching.seeded_graph_matching.JHU'],

'COLLABORATIVEFILTERING': ['d3m.primitives.link_prediction.collaborative_filtering_link_prediction.CollaborativeFilteringLinkPrediction'],

'VERTEXCLASSIFICATION2': ['d3m.primitives.data_preprocessing.largest_connected_component.JHU',
                      'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
                      'd3m.primitives.classification.gaussian_classification.JHU'],

'VERTEXCLASSIFICATION': ['d3m.primitives.data_transformation.vertex_classification_parser.VertexClassificationParser',
                     'd3m.primitives.classification.vertex_nomination.VertexClassification'],

'OBJECTDETECTION': ['d3m.primitives.data_transformation.denormalize.Common',
                    'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                    'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                    'd3m.primitives.feature_extraction.yolo.DSBOX'],

'LINKPREDICTION': ['d3m.primitives.data_transformation.graph_matching_parser.GraphMatchingParser',
                   'd3m.primitives.data_transformation.graph_transformer.GraphTransformer',
                   'd3m.primitives.link_prediction.link_prediction.LinkPrediction'],

'LINKPREDICTION2': ['d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                    'd3m.primitives.data_transformation.link_prediction.DistilLinkPrediction'],

'COMMUNITYDETECTION': ['d3m.primitives.community_detection.community_detection_parser.CommunityDetectionParser',
                       'd3m.primitives.classification.community_detection.CommunityDetection'],

'COMMUNITYDETECTION2': ['d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                       'd3m.primitives.data_transformation.community_detection.DistilCommunityDetection'],

'AUDIO': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
          'd3m.primitives.data_preprocessing.audio_reader.BBN',
          'd3m.primitives.data_preprocessing.channel_averager.BBN',
          'd3m.primitives.data_preprocessing.signal_dither.BBN',
          'd3m.primitives.time_series_segmentation.signal_framer.BBN',
          'd3m.primitives.feature_extraction.signal_mfcc.BBN',
          'd3m.primitives.data_transformation.i_vector_extractor.BBN'],

'FALLBACK1': ['d3m.primitives.classification.gaussian_classification.MeanBaseline']}

classifiers = ['d3m.primitives.classification.bernoulli_naive_bayes.SKlearn',
               'd3m.primitives.classification.linear_discriminant_analysis.SKlearn',
               'd3m.primitives.classification.logistic_regression.SKlearn',
               'd3m.primitives.classification.ada_boost.SKlearn',
               'd3m.primitives.classification.linear_svc.SKlearn',
               'd3m.primitives.classification.extra_trees.SKlearn',
               'd3m.primitives.classification.random_forest.SKlearn',
               'd3m.primitives.classification.bagging.SKlearn',
               'd3m.primitives.classification.gaussian_naive_bayes.SKlearn',
               'd3m.primitives.classification.sgd.SKlearn',
               'd3m.primitives.classification.svc.SKlearn',
               'd3m.primitives.classification.xgboost_gbtree.DataFrameCommon',
               'd3m.primitives.classification.gradient_boosting.SKlearn']

regressors = ['d3m.primitives.regression.ridge.SKlearn',
              'd3m.primitives.regression.lasso.SKlearn',
              'd3m.primitives.regression.elastic_net.SKlearn',
              'd3m.primitives.regression.lasso_cv.SKlearn',
              'd3m.primitives.regression.ada_boost.SKlearn',
              'd3m.primitives.regression.linear_svr.SKlearn',
              'd3m.primitives.regression.random_forest.SKlearn',
              'd3m.primitives.regression.extra_trees.SKlearn',
              'd3m.primitives.regression.sgd.SKlearn',
              'd3m.primitives.regression.xgboost_gbtree.DataFrameCommon',
              'd3m.primitives.regression.gradient_boosting.SKlearn']

regressors_rpi = ['d3m.primitives.regression.random_forest.SKlearn',
                  'd3m.primitives.regression.extra_trees.SKlearn',
                  'd3m.primitives.regression.gradient_boosting.SKlearn']

classifiers_rpi = ['d3m.primitives.classification.random_forest.SKlearn',
                   'd3m.primitives.classification.extra_trees.SKlearn',
                   'd3m.primitives.classification.bagging.SKlearn',
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

def get_solutions(task_name, dataset, primitives, problem_metric, posLabel):
    """
    Get a list of available solutions(pipelines) for the specified task
    Used by both TA2 in "search" phase and TA2-TA3
    """
    solutions = []
    time_used = 0

    try:
        static_dir = os.environ['D3MSTATICDIR']
    except:
        static_dir = None

    if task_name != 'SEMISUPERVISEDCLASSIFICATION':
        basic_sol = solutiondescription.SolutionDescription(None, static_dir)
        basic_sol.initialize_solution('FALLBACK1')
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)

    if task_name == 'TIMESERIESFORECASTING':
        task_name = 'REGRESSION'
    if task_name == 'VERTEXNOMINATION':
        task_name = 'VERTEXCLASSIFICATION'
    basic_sol = solutiondescription.SolutionDescription(None, static_dir)
    basic_sol.initialize_solution(task_name)

    types_present = []
    text_prop = 1.0
    if task_name == 'CLASSIFICATION' or task_name == 'REGRESSION' or task_name == 'SEMISUPERVISEDCLASSIFICATION':
        try:
            (types_present, total_cols, rows, categorical_atts, ordinal_atts, ok_to_denormalize, ok_to_impute, privileged, text_prop) = solutiondescription.column_types_present(dataset)
            logging.info(types_present)
            basic_sol.set_categorical_atts(categorical_atts)
            basic_sol.set_ordinal_atts(ordinal_atts)
            basic_sol.set_denormalize(ok_to_denormalize)
            basic_sol.set_impute(ok_to_impute)
            basic_sol.set_privileged(privileged)
            basic_sol.initialize_solution(task_name)
        except:
            logging.info(sys.exc_info()[0])
            basic_sol = solutiondescription.SolutionDescription(None, static_dir)
            types_present = None
            rows = 0

        if types_present is not None:
            if len(types_present) == 1 and types_present[0] == 'FILES':
                types_present[0] = 'TIMESERIES' 
            try:
                if 'TIMESERIES' in types_present:
                    basic_sol.initialize_solution('TIMESERIES')
                elif 'IMAGE' in types_present:
                    basic_sol.initialize_solution('IMAGE')
                elif 'TEXT' in types_present:
                    if task_name == 'CLASSIFICATION' and text_prop < 0.2:
                        basic_sol.initialize_solution('TEXTCLASSIFICATION')
                    else:
                        basic_sol.initialize_solution('TEXT')
                elif 'AUDIO' in types_present:
                    basic_sol.initialize_solution('AUDIO')
                elif 'VIDEO' in types_present:
                    basic_sol.initialize_solution('VIDEO')

                from timeit import default_timer as timer
                start = timer()
                basic_sol.run_basic_solution(inputs=[dataset], output_step=2)
                end = timer()
                logging.info("Time taken to run basic solution: %s secs", end - start)
                time_used = end - start
                total_cols = basic_sol.get_total_cols()
                logging.info("Total cols = %s", total_cols)
            except:
                logging.info(sys.exc_info()[0])
                basic_sol = None

        # Iterate through primitives which match task type for populative pool of solutions
        listOfSolutions = []
        if basic_sol is not None:
            if task_name == "REGRESSION":
                listOfSolutions = regressors
            elif task_name == "CLASSIFICATION":
                listOfSolutions = classifiers

        for python_path in listOfSolutions:
            if (total_cols > 500 or rows > 100000) and 'xgboost' in python_path:
                continue

            # SVM gets extremely expensive for >10k samples!!!
            if rows > 10000 and 'classification.svc.SKlearn' in python_path:
                continue 
            
            pipe = copy.deepcopy(basic_sol) 
            pipe.id = str(uuid.uuid4())
            pipe.add_step(python_path)
            solutions.append(pipe)

        # Try general relational pipelines
        (general_solutions, general_time_used) = get_general_relational_solutions(task_name, dataset, primitives, problem_metric, posLabel, static_dir)
        solutions = solutions + general_solutions
        time_used = time_used + general_time_used

        # Try RPI primitives for tabular datasets
        rpi_solutions = get_rpi_solutions(task_name, types_present, rows, dataset, primitives, problem_metric, posLabel, static_dir)
        solutions = solutions + rpi_solutions

        if task_name == 'SEMISUPERVISEDCLASSIFICATION':
            # Iterate through variants of possible blackbox hyperparamets.
            for variant in sslVariants:
                pipe = copy.deepcopy(basic_sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step('d3m.primitives.semisupervised_classification.iterative_labeling.AutonBox')
                pipe.add_ssl_variant(variant)
                solutions.append(pipe)

    elif task_name == 'VERTEXCLASSIFICATION' or \
         task_name == 'COMMUNITYDETECTION' or \
         task_name == 'GRAPHMATCHING' or \
         task_name == 'LINKPREDICTION' or \
         task_name == 'CLUSTERING':
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)
        
        # Add a classification pipeline too
        pipe = solutiondescription.SolutionDescription(None, static_dir)
        pipe.initialize_solution('CLASSIFICATION')
        pipe.id = str(uuid.uuid4())
        pipe.add_step('d3m.primitives.classification.extra_trees.SKlearn')
        solutions.append(pipe)

        if task_name == 'GRAPHMATCHING' or \
           task_name == 'VERTEXCLASSIFICATION' or \
           task_name == 'COMMUNITYDETECTION' or \
           task_name == 'LINKPREDICTION':
            pipe = solutiondescription.SolutionDescription(None, static_dir)
            second_name = task_name + '2'
            pipe.initialize_solution(second_name)
            pipe.id = str(uuid.uuid4())
            pipe.add_outputs()
            solutions.append(pipe)
    elif task_name == 'COLLABORATIVEFILTERING':
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)

        # Add a regression pipeline too
        pipe = solutiondescription.SolutionDescription(None, static_dir)
        pipe.initialize_solution('REGRESSION')
        pipe.id = str(uuid.uuid4())
        pipe.add_step('d3m.primitives.regression.extra_trees.SKlearn')
        solutions.append(pipe)
    else:
        logging.info("No matching solutions")

    if task_name == 'CLASSIFICATION' and 'TIMESERIES' in types_present:
        pipe = solutiondescription.SolutionDescription(None, static_dir)
        pipe.initialize_solution('TIMESERIES2')
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)

    return (solutions, time_used)

def get_general_relational_solutions(task_name, dataset, primitives, problem_metric, posLabel, static_dir):
    solutions = []
    basic_sol = solutiondescription.SolutionDescription(None, static_dir)
    basic_sol.initialize_solution('GENERAL_RELATIONAL')

    from timeit import default_timer as timer
    start = timer()    
    try:
        basic_sol.run_basic_solution(inputs=[dataset], output_step=3, primitive_dict=primitives, metric_type=problem_metric, posLabel=posLabel)
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
        logging.info(sys.exc_info()[0])

    end = timer()
    time_used = end - start
    logging.info("Time taken to run general solution: %s secs", end - start)
    return (solutions, time_used)  

def get_rpi_solutions(task_name, types_present, rows, dataset, primitives, problem_metric, posLabel, static_dir):
    solutions = []

    if task_name != "REGRESSION" and task_name != "CLASSIFICATION":
        return solutions

    if 'AUDIO' in types_present or \
       'VIDEO' in types_present or \
       'TEXT' in types_present or \
       'TIMESERIES' in types_present or \
       'IMAGE' in types_present or \
       rows > 100000:
       return solutions

    basic_sol = solutiondescription.SolutionDescription(None, static_dir)
    basic_sol.initialize_RPI_solution(task_name)

    try:
        basic_sol.run_basic_solution(inputs=[dataset], output_step=3, primitive_dict=primitives, metric_type=problem_metric, posLabel=posLabel)
        total_cols = basic_sol.get_total_cols()
        logging.info("Total cols = %s", total_cols)
    except:
        logging.info(sys.exc_info()[0])
        end = timer()
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
            if ('gradient_boosting' in python_path or 'bagging' in python_path) and ((rows > 1000 and total_cols > 50) or (rows > 5000)):
                continue

            pipe = copy.deepcopy(basic_sol)
            pipe.id = str(uuid.uuid4())
            pipe.add_RPI_step(python_path, 3)
            solutions.append(pipe)

    return solutions

