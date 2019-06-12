import os, copy, uuid, sys
import solutiondescription
import logging

logging.basicConfig(level=logging.INFO)

task_paths = {
'TEXT': ['d3m.primitives.data_transformation.denormalize.Common',
         'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
         'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
         'd3m.primitives.data_cleaning.imputer.SKlearn',
         'd3m.primitives.feature_construction.corex_text.DSBOX',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'TIMESERIES': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX',
               'd3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'IMAGE': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.data_preprocessing.dataframe_to_tensor.DSBOX',
          'd3m.primitives.feature_extraction.resnet50_image_feature.DSBOX',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

#'VIDEO': ['d3m.primitives.data_transformation.denormalize.Common',
#          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
#          'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
#          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
#          'd3m.primitives.data_preprocessing.video_reader.DataFrameCommon',
#          'd3m.primitives.feature_extraction.resnext101_kinetics_video_features.VideoFeaturizer',
#          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],
'VIDEO': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
          'd3m.primitives.data_preprocessing.video_reader.DataFrameCommon',
          'd3m.primitives.feature_extraction.inceptionV3_image_feature.DSBOX',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'CLASSIFICATION': ['d3m.primitives.data_transformation.denormalize.Common',
                   'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                   'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                   'd3m.primitives.data_cleaning.imputer.SKlearn',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'SEMISUPERVISEDCLASSIFICATION': ['d3m.primitives.data_transformation.denormalize.Common',
                                 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                                 'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                                 'd3m.primitives.data_cleaning.imputer.SKlearn',
                                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                                 'd3m.primitives.semisupervised_classification.iterative_labeling.AutonBox',
                                 'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon'],

'REGRESSION': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
               'd3m.primitives.data_cleaning.imputer.SKlearn',
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
                    'd3m.primitives.object_detection.retina_net.JPLPrimitives'],

'LINKPREDICTION': ['d3m.primitives.data_transformation.graph_matching_parser.GraphMatchingParser',
                   'd3m.primitives.data_transformation.graph_transformer.GraphTransformer',
                   'd3m.primitives.link_prediction.link_prediction.LinkPrediction'],

'COMMUNITYDETECTION': ['d3m.primitives.community_detection.community_detection_parser.CommunityDetectionParser',
                       'd3m.primitives.classification.community_detection.CommunityDetection'],

'AUDIO': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.data_preprocessing.audio_reader.AudioReader',
          'd3m.primitives.data_preprocessing.channel_averager.ChannelAverager',
          'd3m.primitives.data_preprocessing.signal_dither.SignalDither',
          'd3m.primitives.time_series_segmentation.signal_framer.SignalFramer',
          'd3m.primitives.feature_extraction.signal_mfcc.SignalMFCC',
          'd3m.primitives.data_transformation.i_vector_extractor.IVectorExtractor',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'FALLBACK1': ['d3m.primitives.classification.gaussian_classification.MeanBaseline']}

classifiers = ['d3m.primitives.classification.bernoulli_naive_bayes.SKlearn',
               'd3m.primitives.classification.linear_discriminant_analysis.SKlearn',
               'd3m.primitives.classification.logistic_regression.SKlearn',
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
              'd3m.primitives.regression.linear_svr.SKlearn',
              'd3m.primitives.regression.random_forest.SKlearn',
              'd3m.primitives.regression.extra_trees.SKlearn',
              'd3m.primitives.regression.sgd.SKlearn',
              'd3m.primitives.regression.xgboost_gbtree.DataFrameCommon',
              'd3m.primitives.regression.gradient_boosting.SKlearn']

def get_solutions(task_name, dataset, primitives, problem):
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
        basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
        basic_sol.initialize_solution('FALLBACK1')
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)

    if task_name == 'TIMESERIESFORECASTING':
        task_name = 'REGRESSION'
    if task_name == 'VERTEXNOMINATION':
        task_name = 'VERTEXCLASSIFICATION'
    basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
    basic_sol.initialize_solution(task_name)

    if task_name == 'CLASSIFICATION' or task_name == 'REGRESSION' or task_name == 'SEMISUPERVISEDCLASSIFICATION':
        try:
            (types_present, total_cols, rows, categorical_atts, ordinal_atts, ok_to_denormalize, ok_to_impute, privileged) = solutiondescription.column_types_present(dataset)
            print(types_present)
            basic_sol.set_categorical_atts(categorical_atts)
            basic_sol.set_ordinal_atts(ordinal_atts)
            basic_sol.set_denormalize(ok_to_denormalize)
            basic_sol.set_impute(ok_to_impute)
            basic_sol.set_privileged(privileged)
            basic_sol.initialize_solution(task_name)
        except:
            logging.info(sys.exc_info()[0])
            basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
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
                    basic_sol.initialize_solution('TEXT')
                elif 'AUDIO' in types_present:
                    basic_sol.initialize_solution('AUDIO')
                elif 'VIDEO' in types_present:
                    basic_sol.initialize_solution('VIDEO')

                from timeit import default_timer as timer
                start = timer()
                basic_sol.run_basic_solution(inputs=[dataset])
                end = timer()
                logging.info("Time taken to run basic solution: %s seconds", end - start)
                time_used = end - start
                total_cols = basic_sol.get_total_cols()
                print("Total cols = ", total_cols)
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
            if total_cols > 500 and 'xgboost' in python_path:
                continue

            # SVM gets extremely expensive for >10k samples!!!
            if rows > 10000 and 'classification.svc.SKlearn' in python_path:
                continue
          
            pipe = copy.deepcopy(basic_sol) 
            pipe.id = str(uuid.uuid4())
            pipe.add_step(python_path)
            solutions.append(pipe)

        if len(listOfSolutions) == 0: # Currently hack for SSL to work
            pipe = basic_sol
            pipe.id = str(uuid.uuid4())
            pipe.add_outputs()
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
        pipe = solutiondescription.SolutionDescription(problem, static_dir)
        pipe.initialize_solution('CLASSIFICATION')
        pipe.id = str(uuid.uuid4())
        pipe.add_step('d3m.primitives.classification.random_forest.SKlearn')
        solutions.append(pipe)

        if task_name == 'GRAPHMATCHING':
            pipe = solutiondescription.SolutionDescription(problem, static_dir)
            pipe.initialize_solution('GRAPHMATCHING2')
            pipe.id = str(uuid.uuid4())
            pipe.add_outputs()
            solutions.append(pipe)
        if task_name == 'VERTEXCLASSIFICATION':
            pipe = solutiondescription.SolutionDescription(problem, static_dir)
            pipe.initialize_solution('VERTEXCLASSIFICATION2')
            pipe.id = str(uuid.uuid4())
            pipe.add_outputs()
            solutions.append(pipe)
    elif task_name == 'COLLABORATIVEFILTERING':
        pipe = copy.deepcopy(basic_sol)
        pipe.id = str(uuid.uuid4())
        pipe.add_outputs()
        solutions.append(pipe)

        # Add a regression pipeline too
        pipe = solutiondescription.SolutionDescription(problem, static_dir)
        pipe.initialize_solution('REGRESSION')
        pipe.id = str(uuid.uuid4())
        pipe.add_step('d3m.primitives.regression.random_forest.SKlearn')
        solutions.append(pipe)
    else:
        logging.info("No matching solutions")

    return (solutions, time_used)

