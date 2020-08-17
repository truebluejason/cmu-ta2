__author__ = "Saswati Ray"
__email__ = "sray@cs.cmu.edu"

import os, sys

task_paths = {
'DISTILTEXT': ['d3m.primitives.data_transformation.denormalize.Common',
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

'TIMESERIES': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.schema_discovery.profiler.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
               'd3m.primitives.data_preprocessing.time_series_to_list.DSBOX',
               'd3m.primitives.feature_extraction.random_projection_timeseries_featurization.DSBOX',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'],

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

'CONDITIONER': ['d3m.primitives.data_transformation.denormalize.Common',
                'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                'd3m.primitives.schema_discovery.profiler.Common',
                'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                'd3m.primitives.data_transformation.column_parser.Common',
                'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                'd3m.primitives.data_transformation.conditioner.Conditioner',
                'd3m.primitives.data_preprocessing.feature_agglomeration.SKlearn'],

'SEMISUPERVISED': ['d3m.primitives.data_transformation.denormalize.Common',
                   'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                   'd3m.primitives.schema_discovery.profiler.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                   'd3m.primitives.data_transformation.column_parser.Common',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                   'd3m.primitives.data_cleaning.imputer.SKlearn'],

'SEMISUPERVISED_HDB': ['d3m.primitives.data_transformation.denormalize.Common',
                       'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                       'd3m.primitives.schema_discovery.profiler.Common',
                       'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                       'd3m.primitives.data_transformation.column_parser.Common',
                       'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                       'd3m.primitives.clustering.hdbscan.Hdbscan',
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
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                 'd3m.primitives.time_series_forecasting.vector_autoregression.VAR'],

'FORECASTING4': ['d3m.primitives.data_transformation.denormalize.Common',
                 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                 'd3m.primitives.schema_discovery.profiler.Common',
                 'd3m.primitives.data_transformation.column_parser.Common',
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                 'd3m.primitives.time_series_forecasting.lstm.DeepAR'],

'FORECASTING3': ['d3m.primitives.data_transformation.denormalize.Common',
                 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                 'd3m.primitives.schema_discovery.profiler.Common',
                 'd3m.primitives.data_transformation.column_parser.Common',
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                 'd3m.primitives.data_cleaning.imputer.SKlearn',
                 'd3m.primitives.data_transformation.grouping_field_compose.Common',
                 'd3m.primitives.time_series_forecasting.esrnn.RNN',
                 'd3m.primitives.data_transformation.construct_predictions.Common'],

'REGRESSION': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.schema_discovery.profiler.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
               'd3m.primitives.data_transformation.column_parser.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
               'd3m.primitives.data_cleaning.imputer.SKlearn'],

'FORE_REGRESSION': ['d3m.primitives.data_transformation.denormalize.Common',
                    'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                    'd3m.primitives.schema_discovery.profiler.Common',
                    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                    'd3m.primitives.data_transformation.column_parser.Common',
                    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                    'd3m.primitives.data_cleaning.imputer.SKlearn'],

'PIPELINE_RPI': ['d3m.primitives.data_transformation.denormalize.Common',
                 'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                 'd3m.primitives.schema_discovery.profiler.Common',
                 'd3m.primitives.data_transformation.column_parser.Common',
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                 'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common'],

'NOTUNE_PIPELINE_RPI': ['d3m.primitives.data_transformation.denormalize.Common',
                        'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                        'd3m.primitives.schema_discovery.profiler.Common',
                        'd3m.primitives.data_transformation.column_parser.Common',
                        'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
                        'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                        'd3m.primitives.feature_selection.joint_mutual_information.AutoRPI',
                        'd3m.primitives.data_cleaning.imputer.SKlearn'],

'CLUSTERING': ['d3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.schema_discovery.profiler.Common',
               'd3m.primitives.data_transformation.column_parser.Common',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
               'd3m.primitives.clustering.k_means.Fastlvm',
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

'VERTEXCLASSIFICATION2': ['d3m.primitives.data_transformation.load_graphs.JHU',
                          'd3m.primitives.data_preprocessing.largest_connected_component.JHU',
                          'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
                          'd3m.primitives.classification.gaussian_classification.JHU'],

'VERTEXCLASSIFICATION': ['d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                         'd3m.primitives.vertex_nomination.vertex_nomination.DistilVertexNomination'],

'OBJECTDETECTION': ['d3m.primitives.data_transformation.denormalize.Common',
                    'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                    'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
                    'd3m.primitives.feature_extraction.yolo.DSBOX'],

'OBJECTDETECTION2': ['d3m.primitives.data_transformation.denormalize.Common',
                     'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                     'd3m.primitives.object_detection.retina_net.ObjectDetectionRN'],

'LINKPREDICTION2': ['d3m.primitives.link_prediction.data_conversion.JHU', 
                    'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
                    'd3m.primitives.link_prediction.rank_classification.JHU'],

'LINKPREDICTION': ['d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                    'd3m.primitives.link_prediction.link_prediction.DistilLinkPrediction'],

'COMMUNITYDETECTION': ['d3m.primitives.data_transformation.load_single_graph.DistilSingleGraphLoader',
                       'd3m.primitives.community_detection.community_detection.DistilCommunityDetection'],

'COMMUNITYDETECTION2': ['d3m.primitives.data_transformation.load_graphs.JHU',
	                'd3m.primitives.data_preprocessing.largest_connected_component.JHU',
                        'd3m.primitives.data_transformation.adjacency_spectral_embedding.JHU',
	                'd3m.primitives.graph_clustering.gaussian_clustering.JHU'],

'IMVADIO': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.schema_discovery.profiler.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
          'd3m.primitives.data_transformation.add_semantic_types.Common',
          'd3m.primitives.data_transformation.column_parser.Common',
          'd3m.primitives.data_preprocessing.text_reader.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',
          'd3m.primitives.data_transformation.encoder.DistilTextEncoder'],

'AUDIO': ['d3m.primitives.data_preprocessing.audio_reader.DistilAudioDatasetLoader',
          'd3m.primitives.data_transformation.column_parser.Common',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.Common',  # targets
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
               'd3m.primitives.classification.mlp.SKlearn',
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
              'd3m.primitives.regression.bagging.SKlearn',
              'd3m.primitives.regression.mlp.SKlearn',
              'd3m.primitives.regression.gradient_boosting.SKlearn']

regressors_rpi = ['d3m.primitives.regression.random_forest.SKlearn',
                  'd3m.primitives.regression.extra_trees.SKlearn',
                  'd3m.primitives.regression.gradient_boosting.SKlearn']

classifiers_rpi = ['d3m.primitives.classification.random_forest.SKlearn',
                   'd3m.primitives.classification.extra_trees.SKlearn',
                   'd3m.primitives.classification.gradient_boosting.SKlearn',
                   'd3m.primitives.classification.linear_discriminant_analysis.SKlearn']

sslVariants = ['d3m.primitives.classification.gradient_boosting.SKlearn',
               'd3m.primitives.classification.extra_trees.SKlearn',
               'd3m.primitives.classification.random_forest.SKlearn',
               'd3m.primitives.classification.bagging.SKlearn',
               'd3m.primitives.classification.linear_svc.SKlearn',
               'd3m.primitives.classification.svc.SKlearn']
