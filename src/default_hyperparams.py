__author__ = "Saswati Ray"
__email__ = "sray@cs.cmu.edu"

__version__ = "0.1.0"

import os, json
import pandas as pd
import numpy as np

def set_default_hyperparameters(path, taskname):
    """     
    Retrieve the default hyperparameters to be set for a primitive.
    Parameters     ---------     
    path: Python path of a primitive
    """

    hyperparams = None
    # Set hyperparameters for specific primitives
    if path == 'd3m.primitives.data_cleaning.imputer.SKlearn':
        hyperparams = {}
        hyperparams['use_semantic_types'] = True
        hyperparams['return_result'] = 'replace'
        hyperparams['strategy'] = 'median'
        hyperparams['error_on_no_input'] = False

    if path == 'd3m.primitives.data_transformation.one_hot_encoder.SKlearn':
        hyperparams = {}
        hyperparams['use_semantic_types'] = True
        hyperparams['return_result'] = 'replace'
        hyperparams['handle_unknown'] = 'ignore'

    if path == 'd3m.primitives.data_preprocessing.robust_scaler.SKlearn':
        hyperparams = {}
        hyperparams['return_result'] = 'replace'

    if path == 'd3m.primitives.feature_construction.corex_text.DSBOX':
        hyperparams = {}
        hyperparams['threshold'] = 500

    if 'conditioner' in path:
        hyperparams = {}
        hyperparams['ensure_numeric'] = True
        hyperparams['maximum_expansion'] = 30

    if path == 'd3m.primitives.time_series_classification.k_neighbors.Kanine':
        hyperparams = {}
        hyperparams['n_neighbors'] = 1

    if path == 'd3m.primitives.clustering.k_means.Fastlvm':
        hyperparams = {}
        hyperparams['k'] = 100

    if 'adjacency_spectral_embedding.JHU' in path:
        hyperparams = {}
        hyperparams['max_dimension'] = 5
        hyperparams['use_attributes'] = True
        if 'LINK' in taskname:
            hyperparams['max_dimension'] = 2
            hyperparams['use_attributes'] = False
            hyperparams['which_elbow'] = 1

    if 'splitter' in path:
        hyperparams = {}
        if taskname == 'IMAGE' or taskname == 'IMAGE2' or taskname == 'AUDIO':
            hyperparams['threshold_row_length'] = 1200
        else:
            hyperparams['threshold_row_length'] = 50000

    if path == 'd3m.primitives.link_prediction.link_prediction.DistilLinkPrediction':
        hyperparams = {}
        hyperparams['metric'] = 'accuracy'

    if path == 'd3m.primitives.vertex_nomination.vertex_nomination.DistilVertexNomination':
        hyperparams = {}
        hyperparams['metric'] = 'accuracy'

    if path == 'd3m.primitives.graph_matching.seeded_graph_matching.DistilSeededGraphMatcher':
        hyperparams = {}
        hyperparams['metric'] = 'accuracy'

    if 'PCA' in path:
        hyperparams = {}
        hyperparams['n_components'] = 10

    if path == 'd3m.primitives.data_preprocessing.image_reader.Common':
        hyperparams = {}
        hyperparams['use_columns'] = [0,1]
        hyperparams['return_result'] = 'replace'

    if path == 'd3m.primitives.data_preprocessing.text_reader.Common':
        hyperparams = {}
        hyperparams['return_result'] = 'replace'

    if path == 'd3m.primitives.schema_discovery.profiler.Common':
        hyperparams = {}
        hyperparams['categorical_max_absolute_distinct_values'] = None

    if path == 'd3m.primitives.time_series_forecasting.arima.DSBOX':
        hyperparams = {}
        hyperparams['take_log'] = False

    if path == 'd3m.primitives.time_series_forecasting.lstm.DeepAR':
        hyperparams = {}
        hyperparams['epochs'] = 3

    if path == 'd3m.primitives.graph_clustering.gaussian_clustering.JHU':
        hyperparams = {}
        hyperparams['max_clusters'] = 10 

    if path == 'd3m.primitives.time_series_forecasting.esrnn.RNN':
        hyperparams = {}
        hyperparams['auto_tune'] = True
        hyperparams['output_size'] = 60

    return hyperparams
