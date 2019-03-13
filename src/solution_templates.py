import os, copy, uuid, sys
import solutiondescription
import logging

logging.basicConfig(level=logging.INFO)

task_paths = {
'TEXT': ['d3m.primitives.data_transformation.denormalize.Common',
         'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
         'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
         'd3m.primitives.feature_construction.corex_text.CorexText',
         'd3m.primitives.data_cleaning.imputer.SKlearn',
         'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'TIMESERIES': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
               'd3m.primitives.data_preprocessing.TimeseriesToList.DSBOX',
               'd3m.primitives.feature_extraction.RandomProjectionTimeSeriesFeaturization.DSBOX',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'IMAGE': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
          'd3m.primitives.data_preprocessing.DataFrameToTensor.DSBOX',
          'd3m.primitives.feature_extraction.ResNet50ImageFeature.DSBOX',
          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'CLASSIFICATION': ['d3m.primitives.data_transformation.denormalize.Common',
                   'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
                   'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                   'd3m.primitives.data_cleaning.imputer.SKlearn',
                   'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'REGRESSION': ['d3m.primitives.data_transformation.denormalize.Common',
               'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
               'd3m.primitives.data_cleaning.imputer.SKlearn',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'CLUSTERING': ['d3m.primitives.data_transformation.dataset_to_dataframe.Common',
               'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
               'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
               'd3m.primitives.cmu.fastlvm.CoverTree',
               'd3m.primitives.data_transformation.construct_predictions.DataFrameCommon'],

'GRAPHMATCHING': ['d3m.primitives.sri.psl.GraphMatchingLinkPrediction'],

'GRAPHMATCHING2': ['d3m.primitives.graph_matching.seeded_graph_matching.JHU'],

'COLLABORATIVEFILTERING': ['d3m.primitives.sri.psl.CollaborativeFilteringLinkPrediction'],

'VERTEXNOMINATION': ['d3m.primitives.sri.graph.VertexNominationParser',
                     'd3m.primitives.sri.psl.VertexNomination'],

'LINKPREDICTION': ['d3m.primitives.sri.graph.GraphMatchingParser',
                   'd3m.primitives.sri.graph.GraphTransformer',
                   'd3m.primitives.sri.psl.LinkPrediction'],

'COMMUNITYDETECTION': ['d3m.primitives.sri.graph.CommunityDetectionParser',
                       'd3m.primitives.sri.psl.CommunityDetection'],

'TIMESERIESFORECASTING': ['d3m.primitives.data_transformation.dataset_to_dataframe.Common',
                          'd3m.primitives.data_transformation.column_parser.DataFrameCommon',
                          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon',
                          'd3m.primitives.data_transformation.extract_columns_by_semantic_types.DataFrameCommon'],

'AUDIO': ['d3m.primitives.data_transformation.denormalize.Common',
          'd3m.primitives.data_transformation.dataset_to_dataframe.Common',
          'd3m.primitives.bbn.time_series.AudioReader',
          'd3m.primitives.bbn.time_series.ChannelAverager',
          'd3m.primitives.bbn.time_series.SignalDither',
          'd3m.primitives.bbn.time_series.SignalFramer',
          'd3m.primitives.bbn.time_series.SignalMFCC',
          'd3m.primitives.bbn.time_series.IVectorExtractor',
          'd3m.primitives.bbn.time_series.TargetsReader'],

'FALLBACK1': ['d3m.primitives.sri.baseline.MeanBaseline'],

'FALLBACK2': ['d3m.primitives.sri.psl.GeneralRelationalDataset']}

def get_solutions(task_name, dataset, primitives, problem):
    """
    Get a list of available solutions(pipelines) for the specified task
    Used by both TA2 in "search" phase and TA2-TA3
    """
    solutions = []

    try:
        static_dir = os.environ['D3MSTATICDIR']
    except:
        static_dir = None

    basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
    basic_sol.initialize_solution(task_name)

    if task_name == 'CLASSIFICATION' or task_name == 'REGRESSION':
        try:
            (types_present, total_cols, rows, categorical_atts) = solutiondescription.column_types_present(dataset)
            print(types_present)
            basic_sol.set_categorical_atts(categorical_atts)
            basic_sol.initialize_solution(task_name)
        except:
            logging.info(sys.exc_info()[0])
            basic_sol = None
            types_present = None

        if types_present is not None:
            if 'TIMESERIES' in types_present:
                basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
                basic_sol.initialize_solution('TIMESERIES')
            elif 'IMAGE' in types_present:
                basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
                basic_sol.initialize_solution('IMAGE')
            elif 'TEXT' in types_present:
                basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
                basic_sol.initialize_solution('TEXT')
            elif 'AUDIO' in types_present:
                basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
                basic_sol.initialize_solution('AUDIO')

            try:
                basic_sol.run_basic_solution(inputs=[dataset])
                total_cols = basic_sol.get_total_cols()
                print("Total cols = ", total_cols)
            except:
                logging.info(sys.exc_info()[0])
                basic_sol = None

        # Iterate through primitives which match task type for populative pool of solutions
        for classname, p in primitives.items():
            if p.primitive_class.family == task_name and basic_sol is not None:
                python_path = p.primitive_class.python_path
                if 'd3m.primitives.sri.' in python_path or 'JHU' in python_path or 'lupi_svm' in python_path or 'bbn' in python_path:
                    continue

                if 'Find_projections' in python_path and (total_cols > 20 or rows > 10000):
                    continue

                pipe = copy.deepcopy(basic_sol)
                pipe.id = str(uuid.uuid4())
                pipe.add_step(p.primitive_class.python_path)
                solutions.append(pipe)
    elif task_name == 'VERTEXNOMINATION' or \
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
    elif task_name == 'COLLABORATIVEFILTERING' or \
         task_name == 'TIMESERIESFORECASTING':
        if task_name == 'TIMESERIESFORECASTING':
            basic_sol.add_step('d3m.primitives.sri.psl.RelationalTimeseries')
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

    basic_sol = solutiondescription.SolutionDescription(problem, static_dir)
    basic_sol.initialize_solution('FALLBACK1')
    pipe = copy.deepcopy(basic_sol)
    pipe.id = str(uuid.uuid4())
    pipe.add_outputs()
    solutions.append(pipe)

    return solutions

