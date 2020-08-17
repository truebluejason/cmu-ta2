import csv
import sys
import json

# Verify we received expected command line arguments
if len(sys.argv) != 5 or len(sys.argv[1]) < 1 or len(sys.argv[2]) < 1 or len(sys.argv[3]) < 1 or len(sys.argv[4]) < 1:
    print("Record score requires three parameters:\n  - JSON pipeline file\n  - Scores file\n  - Dataset name\n\nExample: python record_score.py output/pipelines_ranked/8beaf9e1-3d1e-4cb8-82cb-810fac89e353.json scores.csv LL1_50words 0.572619284582\n")
    quit()

# Grab command line arguments
json_filepath = sys.argv[1]
json_filename = json_filepath.split('/')
json_filename = json_filename[len(json_filename) - 1]
scores_filepath = sys.argv[2]
dataset_name = sys.argv[3]
score = sys.argv[4]

# Open the pipeline & scores files
with open(json_filepath, 'r') as json_file, open(scores_filepath, 'a', newline='') as scores_file:
    # Parse the pipeline object from JSON
    pipeline = json.load(json_file)
    
    # Setup csv writer
    csvwriter = csv.writer(scores_file)
    
    # For the correct "main" primitive in the pipeline. Start from the end,
    # looking forward keywords indicating it is the main primitive. If not
    # found, we will default to the final primitive.
    mpi = -1  # main primitive index
    for i in range(len(pipeline['steps']) - 1, -1, -1):
        path = pipeline['steps'][i]['primitive']['python_path']
        if 'classification' in path or 'regression' in path or 'RPI' in path or 'FCN' in path:
            mpi = i
            break
    
    # Append a row to the CSV file
    csvwriter.writerow([
        dataset_name,
        json_filename,
        pipeline['steps'][mpi]['primitive']['name'],
        pipeline['steps'][mpi]['primitive']['python_path'],
        (json.dumps(pipeline['steps'][mpi]['hyperparams']) if 'hyperparams' in pipeline['steps'][mpi] else ''),
        pipeline['pipeline_rank'],
        score
    ])