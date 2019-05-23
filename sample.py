from d3m.metadata import base as metadata_base, hyperparams as hyperparams_module, pipeline as pipeline_module, problem
from d3m.metadata.base import Context
from d3m.container.dataset import Dataset, D3MDatasetLoader
from d3m.runtime import Runtime
import os, sys

# Loading problem description.
problem_description = problem.parse_problem_description(sys.argv[1])

# Loading dataset.
path = 'file://{uri}'.format(uri=os.path.abspath(sys.argv[2]))
dataset = D3MDatasetLoader()
dataset = dataset.load(dataset_uri=path)

# Loading pipeline description file.
with open(sys.argv[3], 'r') as file:
    pipeline_description = pipeline_module.Pipeline.from_json(string_or_file=file)

# Creating an instance on runtime with pipeline description and problem description.
runtime = Runtime(pipeline=pipeline_description, problem_description=problem_description, is_standard_pipeline=True, context=Context.EVALUATION)

# Fitting pipeline on input dataset.
fit_outputs = runtime.fit(inputs=[dataset])

path = 'file://{uri}'.format(uri=os.path.abspath(sys.argv[4]))
dataset = D3MDatasetLoader()
dataset = dataset.load(dataset_uri=path)
outputs = runtime.produce(inputs=[dataset]).values
with open("results.csv", 'w') as outputFile:
    outputs['outputs.0'].to_csv(outputFile, header=True, index=False)
