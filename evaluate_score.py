#!/usr/bin/env python3

import pandas as pd
import sys
from sklearn import metrics
import math

target = sys.argv[3]
metric = sys.argv[4]
Ytest = pd.read_csv(sys.argv[1])[target]
predictions = pd.read_csv(sys.argv[2])[target]

if metric == 'MSE' or metric == 'meanSquaredError':
    print(metrics.mean_squared_error(Ytest, predictions))
elif metric == 'F1Macro' or metric == 'f1Macro':
    print(metrics.f1_score(Ytest, predictions, average='macro'))
elif metric == 'F1' or metric == 'f1':
    pos_label=sys.argv[5]
    if pos_label == '1':
        pos_label = int(pos_label)
    print(metrics.f1_score(Ytest, predictions, pos_label=pos_label))
elif metric == 'MAE' or metric == 'meanAbsoluteError':
    print(metrics.mean_absolute_error(Ytest, predictions))
elif metric == 'ACC' or metric == 'accuracy':
    print(metrics.accuracy_score(Ytest, predictions))
elif metric == 'NMI' or metric == 'normalizedMutualInformation':
    print(metrics.normalized_mutual_info_score(Ytest, predictions))
elif metric == 'RMSE' or metric == 'rootMeanSquaredError':
    print(math.sqrt(metrics.mean_squared_error(Ytest, predictions)))
