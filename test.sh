#!/usr/bin/env bash

export D3MOUTPUTDIR=/home/sray/cmu-ta2/cmu-ta2/output
export D3MSTATICDIR=/home/sray
export D3MDATADIR=/home/sray/DARPA_D3M

export D3MTIMEOUT=1200
export D3MCPU=8

datasets=(SEMI_1053_jm1)
targets=(defects)
metrics=(F1Macro)

rm scores.csv

for (( i=0; i<1; i++ ))
do
    rm ./output/pipelines_ranked/*
    rm ./output/pipelines_searched/*

    export D3MINPUTDIR="$D3MDATADIR/${datasets[i]}"
    export D3MPROBLEMPATH="$D3MDATADIR/${datasets[i]}/TRAIN/problem_TRAIN/problemDoc.json"

    ./src/main.py search

    FILE=$D3MDATADIR/${datasets[i]}/SCORE/targets.csv
    if [ -f $FILE ]; then
       echo "SCORE/targets.csv exists"
    else
       FILE=$D3MDATADIR/${datasets[i]}/SCORE/dataset_SCORE/tables/learningData.csv
    fi

    for jsonfile in ./output/pipelines_ranked/*
    do
        echo ${jsonfile}
        grep "pipeline_rank" ${jsonfile} >> scores.csv
        grep "classification" ${jsonfile} | awk -F':' '{print $2}' >> scores.csv
        grep "regression" ${jsonfile} | awk -F':' '{print $2}' >> scores.csv
        python sample.py $D3MDATADIR/${datasets[i]}/TRAIN/problem_TRAIN/problemDoc.json $D3MDATADIR/${datasets[i]}/TRAIN/dataset_TRAIN/datasetDoc.json ${jsonfile} $D3MDATADIR/${datasets[i]}/TEST/dataset_TEST/datasetDoc.json

        python evaluate_score.py $FILE results.csv ${targets[i]} ${metrics[i]} >> scores.csv

	done
done

cat scores.csv
