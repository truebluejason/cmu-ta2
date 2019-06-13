#!/usr/bin/env bash

export D3MOUTPUTDIR=/zfsauton2/home/gwelter/code/cmu-ta2/output
export D3MSTATICDIR=/zfsauton2/home/gwelter/code/cmu-ta2/static
export D3MDATADIR=/home/scratch/gwelter/datasets/seed_datasets_current

export D3MTIMEOUT=1200
export D3MCPU=8

#datasets=(SEMI_1053_jm1)
#targets=(defects)
#metrics=(F1Macro)

datasets=(SEMI_155_pokerhand SEMI_1040_sylva_prior SEMI_1044_eye_movements SEMI_1053_jm1 SEMI_1217_click_prediction_small SEMI_1459_artificial_characters)
targets=(class label label defects click Class)
metrics=(F1Macro F1Macro F1Macro F1Macro F1Macro F1Macro)

rm scores.csv

for (( i=0; i<${#datasets[@]}; i++ ))
do

    echo "Running D3M CMU-TA2 on ${datasets[i]}"

    rm ./output/pipelines_ranked/* > /dev/null 2>&1
    rm ./output/pipelines_searched/* > /dev/null 2>&1

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
