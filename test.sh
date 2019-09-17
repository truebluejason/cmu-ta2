#!/usr/bin/env bash

export D3MOUTPUTDIR=/zfsauton2/home/gwelter/code/cmu-ta2/output
export D3MSTATICDIR=/zfsauton2/home/gwelter/code/cmu-ta2/static
export D3MDATADIR=/home/scratch/gwelter/datasets/seed_datasets_current

export D3MTIMEOUT=1200
export D3MCPU=8

datasets=(SEMI_155_pokerhand SEMI_1040_sylva_prior SEMI_1044_eye_movements SEMI_1053_jm1 SEMI_1217_click_prediction_small SEMI_1459_artificial_characters)
targets=(class label label defects click Class)
metrics=(F1Macro F1Macro F1Macro F1Macro F1Macro F1Macro)

datasets=(LL1_50words LL1_Adiac LL1_ArrowHead LL1_CinC_ECG_torso LL1_Cricket_Y LL1_ECG200 LL1_ElectricDevices LL1_FISH LL1_FaceFour LL1_FordA LL1_HandOutlines LL1_Haptics LL1_ItalyPowerDemand LL1_Meat LL1_OSULeaf)
targets=(label label label label label label label label label label label label label label label)
metrics=(F1Macro F1Macro F1Macro F1Macro F1Macro F1 F1Macro F1Macro F1Macro F1 F1 F1Macro F1 F1Macro F1Macro)

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
        grep "RPI" ${jsonfile} | awk -F':' '{print $2}' >> scores.csv

        python -m d3m.runtime fit-produce -p ${jsonfile} -r $D3MDATADIR/${datasets[i]}/TRAIN/problem_TRAIN/problemDoc.json -i $D3MDATADIR/${datasets[i]}/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MDATADIR/${datasets[i]}/TEST/dataset_TEST/datasetDoc.json -o results.csv
        python evaluate_score.py $FILE results.csv ${targets[i]} ${metrics[i]} >> scores.csv

	done
done

cat scores.csv
