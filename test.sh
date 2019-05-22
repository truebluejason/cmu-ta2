
export D3MOUTPUTDIR=/home/sray/cmu-ta2/cmu-ta2/output
export D3MTIMEOUT=1200
export D3MCPU=8
export D3MSTATICDIR=/home/sray

datasets=(185_baseball)
targets=(Hall_of_Fame)
metrics=(F1Macro)

for (( i=0; i<1; i++ ))
do
        rm ./output/pipelines_ranked/*
        rm ./output/pipelines_searched/*

	export D3MINPUTDIR="/home/sray/DARPA_D3M/${datasets[i]}"
        export D3MPROBLEMPATH="/home/sray/DARPA_D3M/${datasets[i]}/TRAIN/problem_TRAIN/problemDoc.json"

	./src/main.py search

        for jsonfile in ./output/pipelines_ranked/*
        do 
                echo ${jsonfile}
                grep "pipeline_rank" ${jsonfile} >> scores.csv
                grep "classification" ${jsonfile} | awk -F':' '{print $2}' >> scores.csv
                grep "regression" ${jsonfile} | awk -F':' '{print $2}' >> scores.csv
                python sample.py /home/sray/DARPA_D3M/${datasets[i]}/TRAIN/problem_TRAIN/problemDoc.json /home/sray/DARPA_D3M/${datasets[i]}/TRAIN/dataset_TRAIN/datasetDoc.json ${jsonfile} /home/sray/DARPA_D3M/${datasets[i]}/TEST/dataset_TEST/datasetDoc.json

	        python evaluate_score.py /home/sray/DARPA_D3M/${datasets[i]}/SCORE/targets.csv results.csv ${targets[i]} ${metrics[i]} >> scores.csv
	done
done
