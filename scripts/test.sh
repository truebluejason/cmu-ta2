#!/usr/bin/env bash


# --------------------------------------
# SCRIPT CONFIGURATION
# Configure test script variables below.
# --------------------------------------

# DATASETS

datasets=(66_chlorineConcentration_MIN_METADATA)
targets=(Hall_of_Fame)
metrics=(F1Macro)

# Time series classification
#datasets=(66_chlorineConcentration_MIN_METADATA LL1_50words_MIN_METADATA LL1_Adiac_MIN_METADATA LL1_ArrowHead_MIN_METADATA LL1_CinC_ECG_torso_MIN_METADATA LL1_Cricket_Y_MIN_METADATA LL1_ECG200_MIN_METADATA LL1_ElectricDevices_MIN_METADATA LL1_FISH_MIN_METADATA LL1_FaceFour_MIN_METADATA LL1_FordA_MIN_METADATA LL1_HandOutlines_MIN_METADATA LL1_Haptics_MIN_METADATA LL1_ItalyPowerDemand_MIN_METADATA LL1_Meat_MIN_METADATA LL1_OSULeaf_MIN_METADATA LL1_crime_chicago_MIN_METADATA uu1_datasmash_MIN_METADATA)

## Image classification
#datasets=(124_174_cifar10_MIN_METADATA 124_188_usps_MIN_METADATA 124_214_coil20_MIN_METADATA 124_95_uc_merced_land_use_MIN_METADATA uu_101_object_categories_MIN_METADATA LL1_336_MS_Geolife_transport_mode_prediction_MIN_METADATA LL1_336_MS_Geolife_transport_mode_prediction_separate_lat_lon_MIN_METADATA)

#datasets=(LL1_FordA_MIN_METADATA)
#
#datasets=(LL1_Haptics_MIN_METADATA)
#
#datasets=(185_baseball_MIN_METADATA 1567_poker_hand_MIN_METADATA 313_spectrometer_MIN_METADATA 56_sunspots_monthly_MIN_METADATA 6_70_com_amazon_MIN_METADATA LL0_acled_reduced_MIN_METADATA LL1_736_population_spawn_simpler_MIN_METADATA LL1_bn_fly_drosophila_medulla_net_MIN_METADATA LL1_ECG200_MIN_METADATA LL1_h1b_visa_apps_7480 LL1_OSULeaf_MIN_METADATA LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA LL1_TXT_CLS_3746_newsgroup_MIN_METADATA LL1_VTXC_1343_cora_MIN_METADATA SEMI_1053_jm1_MIN_METADATA uu10_posts_3_MIN_METADATA uu4_SPECT_MIN_METADATA uu9_posts_2_MIN_METADATA)
#
#datasets=(56_sunspots_monthly_MIN_METADATA)
#datasets=(LL1_736_stock_market_MIN_METADATA)
#
#datasets=(124_174_cifar10_MIN_METADATA 124_95_uc_merced_land_use_MIN_METADATA 1491_one_hundred_plants_margin_MIN_METADATA 30_personae_MIN_METADATA 31_urbansound_MIN_METADATA 32_wikiqa_MIN_METADATA 49_facebook_MIN_METADATA 56_sunspots_MIN_METADATA 59_LP_karate_MIN_METADATA 59_umls_MIN_METADATA 60_jester_MIN_METADATA 66_chlorineConcentration_MIN_METADATA 6_70_com_amazon_MIN_METADATA LL0_acled_reduced_MIN_METADATA)
#
# All forecasting datasets
#datasets=(56_sunspots_MIN_METADATA 56_sunspots_monthly_MIN_METADATA LL1_736_population_spawn_MIN_METADATA LL1_736_population_spawn_simpler_MIN_METADATA LL1_736_stock_market_MIN_METADATA LL1_CONFLICT_3457_atrocity_forecasting LL1_PHEM_Monthly_Malnutrition_MIN_METADATA LL1_PHEM_weeklyData_malnutrition_MIN_METADATA LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA )

#datasets=(185_baseball_MIN_METADATA 66_chlorineConcentration_MIN_METADATA)

#datasets=(LL1_736_stock_market_MIN_METADATA)

# Clustering
#datasets=(1491_one_hundred_plants_margin_clust_MIN_METADATA)

# JIDO datasets
#datasets=(JIDO_SOHR_Articles_1061 JIDO_SOHR_Tab_Articles_8569 LL0_jido_reduced_MIN_METADATA)
datasets=(LL0_jido_reduced_MIN_METADATA)

# One dataset for each category:
# binary classification geospatial tabular,
# binary classification lupi tabular,
# binary classification relational,
# binary classification tabular,
# binary classification text,
# binary classification text relational,
# binary classification time series,
# binary semi-supervised classification tabular,
# collaborative filtering tabular
# community detection graph,
# graph matching,
# link prediction graph,
# link prediction graph time series,
# multiclass classification audio,
# multiclass classification geospatial tabular,
# multiclass classification image,
# multiclass classification image remote sensing,
# multiclass classification relational,
# multiclass classification tabular,
# multiclass classification text,
# multiclass classification time series,
# multiclass classification time series geospatial relational,
# multiclass classification time series geospatial tabular,
# multiclass classification video,
# multiclass semi-supervised classification tabular,
# multivariate regression relational,
# multivariate regression tabular,
# object detection image,
# regression tabular,
# regression image,
# regression relational,
# time series classification binary grouped tabular,
# time series forecasting grouped,
# time series forecasting grouped tabular,
# time series forecasting tabular,
# vertex classification multiClass graph
#datasets=(124_174_cifar10_MIN_METADATA 124_95_uc_merced_land_use_MIN_METADATA 1491_one_hundred_plants_margin_MIN_METADATA 196_autoMpg_MIN_METADATA 22_handgeometry_MIN_METADATA 30_personae_MIN_METADATA 31_urbansound_MIN_METADATA 32_wikiqa_MIN_METADATA 49_facebook_MIN_METADATA 56_sunspots_MIN_METADATA 59_LP_karate_MIN_METADATA 60_jester_MIN_METADATA 66_chlorineConcentration_MIN_METADATA 6_70_com_amazon_MIN_METADATA LL0_acled_reduced_MIN_METADATA LL1_3476_HMDB_actio_recognition_MIN_METADATA LL1_726_TIDY_GPS_carpool_bus_service_rating_prediction_MIN_METADATA LL1_736_population_spawn_MIN_METADATA LL1_ACLED_TOR_online_behavior_MIN_METADATA LL1_CONFLICT_3457_atrocity_forecasting LL1_ECG200_MIN_METADATA LL1_EDGELIST_net_nomination_seed_MIN_METADATA LL1_GS_process_classification_tabular_MIN_METADATA LL1_GT_actor_group_association_prediction_MIN_METADATA LL1_TXT_CLS_3746_newsgroup_MIN_METADATA LL1_penn_fudan_pedestrian_MIN_METADATA LL1_retail_sales_total_MIN_METADATA SEMI_1040_sylva_prior_MIN_METADATA SEMI_1044_eye_movements_MIN_METADATA loan_status_MIN_METADATA political_instability_MIN_METADATA uu10_posts_3_MIN_METADATA uu2_gp_hyperparameter_estimation_MIN_METADATA uu4_SPECT_MIN_METADATA uu8_posts_1_MIN_METADATA LL1_336_MS_Geolife_transport_mode_prediction_MIN_METADATA)

# SEMI datasets
datasets=(SEMI_1040_sylva_prior_MIN_METADATA SEMI_1044_eye_movements_MIN_METADATA SEMI_1053_jm1_MIN_METADATA SEMI_1217_click_prediction_small_MIN_METADATA SEMI_1459_artificial_characters_MIN_METADATA SEMI_155_pokerhand_MIN_METADATA)

# Jobs Directory (where output from jobs will be stored)
#jobsdir="$( cd "${scriptsdir}/../jobs" > /dev/null 2>&1 && pwd )"
if [ "$USER" == "root" ]; then
    jobsdir=/jobs
elif [ "$USER" == "gwelter" ]; then
    jobsdir=/zfsauton/data/public/gwelter/ta2jobs
elif [ "$USER" == "sray" ]; then
    jobsdir=???
fi

# Export relevant project directories
scriptsdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
codedir="$( cd "${scriptsdir}/../src" > /dev/null 2>&1 && pwd )"
prjrootdir="$( cd "${scriptsdir}/.." > /dev/null 2>&1 && pwd )"

# D3M ENVIRONMENTAL VARIABLES
export D3MTIMEOUT=3600
export D3MCPU=12
export D3MDATADIR=/zfsauton/data/d3m/datasets/seed_datasets_current
export D3MSTATICDIR="${prjrootdir}/static"

# DOCKER OVERRIDES (override variables if running inside a Docker container)
if [ -f /.dockerenv ]; then
    echo "Overriding D3MDATADIR and jobsdir for Docker."
    D3MDATADIR=/data/seed_datasets_current
    jobsdir=/jobs
fi


# --------------------------------------
# FUNCTION HELPERS
# Functions to perform some of the work.
# --------------------------------------

# Function to create a new job folder
function create_new_job_folder {

    # Establish a job name and, therefore, job directory
    jobname=$(date '+%Y%m%d_%H%M%S')
    jobdir="${jobsdir}/${jobname}"

    # Create the job directory & scaffolding
    mkdir -p "${jobdir}" > /dev/null 2>&1

    # Create a symlink to this job as the latest job
    rm -rf "${jobsdir}/latest" && ln --symbolic --force --relative "${jobdir}" "${jobsdir}/latest"

    # Output the new job directory absolute path
    echo $(cd "${jobdir}" >/dev/null 2>&1 && pwd)

}

# Function for formatting number of seconds as human-readable time.
function hrtime {
    local T=$1
    local D=$((T/60/60/24))
    local H=$((T/60/60%24))
    local M=$((T/60%60))
    local S=$((T%60))
    (( $D > 0 )) && printf '%d days ' $D
    (( $H > 0 )) && printf '%d hours ' $H
    (( $M > 0 )) && printf '%d minutes ' $M
    (( $D > 0 || $H > 0 || $M > 0 )) && printf 'and '
    printf '%d seconds\n' $S
}

# Function for running the tests (this is a separate function so that we can tee
# the output to console and file).
function do_the_work {

    printf "\nBeginning test script.\n\n"

    # Print out config variables
    printf -- "Configuration variables\n-----------------------\n\n"
    printf "D3MOUTPUTDIR=${D3MOUTPUTDIR}\n"
    printf "D3MSTATICDIR=${D3MSTATICDIR}\n"
    printf "D3MDATADIR=${D3MDATADIR}\n"
    printf "D3MTIMEOUT=${D3MTIMEOUT}\n"
    printf "D3MCPU=${D3MCPU}\n"
    printf "prjrootdir=${prjrootdir}\n"
    printf "codedir=${codedir}\n"
    printf "scriptsdir=${scriptsdir}\n"
    printf "jobdir=${jobdir}\n"
    printf "datasets=( "
    printf '%s ' "${datasets[@]}"
    printf ")\n"

    # Write CSV column headers to file
    printf "Dataset,JSON File,Main Primitive,Main Primitive Path,Hyperparameters,Pipeline Rank,Test Score\n" >> "${scores_file}"
    printf "Dataset,TA2 Time,D3M Test Time\n" >> "${times_file}"

    for (( i=0; i<${#datasets[@]}; i++ ))
    do

        printf -- "\n\n\n--------------------------------------------------------------\n"
        printf "Running D3M CMU-TA2 on ${datasets[i]}\n"
        printf -- "--------------------------------------------------------------\n\n"

        printf ${datasets[i]} >> "${times_file}"

        export D3MINPUTDIR="$D3MDATADIR/${datasets[i]}"
        export D3MPROBLEMPATH="$D3MDATADIR/${datasets[i]}/TRAIN/problem_TRAIN/problemDoc.json"

        echo "D3MINPUTDIR=${D3MINPUTDIR}"
        echo "D3MPROBLEMPATH=${D3MPROBLEMPATH}"

        # Parse target & metric from problem doc
        target=$(cat "${D3MPROBLEMPATH}" | jq --raw-output '.inputs.data[0].targets[0].colName')
        metric=$(cat "${D3MPROBLEMPATH}" | jq --raw-output '.inputs.performanceMetrics[0].metric')

        echo "target=${target}"
        echo "metric=${metric}"

        # We need a dataset-specific output directory if more than one dataset
        if [ "${#datasets[@]}" -gt 1 ]; then
        	export D3MOUTPUTDIR="${jobdir}/${datasets[i]}/output"
        	mkdir -p "${D3MOUTPUTDIR}" > /dev/null 2>&1
        	echo "D3MOUTPUTDIR (dataset) = ${D3MOUTPUTDIR}"
        fi

        # TA2 system builds pipelines, runs CV on them, ranks them, and outputs the
        # jsons. For this purpose, uses training data.
        start=`date +%s`
        "${codedir}/main.py" search
        end=`date +%s`

        # Output TA2 time to times file
        runtime=$((end-start))
        printf ",$(hrtime ${runtime})" >> "${times_file}"
        echo ">>> TA2 completed. Time: ${runtime}"

        FILE=$D3MDATADIR/${datasets[i]}/SCORE/targets.csv
        if [ -f $FILE ]; then
            echo "${FILE} exists. Using it."
        else
            echo "${FILE} does not exist."
            FILE=$D3MDATADIR/${datasets[i]}/SCORE/dataset_SCORE/tables/learningData.csv
            echo "Using ${FILE}."
        fi

        # Evaluate pipelines on test data and compute score
        start=`date +%s`
        for search_dir in "${D3MOUTPUTDIR}"/*
        do
            echo ${search_dir}
            for jsonfile in ${search_dir}/pipelines_ranked/*
            do

                echo ${jsonfile}
                python3 -m d3m runtime -v $D3MSTATICDIR fit-produce -p ${jsonfile} -r $D3MDATADIR/${datasets[i]}/TRAIN/problem_TRAIN/problemDoc.json -i $D3MDATADIR/${datasets[i]}/TRAIN/dataset_TRAIN/datasetDoc.json -t $D3MDATADIR/${datasets[i]}/TEST/dataset_TEST/datasetDoc.json -o "${results_file}"
                score=$(python3 "${prjrootdir}/evaluate_score.py" "$FILE" "${results_file}" "${target}" "${metric}" 1)

                # Write row to the scores_new file
                echo "python3 \"${prjrootdir}/record_score.py\" \"${jsonfile}\" \"${scores_file}\" \"${datasets[i]}\" \"${score}\""
                python3 "${prjrootdir}/record_score.py" "${jsonfile}" "${scores_file}" "${datasets[i]}" "${score}"
                # python3 "${prjrootdir}/record_score.py" ${jsonfile} "${scores_file}" ${datasets[i]} 0

            done

        done
        end=`date +%s`

        # Output total test time for this dataset to times file
        runtime=$((end-start))
        printf ",$(hrtime ${runtime})\n" >> "${times_file}"

    done

    printf -- "\n\nOutput of results.csv:\n\n"

    cat "${scores_file}"

    printf -- "\n\nTest script concluded.\n\n"

}


# --------------------------------------
# MAIN
# Main functionality.
# --------------------------------------

# Make sure our working directory is the scripts directory
cd "$scriptsdir"

# Create a new job directory
jobdir=$(create_new_job_folder)

# Location of output files
export D3MOUTPUTDIR="${jobdir}/output"
scores_file="${jobdir}/scores.csv"
times_file="${jobdir}/times.csv"
results_file="${jobdir}/results.csv"
running_indicator_file="${jobdir}/RUNNING"
terminal_output_file="${jobdir}/output.log"
complete_indicator_file="${jobdir}/COMPLETE"

# If we have 3 command arguments, this represents the dataset, target, and
# metric respectively, so override the config values with these values.
if [[ -n $1 && -n $2 && -n $3 ]]
then
    datasets=($1)
    targets=($2)
    metrics=($3)
    echo "Dataset, target, metric overridden to: ${datasets} ${targets} ${metrics}"
fi

# Place a file "RUNNING" to indicate the script is currently running.
echo "The script is currently running." >> ${running_indicator_file}

do_the_work 2>&1 | tee ${terminal_output_file}

# Remove running file, and add complete file to indicate the script finished.
rm -rf ${running_indicator_file}
echo "The script has completed." >> ${complete_indicator_file}
