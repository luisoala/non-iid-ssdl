#!/bin/bash
#
#EXPERIMENT CONFIGS
#DATA
#DLIDS are the "download ids" of the gdrive for each data set
export CUDA_VISIBLE_DEVICES=1
declare -A DATA_DLIDS=( ["MNIST"]="10pULG3xRIkl5tDo6VJHUK6NHIguMT7Jq" ["FASHIONMNIST"]="11AJ-OEgtj7XDeLHPVEyWuvHzFIIadj9_" ["CIFAR10"]="1O6uarg54CwtZ3h_B6YzD1KW9nbQ-E7Xl" ["TINYIMAGENET"]="10i1FV1SgXxMWfgTEpAXr0qe7q7s6Ko6e" ["SVHN"]="1wgTQJOtGxWPLNKPMuDc7vzYejYnWJsVr" ["SVHN-different"]="10HbYSMt3CbHeieUBqO675eNpAef-UY3_" ["GaussianNoise"]="1GXljou_EJGcdfVsfVJiVMVo7-RKNu106" ["SALTANDPEPPER"]="1iBwKyR7M4_ca2Ti7xW5FJ-BNdqEDG-vK" ["FASHIONPRODUCT"]="1zN1BF1u1SJl81JpH6hexYgrVvivaGxa5")
#FPATHS are the file paths for the data sets in the local working directory when doing experiments
declare -A DATA_FPATHS=(["MNIST"]="data/MNIST/" ["FASHIONMNIST"]="data/FASHIONMNIST/" ["CIFAR10"]="data/CIFAR10/" ["TINYIMAGENET"]="data/TINYIMAGENET/" ["SVHN-different"]="data/SVHN-different/" ["GaussianNoise"]="data/GaussianNoise/" ["SALTANDPEPPER"]="data/SALTANDPEPPER/" ["SVHN"]="data/SVHN/" ["FASHIONPRODUCT"]="data/FASHIONPRODUCT/")

BASE_DATA="CIFAR10" #the base data that is used for the iod data. around it which the experiment is centered
DIFFERENT_DATA="MNIST" #array of datasets used for the ood setting "different" -> contrasting data sets

OOD_PERC_PP_LIST=(0 50 100) #the ood percentage in percentage points
NUM_UNLABELED=3000 #should be 3000
MIN_CLASS_ID=0 #the lowest class id for the classes in the data set
MAX_CLASS_ID=9 #the highest class id for the classes of the data set
NUM_CLASSES_IN_DIST=5 #the number of classes to select for the in dist class
NUM_LABELED_LIST=(60 100 150)
OOD_TYPE="different" #can be "half-half" (ood samples come from same dataset but are a subset of the classes) or "different" (ood samples come from different dataset)
#NOTE: seed for shuf of iod class permuations is given by batch id on run level
#
#RUN CONFIGS (a run is an iteration of teh experiment with one of the random data batches)
BATCHES=(0 1 2 3 4 5 6 7 8 9) #the batch id used for the different runs of the experiment
#
#MIX MATCH ALGO CONFIGS
MODEL="wide_resnet"
DATASET="CIFAR10-BASELINE"
RESULTS_FILE="stats_OOD_4_SSDL.csv"
WORKERS="1"
EPOCHS="50" #should be 50
BATCH_SIZE="16" #should be 16
LR="0.0002"
WEIGHT_DECAY="0.0001"
K_TRANSFORMS="2"
T_SHARPENING="0.25"
ALPHA_MIX="0.75"
MODE="ssdl"
BALANCED="5" #int -1 no bal, 5 bal
GAMMA_US="25" #the gamma for the unsupervised loss
IMG_SIZE="32"
NORM_STATS="MNIST" #is not used
#
LOG_FOLDER="logs"
SAVE_WEIGHTS="FALSE"
WEIGHTS_PATH=""
RAMPUP_COEFFICIENT="3000"
#
N=10 #number of parallel processes
#DOWNLOAD DATA
#steps
#create data dir
mkdir data
#cd insto data dir
cd data

#do it for base data
##downlaod file
###get file id
FILEID="${DATA_DLIDS["${BASE_DATA}"]}"
###download
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${BASE_DATA}.zip && rm -rf /tmp/cookies.txt
##unzip
unzip ${BASE_DATA}.zip
##remove zip
rm ${BASE_DATA}.zip

#do it for different data
###get file id
FILEID="${DATA_DLIDS["${DIFFERENT_DATA}"]}"
###download
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${FILEID}" -O ${DIFFERENT_DATA}.zip && rm -rf /tmp/cookies.txt
##unzip
unzip ${DIFFERENT_DATA}.zip
##remove zip
rm ${DIFFERENT_DATA}.zip
#when done cd .. back to work dir
cd ..

#EXPERIMENTS
PATH_BASE="${DATA_FPATHS["${BASE_DATA}"]}" #path to the dataset that will be used for the experiments and construction of the train and val sets. assumes at the end of path exists another directory /all/ which contains subdirectories with id name of class and then each containing the samples of this class
PATH_DIFFERENT="${DATA_FPATHS["${DIFFERENT_DATA}"]}"

train_batch(){
	local batch_id=$1
	local OOD_PERC=$2
	local OOD_PERC_PP=$3
	local NUM_UNLABELED=$4
	local IDS=$5
	local LIST_IN_DIST_CLASSES=$(echo $IDS | tr ' ' ,) #the subset of the classes in the dataset that are used as in dist data
	
	local PATH_LABELED="$PATH_BASE/batches_labeled_in_dist/batch_$batch_id"
	local PATH_UNLABELED="$PATH_BASE/unlabeled/batch_${batch_id}_num_unlabeled_${NUM_UNLABELED}_ood_perc_${OOD_PERC_PP}"

	python MixMatch_OOD_main.py --dataset $DATASET --path_labeled $PATH_LABELED --path_unlabeled $PATH_UNLABELED  --results_file_name $RESULTS_FILE --workers $WORKERS --epochs $EPOCHS --batch_size $BATCH_SIZE --lr  $LR --weight_decay $WEIGHT_DECAY --K_transforms $K_TRANSFORMS --T_sharpening $T_SHARPENING --alpha_mix $ALPHA_MIX --mode $MODE --balanced $BALANCED --lambda_unsupervised $GAMMA_US --number_labeled $NUM_LABELED --model $MODEL --num_classes $NUM_CLASSES_IN_DIST --size_image $IMG_SIZE --log_folder $LOG_FOLDER --norm_stats $NORM_STATS --save_weights $SAVE_WEIGHTS --weights_path_name "$WEIGHTS_PATH" --rampup_coefficient $RAMPUP_COEFFICIENT
}



if [ $OOD_TYPE = "half-half" ]
then
	for NUM_LABELED in ${NUM_LABELED_LIST[@]} #axis 1 loop, i.e. portion of labeled data
	do
		for OOD_PERC_PP in ${OOD_PERC_PP_LIST[@]} #axis 2 loop, i.e. ood portion in unlabeled data
		do
			OOD_PERC=$(bc <<< "${OOD_PERC_PP} * 0.01") #go from percentage points to decimals for OOD percentage, i.e. e.g. 66 -> 0.66, for the second python command
			#first iteration over batches creates the data
			IDS_LIST=() #keep track of the class ids per batch for running mixmatch later
			for batch_id in ${BATCHES[@]}
			do
				IDS=$(shuf -i $MIN_CLASS_ID-$MAX_CLASS_ID -n $NUM_CLASSES_IN_DIST) # --random-source=<(echo $batch_id)) #randomly select class ids for in-dist data, random seed is provided by batch_id for reproducability
				IDS_LIST+=("$IDS") #add class ids for this batch to the class ids list that mixmatch can use later for training
				LIST_IN_DIST_CLASSES=$(echo $IDS | tr ' ' ,) #the subset of the classes in the dataset that are used as in dist data
				
				python utilities/dataset_partitioner.py --mode train_partitioner --path_base "$PATH_BASE/" --batch_id_num $batch_id --list_in_dist_classes $LIST_IN_DIST_CLASSES
				
				python utilities/dataset_partitioner.py --mode unlabeled_partitioner --path_ood "$PATH_BASE/batches_unlabeled_out_dist/batch_$batch_id" --path_iod "$PATH_BASE/batches_labeled_in_dist/batch_$batch_id/train" --path_dest "$PATH_BASE/unlabeled" --ood_perc "$OOD_PERC" --num_unlabeled "$NUM_UNLABELED" --batch_id_num "$batch_id"
			
			done
			
			#the second iteration over batches runs mixmatch in parallel
			for ((i=0;i<${#BATCHES[@]};++i))
			do
				((j=j%N)); sleep 3; ((j++==0)) && wait #short sleep to prevent conflicts in creating the documentation files
				train_batch "${BATCHES[i]}" "$OOD_PERC" "$OOD_PERC_PP" "$NUM_UNLABELED" "${IDS_LIST[i]}"&
			done
			wait
			#)
			#do clean up prior to the next experiment
			rm -r $PATH_BASE/unlabeled/
			rm -r $PATH_BASE/batches_labeled_in_dist/
			rm -r $PATH_BASE/batches_unlabeled_out_dist/
		done
	done
elif [[ $OOD_TYPE -eq "different" ]]
then
	#add loop over the different datasets -> no loop as I split experiments into invidual data sets
	
	for NUM_LABELED in ${NUM_LABELED_LIST[@]} #axis 1 loop, i.e. portion of labeled data
	do
		for OOD_PERC_PP in ${OOD_PERC_PP_LIST[@]} #axis 2 loop, i.e. ood portion in unlabeled data
		do
			OOD_PERC=$(bc <<< "${OOD_PERC_PP} * 0.01") #go from percentage points to decimals for OOD percentage, i.e. e.g. 66 -> 0.66, for the second python command
			IDS_LIST=() #keep track of the class ids per batch for running mixmatch later
			for batch_id in ${BATCHES[@]}
			do
				IDS=$(shuf -i $MIN_CLASS_ID-$MAX_CLASS_ID -n $NUM_CLASSES_IN_DIST) # --random-source=<(echo $batch_id)) #randomly select class ids for in-dist data, random seed is provided by batch_id for reproducability
				IDS_LIST+=("$IDS") #add class ids for this batch to the class ids list that mixmatch can use later for training
				LIST_IN_DIST_CLASSES=$(echo $IDS | tr ' ' ,) #the subset of the classes in the dataset that are used as in dist data#the subset of the classes in the dataset that are used as in dist data
				python utilities/dataset_partitioner.py --mode train_partitioner --path_base "$PATH_BASE/" --batch_id_num $batch_id --list_in_dist_classes $LIST_IN_DIST_CLASSES
				
				python utilities/dataset_partitioner.py --mode unlabeled_partitioner --path_ood "$PATH_DIFFERENT" --path_iod "$PATH_BASE/batches_labeled_in_dist/batch_$batch_id/train" --path_dest "$PATH_BASE/unlabeled" --ood_perc "$OOD_PERC" --num_unlabeled "$NUM_UNLABELED" --batch_id_num "$batch_id"
				#path_ood here has one subfolder with all the unlabelled images
				
			done
			
			#the second iteration over batches runs mixmatch in parallel
			PATH_LABELED="$PATH_BASE/batches_labeled_in_dist/batch_$batch_id"
			PATH_UNLABELED="$PATH_BASE/unlabeled/batch_0_num_unlabeled_${NUM_UNLABELED}_ood_perc_${OOD_PERC_PP}"
			python MixMatch_OOD_main.py --dataset $DATASET --path_labeled $PATH_LABELED --path_unlabeled $PATH_UNLABELED  --results_file_name $RESULTS_FILE --workers $WORKERS --epochs $EPOCHS --batch_size $BATCH_SIZE --lr  $LR --weight_decay $WEIGHT_DECAY --K_transforms $K_TRANSFORMS --T_sharpening $T_SHARPENING --alpha_mix $ALPHA_MIX --mode $MODE --balanced $BALANCED --lambda_unsupervised $GAMMA_US --number_labeled $NUM_LABELED --model $MODEL --num_classes $NUM_CLASSES_IN_DIST --size_image $IMG_SIZE --log_folder $LOG_FOLDER --norm_stats $NORM_STATS --save_weights $SAVE_WEIGHTS --weights_path_name "$WEIGHTS_PATH" --rampup_coefficient $RAMPUP_COEFFICIENT --exp_creator "Yes"
			for ((i=0;i<${#BATCHES[@]};++i))
			do
				((j=j%N)); sleep 3; ((j++==0)) && wait #short sleep to prevent conflicts in creating the documentation files
				train_batch "${BATCHES[i]}" "$OOD_PERC" "$OOD_PERC_PP" "$NUM_UNLABELED" "$LIST_IN_DIST_CLASSES"&
			done
			wait
			
			#do clean up prior to the next experiment
			rm -r $PATH_BASE/unlabeled/
			rm -r $PATH_BASE/batches_labeled_in_dist/
			rm -r $PATH_BASE/batches_unlabeled_out_dist/
		done
	done
	
	
	
else
	echo "No valid OOD_TYPE was specified. Choose 'same' or 'different'"
fi
#final cleanup -> delete all data
rm -r data/
#dynamic resizing of images for different setting -> when they do not exactly match -> defined in mixmatch
