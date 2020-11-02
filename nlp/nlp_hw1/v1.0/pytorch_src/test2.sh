#! /bin/bash

EPOCH1=$1
EPOCH2=$2
GPU=$3

TIMESTAMP="`date +%Y_%m_%d_%H_%M_%S`"

CUDA_VISIBLE_DEVICES=${GPU} \
	python generate.py \
	--vocab=./all_vocab.pkl \
	--conditional-data=./results/roc_key_${EPOCH1}.test \
	--checkpoint=./title2story_models/${EPOCH2}_model.pt \
	--outf=./results/roc_${EPOCH1}_${EPOCH2}.test \
	--sents=10000 \
	--temperature=0.4 2>&1 | tee logs/test_stage2_log_${TIMESTAMP}
