#! /bin/bash

EPOCH1=$1
GPU=$2

TIMESTAMP="`date +%Y_%m_%d_%H_%M_%S`"

CUDA_VISIBLE_DEVICES=${GPU} \
	python generate.py \
	--vocab=./all_vocab.pkl \
	--conditional-data=./results/roc_key_${EPOCH1}.test \
	--checkpoint=./pretrain/title2story_model.pt \
	--outf=./results/roc_${EPOCH1}.test \
	--sents=10000 \
	--temperature=0.4 2>&1 | tee logs/test_stage2_log_${TIMESTAMP}
