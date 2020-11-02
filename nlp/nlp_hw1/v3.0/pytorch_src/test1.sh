#! /bin/bash

EPOCH=$1
GPU=$2

TIMESTAMP="`date +%Y_%m_%d_%H_%M_%S`"

CUDA_VISIBLE_DEVICES=${GPU} \
	python generate.py \
	--vocab=./small_vocab.pkl \
	--conditional-data=/home/charlie/nlp/data/roc_title.test \
	--checkpoint=./title2key_models/${EPOCH}_model.pt \
	--outf=./results/roc_key_${EPOCH}.test \
	--sents=10000 \
	--dedup \
	--temperature=0.6 2>&1 | tee logs/test_stage1_log_${TIMESTAMP}
