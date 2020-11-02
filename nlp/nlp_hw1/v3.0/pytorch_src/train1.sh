#! /bin/bash

GPU=$1
TIMESTAMP="`date +%Y_%m_%d_%H_%M_%S`"

CUDA_VISIBLE_DEVICES=${GPU} \
	python main.py \
	--train-data=/home/charlie/nlp/data/new_key_roc_key.train \
	--valid-data=/home/charlie/nlp/data/new_key_roc_key.test \
	--test-data=/home/charlie/nlp/data/new_key_roc_key.test \
	--vocab-file=small_vocab.pkl \
	--emsize=500 \
	--nhid=1000 \
	--nlayers=3 \
	--epochs=128 \
	--dropout=0.1 --dropouth=0.1 --wdrop=0.1 \
	--dropouti=0.4 --dropoute=0.4 \
	--save_dir=./title2key_models 2>&1 | tee logs/train_stage1_log_${TIMESTAMP}
