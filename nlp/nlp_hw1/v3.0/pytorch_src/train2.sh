#! /bin/bash

GPU=$1
TIMESTAMP="`date +%Y_%m_%d_%H_%M_%S`"

CUDA_VISIBLE_DEVICES=${GPU} \
	python main.py \
	--train-data=/home/charlie/nlp/data/new_key_roc.train \
	--valid-data=/home/charlie/nlp/data/new_key_roc.test \
	--test-data=/home/charlie/nlp/data/new_key_roc.test \
	--vocab-file=all_vocab.pkl \
	--emsize=500 \
	--nhid=1000 \
	--nlayers=3 \
	--epochs=256 \
	--dropout=0.1 --dropouth=0.1 --wdrop=0.1 \
	--dropouti=0.2 --dropoute=0.2 \
	--save_dir=./title2story_models/ 2>&1 | tee logs/train_stage2_log_${TIMESTAMP}
