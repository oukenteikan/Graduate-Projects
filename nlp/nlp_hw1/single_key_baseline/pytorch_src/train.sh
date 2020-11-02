#! /bin/bash

GPU=$1
TIMESTAMP="`date +%Y_%m_%d_%H_%M_%S`"

CUDA_VISIBLE_DEVICES=${GPU} \
	python main.py \
	| tee logs/log_${TIMESTAMP}
