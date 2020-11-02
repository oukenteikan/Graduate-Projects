#! /bin/bash

EPOCH=$1
RESULT=$2
GPU=$3

CUDA_VISIBLE_DEVICES=${GPU} \
	python generate.py \
	--checkpoint models/${EPOCH}_model.pt \
	--outf results/result${RESULT}.txt
