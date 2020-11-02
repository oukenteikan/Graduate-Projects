#! /bin/bash

GPU=$1

CUDA_VISIBLE_DEVICES=${GPU} \
	python generate.py \
