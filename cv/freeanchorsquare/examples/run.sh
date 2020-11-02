#! /bin/bash

export NGPUS=4
python -m torch.distributed.launch \
	--nproc_per_node=$NGPUS \
	--master_port=9901 \
	../tools/train_net.py \
	--config-file ../configs/free_anchor_R-50-FPN_1x.yaml
