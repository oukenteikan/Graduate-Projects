#! /bin/bash

export NGPUS=4
python -m torch.distributed.launch \
	--nproc_per_node=$NGPUS \
	--master_port=9901 \
	../tools/test_net.py \
	--config-file "../configs/free_anchor_R-50-FPN_1x.yaml" \
	MODEL.WEIGHT "./outputs/free_anchor_R-50-FPN_1x/model_0050000.pth" \
	DATASETS.TEST "('coco_person_val',)"
