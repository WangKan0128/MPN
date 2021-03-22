#! /bin/bash
# this file finish the feature extraction process in test phase
epoch=31
while [ $epoch -le 65 ]
do
    python test_baseline.py --gpu_ids 4 --test_dir /path_to_dataset/pytorch_aug  --location /path_to_snapshot/ --which_epoch $epoch
    epoch=$(($epoch+1))
done




