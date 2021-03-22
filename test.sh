#! /bin/bash
# this file finish the feature extraction process in test phase
epoch=31
while [ $epoch -le 65 ]
do
    python test.py --gpu_ids 4 --test_dir /data0/kan/data/reid/Duke/DukeMTMC-reID/pytorch_aug  --location /data1/kan/dsak/mdd/1501/3.14/ --which_epoch $epoch
    epoch=$(($epoch+1))
done




