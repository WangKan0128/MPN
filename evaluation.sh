#!/usr/bin/env bash
epoch=31
while [ $epoch -le 37 ]
do
    python evaluate.py --location /data1/kan/dsak/mdd/1501/3.14/ --epoch $epoch
    epoch=$(($epoch+1))
done



