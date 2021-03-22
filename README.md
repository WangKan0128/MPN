## MPN

### Introduction
This repository hosts a simplified version of the code used for the **https://ieeexplore.ieee.org/document/9200784** **Multi-task Learning with Coarse Priors for Robust Part-aware Person Re-identification**. 

With MPN, we obtain state-of-the-art results on multiple person re-identification databases.

### Citation
If you find our work useful in your research, please consider citing:

```bibtex
@article{ding2020multi,
  title="Multi-task Learning with Coarse Priors for Robust Part-aware Person Re-identification",
  author="Changxing Ding and Kan Wang and Pengfei Wang and Dacheng Tao",
  journal="IEEE Transactions on Pattern Analysis and Machine Intelligence",
  year="2020"
}
```

### Dependencies 
* Python > 3.6.3
* PyTorch > 0.4.1


### Training phase
The MPN is trained in a end-to-end manner with the following settings:

python train_mpn_no_ca.py  --gpu_ids 7 --train_all --n_classes 6 --n_images 8  --erasing_p 0.5 --data_dir /path_to_dataset/

1. finetuning from a IDE snapshot (at ./data/1501.pth);
2. intial learning rate: 0.01, x 0.1 for every 20 epoches, 70 epoch in total;
3. batchsize: 48, 6 randomly sampled identities, 8 images for each identity;
4. triplet margin: 0.4;
5. erasing ratio for random erasing: 0.5;


### Test phase
feature extraction:
    python test_baseline.py --gpu_ids 4 --test_dir /path_to_dataset/  --location /path_to_snapshot/ --which_epoch $epoch
performance evaluation:
    python evaluate.py --location /path_to_feat/ --epoch $epoch


### Performance
Performance obtained by the simplified version of MPN on Market-1501 dataset is:
    top1:0.959620 top5:0.985748 top10:0.992280 mAP:0.892474.

Besides, when equipped with CA module, the performance of MPN is:
    top1:0.960808 top5:0.985451 top10:0.990499 mAP:0.894083.


