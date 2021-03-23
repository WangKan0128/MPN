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
  year="2020",
  DOI="10.1109/TPAMI.2020.3024900".
}

```

### Dependencies 

* Python > 3.6.3
* PyTorch > 0.4.1

### Strcuture of the database

$data_dir

└───$1501

    |───train_all

        └───0002

            └───0002_c1s1_000451_03.jpg

            └───0002_c1s1_000551_01.jpg

            └───...

        └───0007

            └───0007_c1s6_028546_01.jpg

            └───0007_c1s6_028546_04.jpg

            └───...

        ───Person ID

            └───images

            └───...

    |───gallery

        └───Person ID

            └───images

    |───query

        └───Person ID

            └───images





### Training phase

The MPN is trained in a end-to-end manner with the following settings:



python train_mpn_no_ca.py  --gpu_ids 7 --train_all --n_classes 6 --n_images 8  --erasing_p 0.5 --data_dir /path_to_dataset/



1. finetuning from the IDE snapshot (./data/1501.pth);

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

    Rank-1: 96.0%  mAP: 89.2%.



Besides, when equipped with CA module, the performance of MPN is:

    Rank-1: 96.1%  mAP: 89.4%.





