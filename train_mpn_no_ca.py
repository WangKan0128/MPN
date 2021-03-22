# -*- coding: utf-8 -*-
from __future__ import print_function, division

import argparse
import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from torch.nn import init

from utils.random_erasing import RandomErasing
from utils.batch_sample_old_adjust import BalancedBatchSampler
from utils.triplet_sampling_hardestP import HardestNegativeTripletSelector as Hard
from utils.losses import OnlineTripletLoss

# the teacher
from models.mpn_no_ca import TS

# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='output model name')

parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--gpu_ids',default='4', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--n_classes', default=6, type=int, help='how many person in a batch')
parser.add_argument('--n_images', default=8, type=int, help='how many image for each person in a batch')
parser.add_argument('--margin', default=0.40, type=float, help='margin of triplet')
parser.add_argument('--data_dir', default='/data0/kan/data/reid/1501/pytorch', type=str, help='training dir path')

opt = parser.parse_args()
data_dir = opt.data_dir
name = opt.name

# set gpu ids
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >=0:
        gpu_ids.append(gid)

if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
# prepossessing
transform_train_list = [transforms.Resize((384,128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

if opt.erasing_p>0:
    transform_train_list = transform_train_list + [RandomErasing(opt.erasing_p)]
if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list

data_transforms = {'train': transforms.Compose(transform_train_list)}

train_all = ''
if opt.train_all:
     train_all = '_all'

image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all), data_transforms['train'])
Sampler = BalancedBatchSampler(image_datasets, n_classes=opt.n_classes, n_samples=opt.n_images)
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.n_classes*opt.n_images, shuffle=False, sampler=Sampler)

dataset_sizes = len(image_datasets)
class_names = image_datasets.classes

use_gpu = torch.cuda.is_available()
inputs, classes = next(iter(dataloaders))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
def train_model(TSModel,
                Scriterion_part,
                Tcriterion_part,
                Scriterion_tri,
                criterion_cosine,
                optimizer, scheduler, num_epochs=70):

    # train for each epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        scheduler.step()
        TSModel.train(True)

        running_loss_part_s = 0.0
        running_loss_part_t = 0.0
        running_loss_mdd = 0.0
        running_loss_tri_s = 0.0

        # for each batch
        for data in dataloaders:
            inputs, labels, masks, tops, bottoms = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            inputs = [inputs, masks, tops, bottoms]

            optimizer.zero_grad()

            Slocal_feat_list, Slogits_list, Tlocal_feat_list, Tlogits_list, filter_list = TSModel(inputs)

            Spart_loss = torch.sum(torch.stack([Scriterion_part(logit, labels) for logit in Slogits_list]))
            Tpart_loss = torch.sum(torch.stack([Tcriterion_part(logit, labels) for logit in Tlogits_list]))

            ###########################
            # mdd loss
            Slocal_feat_concat = torch.cat(Slocal_feat_list, dim=1)
            Tlocal_feat_concat = torch.cat(Tlocal_feat_list, dim=1)

            person_num = opt.n_classes
            image_num = opt.n_images

            MDD_loss = 0
            for i in range(person_num):
                S_center = torch.mean(Slocal_feat_concat[i * image_num:(i + 1) * image_num, :],dim=0)
                T_center = torch.mean(Tlocal_feat_concat[i * image_num:(i + 1) * image_num, :],dim=0)
                MDD_loss += 1 - criterion_cosine(S_center, T_center)
            MDD_loss *= (1/person_num)

            ############################
            #  triplet loss for student
            Slocal_feat_concat_norm = 1. * Slocal_feat_concat / \
                                      (torch.norm(Slocal_feat_concat, 2, 1, keepdim=True).expand_as(
                                          Slocal_feat_concat) + 1e-12)
            Slocal_feat_concat_norm = Slocal_feat_concat_norm.view(Slocal_feat_concat_norm.size(0), -1)
            S_tri_loss = Scriterion_tri(Slocal_feat_concat_norm, labels)


            #########################
            # backward + optimize
            Tweight_part = 1
            Sweight_part = 1
            Aweight = 1
            Sweight_tri = 1

            loss =  Tweight_part * Tpart_loss + Sweight_part * Spart_loss + \
                    Aweight * MDD_loss + Sweight_tri * S_tri_loss

            loss.backward()
            optimizer.step()

            # statistics
            running_loss_part_s += Spart_loss.item()
            running_loss_part_t += Tpart_loss.item()
            running_loss_mdd += MDD_loss.item()
            running_loss_tri_s += S_tri_loss.item()

        epoch_loss_part_s = running_loss_part_s / dataset_sizes
        epoch_loss_part_t = running_loss_part_t / dataset_sizes
        epoch_loss_mdd = running_loss_mdd / dataset_sizes
        epoch_loss_tri_s = running_loss_tri_s / dataset_sizes

        print('SPartLoss: {:.4f}'.format(epoch_loss_part_s))
        print('TPartLoss: {:.4f}'.format(epoch_loss_part_t))
        print('AtriLoss: {:.4f}'.format(epoch_loss_mdd))
        print('StriLoss: {:.4f}'.format(epoch_loss_tri_s))

        # save model
        if epoch > 30:
            save_network(TSModel, epoch)
        print()

    return TSModel

#########################################################
# save model
def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./checks/mpn_no_ca/1501/1/',save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.to(device)

#########################################################
# define student
TSModel = TS(num_classes=751, num_stripes=6)
TSModel = TSModel.to(device)

##########################################################
#  set the criterion
triplet_selector_S = Hard(opt.margin1)
criterion_tri_S = OnlineTripletLoss(opt.margin1, triplet_selector_S)

criterion_part_S = nn.CrossEntropyLoss()
criterion_part_T = nn.CrossEntropyLoss()

criterion_cosine = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

param_groups = [{'params': TSModel.parameters(), 'lr': 0.01}]

optimizer_ft = optim.SGD(
             param_groups,
             momentum=0.9, weight_decay=5e-4, nesterov=True)

# rule for learning rate
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

model = train_model(TSModel,
                    criterion_part_S,
                    criterion_part_T,
                    criterion_tri_S,
                    criterion_cosine,
                    optimizer_ft, exp_lr_scheduler, num_epochs=70)
