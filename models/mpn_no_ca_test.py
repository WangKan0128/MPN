import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from .base import FtNet  # 24*8

##############################################################
class TS(nn.Module):
    def __init__(
        self,
        num_stripes = 6,
        Slocal_conv_channels = 512,
        Tlocal_conv_channels = 512,
        num_classes=751,
    ):
        super(TS, self).__init__()

        self.num_stripes = num_stripes

        ########################################################
        #  the base
        model = FtNet(class_num=num_classes)

        # fine tuning from IDE Model
        model.load_state_dict(torch.load('./data/1501.pth'))

        self.base = nn.Sequential(*list(model.model.children())[:-2])

        self.local_conv_list = nn.ModuleList()
        for _ in range(self.num_stripes):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048, Tlocal_conv_channels, 1),
                nn.BatchNorm2d(Tlocal_conv_channels),
                nn.ReLU(inplace=True)
            ))

        self.local_conv_list2 = nn.ModuleList()
        for _ in range(self.num_stripes):
            self.local_conv_list2.append(nn.Sequential(
                nn.Conv2d(Tlocal_conv_channels, Tlocal_conv_channels, 1),
                nn.BatchNorm2d(Tlocal_conv_channels),
                nn.ReLU(inplace=True)
            ))

        ########################################################
        # for student
        self.Sfc_list = nn.ModuleList()
        for _ in range(self.num_stripes):
            fc = nn.Linear(Slocal_conv_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.Sfc_list.append(fc)

        ########################################################
        # for teacher
        self.Tfc_list = nn.ModuleList()  # very useful tips
        for _ in range(self.num_stripes):
            fc = nn.Linear(Tlocal_conv_channels, num_classes)
            init.normal_(fc.weight, std=0.001)
            init.constant_(fc.bias, 0)
            self.Tfc_list.append(fc)

    def forward(self, x):

        ######################################################
        # the base network
        feat = self.base(x)

        ######################################################
        # the student
        S_local_feat_list = []

        for i in range(self.num_stripes):
            local_feat = feat
            local_feat = self.local_conv_list[i](local_feat)
            local_feat = F.max_pool2d(local_feat, (feat.size(2), feat.size(3)))
            local_feat = self.local_conv_list2[i](local_feat)

            local_feat = local_feat.view(local_feat.size(0), -1)
            S_local_feat_list.append(local_feat)

        return S_local_feat_list


