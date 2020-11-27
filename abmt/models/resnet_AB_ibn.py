from __future__ import absolute_import
import random
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
from torchvision.models.resnet import Bottleneck
from copy import deepcopy
from .resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
__all__ = ['resnet_ibn50a_AB']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


class ResNetIBN(nn.Module):
    __factory = {
        50: resnet50_ibn_a,
        101: resnet101_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(ResNetIBN, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        # Construct base (pretrained) resnet
        if depth not in ResNetIBN.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNetIBN.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)
        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3)
        self.num_classes = num_classes
        out_planes = resnet.fc.in_features
        self.num_features = out_planes
        # 1st branch
        self.global_branch = resnet.layer4
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        self.feat_bn.bias.requires_grad_(False)
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        # 2nd branch
        self.max_branch = nn.Sequential(deepcopy(resnet.layer4), Bottleneck(2048, 512))
        self.max_branch[1].apply(weights_init_kaiming)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.feat_bn_max = nn.BatchNorm1d(self.num_features)
        self.feat_bn_max.bias.requires_grad_(False)
        if self.num_classes > 0:
            self.classifier_max = nn.Linear(self.num_features, self.num_classes, bias=False)
            init.normal_(self.classifier_max.weight, std=0.001)
        init.constant_(self.feat_bn_max.weight, 1)
        init.constant_(self.feat_bn_max.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, feature_withbn=False):
        x = self.base(x)
        # 1st branch
        x_g = self.global_branch(x)
        x_g = self.gap(x_g)
        x_g = x_g.view(x_g.size(0), -1)
        bn_x_g = self.feat_bn(x_g)

        # 2nd branch
        x_m = self.max_branch(x)
        x_m = self.gmp(x_m)
        x_m = x_m.view(x_m.size(0), -1)
        bn_x_m = self.feat_bn_max(x_m)

        if self.training is False:
            bn_x = F.normalize(torch.cat((bn_x_g, bn_x_m), dim=1))
            return bn_x

        if self.num_classes > 0:
            prob_g = self.classifier(bn_x_g)
            prob_m = self.classifier_max(bn_x_m)
        else:
            return x_g, bn_x_g, x_m, bn_x_m

        if feature_withbn:
            return bn_x_g, bn_x_m, prob_g, prob_m

        return x_g, x_m, prob_g, prob_m

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

        resnet = ResNetIBN.__factory[self.depth](pretrained=self.pretrained)
        self.base[0].load_state_dict(resnet.conv1.state_dict())
        self.base[1].load_state_dict(resnet.bn1.state_dict())
        self.base[2].load_state_dict(resnet.relu.state_dict())
        self.base[3].load_state_dict(resnet.maxpool.state_dict())
        self.base[4].load_state_dict(resnet.layer1.state_dict())
        self.base[5].load_state_dict(resnet.layer2.state_dict())
        self.base[6].load_state_dict(resnet.layer3.state_dict())
        self.base[7].load_state_dict(resnet.layer4.state_dict())


def resnet_ibn50a_AB(**kwargs):
    return ResNetIBN(50, **kwargs)

