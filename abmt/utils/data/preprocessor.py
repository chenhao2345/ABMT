from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, mutual=False, index=False, multi_transform=1):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.mutual = mutual
        self.index = index
        self.multi_transform = multi_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if self.multi_transform >= 3:
            return self._get_multiple_items(indices)

        if self.mutual:
            return self._get_mutual_item(indices)
        else:
            return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid

    def _get_mutual_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img_1 = Image.open(fpath).convert('RGB')
        img_2 = img_1.copy()

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        if self.index:
            return img_1, img_2, pid, index

        return img_1, img_2, pid

    def _get_multiple_items(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.multi_transform == 3:
            img_1 = self.transform(img)
            img_2 = self.transform(img.copy())
            img_3 = self.transform(img.copy())
            return img_1, img_2, img_3, pid

        elif self.multi_transform == 4:
            img_1 = self.transform(img)
            img_2 = self.transform(img.copy())
            img_3 = self.transform(img.copy())
            img_4 = self.transform(img.copy())
            return img_1, img_2, img_3, img_4, pid
        else:
            raise RuntimeError("multi_transform can be set to 3 or 4")

