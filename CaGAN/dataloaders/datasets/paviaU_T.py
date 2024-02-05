from __future__ import print_function, division

import os

import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataloaders import custom_transforms as tr
from mypath import Path
from random import shuffle


class PaviaU_T(Dataset):
    NUM_CLASSES = 9
    def __init__(self, args, base_dir=Path.db_root_dir('paviaU_T'), split='train', max_iters=None):
        super().__init__()
        self._base_dir = base_dir
        self.dir = os.path.join(self._base_dir, split)
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        for split in self.split:
            if split == "train":
                patch_type = "train"
            elif split == 'val':
                patch_type = "testing"

        self.args = args
        self.im_ids = []
        self.images = []
        self.categories = []
        i = 0
        print(len(os.listdir(self.dir)))
        for file in os.listdir(self.dir):
            if os.path.isfile(os.path.join(self.dir, file)):
                temp = scipy.io.loadmat(os.path.join(self.dir, file))
                #temp = {k: v for k, v in temp.items() if k[0] != '_'}
                self.im_ids.append(i)
                self.images.append(temp[patch_type + "_patches"])
                #print(temp[patch_type + "_patches"].shape)
                self.categories.append(temp[patch_type + "_labels"])
                #print(temp[patch_type + "_labels"].shape)
                i += 1
                #print(i)
        
        if not max_iters == None:
            print(len(self.im_ids))
            self.im_ids = self.im_ids * int(np.ceil(float(max_iters) / len(self.im_ids)))
            self.images = self.images * int(np.ceil(float(max_iters) / len(self.images)))
            self.categories = self.categories * int(np.ceil(float(max_iters) / len(self.categories)))

        assert (len(self.images) == len(self.categories))
        print(self.split, len(self.images))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample)


    def _make_img_gt_point_pair(self, index):
        _img = self.images[index]
        _target = self.categories[index]

        return _img, _target


    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.ToTensor()])

        return composed_transforms(sample)


    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.ToTensor()])

        return composed_transforms(sample)


    def __str__(self):
        return 'PaviaU_T(split=' + str(self.split) + ')'
