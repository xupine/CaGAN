from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr

class Panchromatic(Dataset):

    NUM_CLASSES = 6
    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('panchromatic'),
                 split='train',
                 max_iters=None,
                 ):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir,split, 'src')
        self._cat_dir = os.path.join(self._base_dir, split,'label')
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        self.im_ids = []
        self.images = []
        self.categories = []
        n=len([name for name in os.listdir(self._image_dir) if os.path.isfile(os.path.join(self._image_dir, name))])
        for i in range(n):
            i=str(i)
            _image = os.path.join(self._image_dir, i + ".png")
            _cat = os.path.join(self._cat_dir, i + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_cat)
            self.im_ids.append(i)
            self.images.append(_image)
            self.categories.append(_cat)
        if not max_iters==None:
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
        _img = Image.open(self.images[index])
        _target = Image.open(self.categories[index])

        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.GNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.GToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.GNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.GToTensor()])

        return composed_transforms(sample)

    def __str__(self):
        return 'Panchromatic(split=' + str(self.split) + ')'      

