import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import matplotlib.pyplot as plt


class Test_Dataset(Dataset):
    def __init__(self, dataroot, resolution=256, split='train', data_len=20):
        self.resolution = resolution
        self.data_len = data_len
        self.split = split
        self.vis_path = Util.get_paths_from_images('{}/CT-PET-SPECT'.format(dataroot))
        self.ir_path = Util.get_paths_from_images('{}/MRI'.format(dataroot))

        self.dataset_len = len(self.vis_path)

        if self.data_len <= 0:
            self.data_len = self.dataset_len
        else:
            self.data_len = min(self.data_len, self.dataset_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_vis = Image.open(self.vis_path[index]).convert("YCbCr")
        img_ir = Image.open(self.ir_path[index]).convert("YCbCr")

        img_full = Image.open(self.vis_path[index]).convert("RGB")

        img_vis = self.resize_to_multiple_of_8(img_vis)
        img_ir = self.resize_to_multiple_of_8(img_ir)
        img_full = self.resize_to_multiple_of_8(img_full)

        img_vis = img_vis.split()[0]
        img_ir = img_ir.split()[0]

        if self.split == "val":
            [img_vis, img_ir] = Util.transform_augment([img_vis, img_ir], split=self.split, min_max=(-1, 1))
            img_full = Util.transform_full(img_full, min_max=(-1, 1))
            path = str(self.vis_path[index])
            path = path.replace("\\", "/")
            name = str(path.split("/")[-1].split(".png")[0])
            return {'vis': img_vis, 'ir': img_ir, 'img_full': img_full, 'Index': index}, name
        else:
            [img_vis, img_ir], *crop_params = Util.transform_augment([img_vis, img_ir], split=self.split, min_max=(-1, 1))
            img_full = Util.transform_full_augment(img_full, *crop_params, min_max=(-1, 1))
            return {'vis': img_vis, 'ir': img_ir, 'img_full': img_full, 'Index': index}

    def resize_to_multiple_of_8(self, img):
        width, height = img.size
        new_width = width - (width % 8)
        new_height = height - (height % 8)
        img_resized = img.resize((new_width, new_height))
        return img_resized