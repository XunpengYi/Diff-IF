import os
import torch
import torchvision
import random
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img

totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
vflip = torchvision.transforms.RandomVerticalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        crop_params = T.RandomCrop.get_params(imgs[0], (128, 128))
        imgs = [F.crop(img, *crop_params) for img in imgs]
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = vflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
        ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
        return ret_img, *crop_params
    else:
        ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
        return ret_img

def transform_full_augment(img, *crop_params, min_max=(0, 1)):
    img = totensor(img)
    img = F.crop(img, *crop_params)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img

def transform_full(img, min_max=(0, 1)):
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img