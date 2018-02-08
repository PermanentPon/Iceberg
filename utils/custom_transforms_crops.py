"""

@author: PermanentPon
"""
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import warnings
import skimage.transform as transform

class IceDataset(Dataset):
    """total dataset."""
    def __init__(self, data, labels, angle, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.angle = angle

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = []
        for i, imgs in enumerate(self.data):
            img.append(imgs[idx, :, :, :])
        sample = {'images': img, 'labels': np.asarray([self.labels[idx]]),
                  'angle': np.asarray([self.angle[idx]])}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        imgs = []
        images, labels, angle = sample['images'], sample['labels'], sample['angle']
        for image in images:
            imgs.append(torch.from_numpy(image.copy()).float())
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        #image = image.astype(float) / 255
        return {'images': imgs,
                'labels': torch.from_numpy(labels).long(),
                'angle': torch.from_numpy(angle).float()}

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img numpy: Image to be flipped.

        Returns:
            img numpy: Randomly flipped image.
        """
        imgs = []
        images, labels, angle = sample['images'], sample['labels'], sample['angle']
        for image in images:
            if random.random() < 0.5:
                image = np.flip(image, 1)
            imgs.append(image)

        return {'images': imgs, 'labels': labels, 'angle': angle}

class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        imgs = []
        images, labels, angle = sample['images'], sample['labels'], sample['angle']
        for image in images:
            if random.random() < 0.5:
                image = np.flip(image, 0)
            imgs.append(image)
        return {'images': imgs, 'labels': labels, 'angle': angle}

class RandomRotation(object):
    """Rotation"""
    def __init__(self, degrees):

        self.degrees = degrees

    def __call__(self, sample):
        """
        Args:
            img (ndarray): Image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        imgs = []
        images, labels, angle = sample['images'], sample['labels'], sample['angle']
        for image in images:
            rotated_img = transform.rotate(image, self.degrees, mode='symmetric')
            imgs.append(rotated_img)
        return {'images': imgs, 'labels': labels, 'angle': angle}


class RandomTransform8(object):
    """Rotation"""

    def __call__(self, sample):
        """
        Args:
            img (ndarray): Image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        transforms = ['H', 'V', '90', '180', '270', '90H', '90V']

        imgs = []
        images, labels, angle = sample['images'], sample['labels'], sample['angle']
        transform = random.choice(transforms)
        for image in images:
            if transform == 'H':
                transformed_image = np.flip(image, 2)
            if transform == 'V':
                transformed_image = np.flip(image, 1)
            if transform == '90':
                transformed_image = np.rot90(image, 1, (2, 1))
            if transform == '180':
                transformed_image = np.rot90(image, 2, (2, 1))
            if transform == '270':
                transformed_image = np.rot90(image, 1, (1, 2))
            if transform == '90H':
                transformed_image = np.rot90(image, 1, (2, 1))
                transformed_image = np.flip(transformed_image, 2)
            if transform == '90V':
                transformed_image = np.rot90(image, 1, (2, 1))
                transformed_image = np.flip(transformed_image, 1)
            imgs.append(transformed_image)
        return {'images': imgs, 'labels': labels, 'angle': angle}
    
class Resize(object):
    """Rotation"""

    def __init__(self, new_size):

        self.new_size = new_size

    def __call__(self, sample):
        """
        Args:
            img (ndarray): Image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        image, labels, angle = sample['image'], sample['labels'], sample['angle']
        resized_img = transform.resize(image, self.new_size)

        return {'image': resized_img, 'labels': labels, 'angle': angle}


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """

    return img[:, i:h, j:w]

def center_crop(img, output_size):
    _, w, h = img.shape
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return img[:, i:i+th, j:j+tw]

class FiveCrop(object):
    """Rotation"""

    def __init__(self, new_size_1, new_size_2, new_size_3):

        self.new_size = [new_size_1, new_size_2, new_size_3]

    def __call__(self, sample):
        """
        Args:
            img (ndarray): Image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        pad = 3
        imgs = []
        images, labels, angle = sample['images'], sample['labels'], sample['angle']
        for i, image in enumerate(images):
            image = np.pad(image, ((0,0),(pad,pad),(pad,pad)),  mode='edge')
            _, w, h = image.shape
            crop_h = self.new_size[i]
            crop_w = self.new_size[i]
            a = random.randint(1, 5)
            if a == 1:
                resized_img = crop(image, 0, 0, crop_w, crop_h)
            elif a ==2:
                resized_img = crop(image, w - crop_w, 0, w, crop_h)
            elif a == 3:
                resized_img = crop(image,0, h - crop_h, crop_w, h)
            elif a == 4:
                resized_img = crop(image,w - crop_w, h - crop_h, w, h)
            elif a == 5:
                resized_img = center_crop(image, (crop_h, crop_w))
            imgs.append(resized_img)
        return {'images': imgs, 'labels': labels, 'angle': angle}


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        imgs = []
        images = tensor['images']
        for img in images:
            for t, m, s in zip(img, self.mean, self.std):
                t.sub_(m).div_(s)
        return {'images': img, 'labels': tensor['labels'], 'angle': tensor['angle']}