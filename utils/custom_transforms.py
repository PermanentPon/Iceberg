"""

@author: PermanentPon
"""
import torch
import numpy as np
from torch.utils.data import Dataset
import random
import skimage.transform as transform
import scipy.ndimage as ndi

#warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        sample = {'images': self.data[idx, :, :, :], 'labels': np.asarray([self.labels[idx]]),
                  'angle': np.asarray([self.angle[idx]])}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels, angle = sample['images'], sample['labels'], sample['angle']


        return {'images': torch.from_numpy(image.copy()).float(),
                'labels': torch.from_numpy(labels).long(),
                'angle': torch.from_numpy(angle).float()}


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels, angle = sample['images'], sample['labels'], sample['angle']

        if random.random() < 0.5:
            image = np.flip(image, 1)

        return {'images': image, 'labels': labels, 'angle': angle}


class RandomVerticallFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
            img (PIL.Image): Image to be flipped.

        Returns:
            PIL.Image: Randomly flipped image.
        """
        image, labels, angle = sample['images'], sample['labels'], sample['angle']

        if random.random() < 0.5:
            image = np.flip(image, 0)

        return {'images': image, 'labels': labels, 'angle': angle}

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
        image, labels, angle = sample['images'], sample['labels'], sample['angle']
        degree = random.randint(0, self.degrees)
        rotated_img = transform.rotate(image, degree, mode='symmetric')

        return {'images': rotated_img, 'labels': labels, 'angle': angle}

def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is
    transformed about the center of the image.
    NOTE: This is a fairly simple operaion, so can easily be
    moved to full torch.
    Arguments
    ---------
    matrix : 3x3 matrix/array
    x : integer
        height dimension of image to be transformed
    y : integer
        width dimension of image to be transformed
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='nearest', fill_value=0.):
    """Applies an affine transform to a 2D array, or to each channel of a 3D array.
    NOTE: this can and certainly should be moved to full torch operations.
    Arguments
    ---------
    x : np.ndarray
        array to transform. NOTE: array should be ordered CHW

    transform : 3x3 affine transform matrix
        matrix to apply
    """
    x = x.astype('float32')
    transform = transform_matrix_offset_center(transform, x.shape[1], x.shape[2])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                         final_offset, order=0, mode=fill_mode, cval=fill_value) for
                      x_channel in x]
    x = np.stack(channel_images, axis=0)
    return x

class Zoom(object):
    """Zoom"""


    def __init__(self,
                 zoom_range,
                 fill_mode='constant',
                 fill_value=0,
                 target_fill_mode='nearest',
                 target_fill_value=0.,
                 lazy=False):

        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, sample):
        """
        Args:
            img (ndarray): Image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        image, labels, angle = sample['images'], sample['labels'], sample['angle']
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if self.lazy:
            return zoom_matrix
        else:
            zoomed_img = apply_transform(image, zoom_matrix, fill_mode=self.fill_mode,
                                                             fill_value=self.fill_value)

        return {'images': zoomed_img, 'labels': labels, 'angle': angle}



class RandomTransform8(object):
    """Rotation"""

    def __call__(self, sample):
        """
        Args:
            img (ndarray): Image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        #transforms = ['H', 'V', '90', '180', '270', '90H', '90V']
        transforms = ['H', 'V', '180']
        image, labels, angle = sample['images'], sample['labels'], sample['angle']
        transform = random.choice(transforms)
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
        return {'images': transformed_image, 'labels': labels, 'angle': angle}


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
        image, labels, angle = sample['images'], sample['labels'], sample['angle']
        resized_img = transform.resize(image, self.new_size,mode = 'symmetric')

        return {'images': resized_img, 'labels': labels, 'angle': angle}


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

    def __init__(self, new_size):

        self.new_size = new_size

    def __call__(self, sample):
        """
        Args:
            img (ndarray): Image to be flipped.

        Returns:
            ndarray: Randomly flipped image.
        """
        pad = 3
        image, labels, angle = sample['images'], sample['labels'], sample['angle']
        image = np.pad(image, ((0,0),(pad,pad),(pad,pad)),  mode='edge')
        _, w, h = image.shape
        crop_h = self.new_size
        crop_w = self.new_size
        a = random.randint(1, pad)
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

        return {'images': resized_img, 'labels': labels, 'angle': angle}

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
        img = tensor['images'].float()
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'images': img, 'labels': tensor['labels'], 'angle': tensor['angle']}

