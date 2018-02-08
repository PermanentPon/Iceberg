import pandas as pd
import numpy as np
import skimage.transform as transform
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter
import random


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


if __name__ == '__main__':
    train_val = pd.read_json('../data/processed/train.json')
    HH = np.concatenate([im for im in train_val['band_1']]).reshape(-1, 75, 75)
    HV = np.concatenate([im for im in train_val['band_2']]).reshape(-1, 75, 75)
    imgs = np.stack([HH, HV, (HH + HV) / 2], axis=1)

    image = imgs[138, :, :, :]
    #img = img+255
    plt.subplot(1, 2, 1)
    plt.imshow(image[1,:,:])

    image = transform.resize(image, (3,80,80), mode = 'symmetric')
    #plt.imshow(image[1,:,:])
    #plt.show()
    _, w, h = image.shape
    crop_h, crop_w = (75,75)
    a = random.randint(1, 5)
    if a == 1:
        resized_img = crop(image, 0, 0, crop_w, crop_h)
    elif a == 2:
        resized_img = crop(image, w - crop_w, 0, w, crop_h)
    elif a == 3:
        resized_img = crop(image, 0, h - crop_h, crop_w, h)
    elif a == 4:
        resized_img = crop(image, w - crop_w, h - crop_h, w, h)
    elif a == 5:
        resized_img = center_crop(image, (crop_h, crop_w))

    #img2 = transform.rotate(img, 45, mode = 'symmetric')
    #img3 = lee_filter(decibel_to_linear(img), 2, 0.001)
    plt.subplot(1, 2, 2)
    plt.imshow(resized_img[1,:,:])
    plt.show()