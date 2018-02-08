"""
Predict classes based on test data with NNs.
@author: PermanentPon
"""

from trainer import *
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from utils.utils import predict_util
import pathlib
import os
import torch

def get_transformed_testsets(channels, crops):
    """
    Load data from disk and apply transformation (TTA)
    :param channels: number of channels (2, 3 or 4). For more details look at feature_engineering function
    :param crops: number of crops - int > 0. Usually less than 10.
    :return: ndarrays with test data: image data, inc_angles and targets
    """

    # load test data from competition
    test_imgs, test_angles, ids = get_data('test')
    # check if transformed test data saved in the file exists
    transformed_imgs_path = '../temp/transformed_imgs.npz'
    p = pathlib.Path(transformed_imgs_path)
    if p.is_file():
        with np.load(transformed_imgs_path) as data:
            transformed_img_set = [data[file] for file in data.files]
    else: # if file doesn't exist transform images and save them into a file
        transformed_img_set = []
        #transform images by applying 7 different transformation to every image
        for transform in ['H', 'V', '90', '180', '270', '90H', '90V']:
            transformed_imgs = np.zeros(test_imgs.shape)
            for i, image in enumerate(test_imgs):
                if transform == 'H':
                    transformed_image = np.flip(image, 1)
                if transform == 'V':
                    transformed_image = np.flip(image, 0)
                if transform == '90':
                    transformed_image = np.rot90(image, 1, (2, 1))
                if transform == '180':
                    transformed_image = np.rot90(image, 2, (2, 1))
                if transform == '270':
                    transformed_image = np.rot90(image, 1, (1, 2))
                if transform == '90H':
                    transformed_image = np.rot90(image, 1, (2, 1))
                    transformed_image = np.flip(transformed_image, 1)
                if transform == '90V':
                    transformed_image = np.rot90(image, 1, (2, 1))
                    transformed_image = np.flip(transformed_image, 1)
                transformed_imgs[i,:,:,:] = transformed_image
            transformed_img_set.append(transformed_imgs)
        # save transformed images into a file
        with open(transformed_imgs_path, 'wb') as f:
            np.savez(f, *transformed_img_set)
    #add initial images to list of image sets (len = 8)
    transformed_img_set.append(test_imgs)

    return transformed_img_set, test_angles, ids

def get_dataset(channels, crops):
    """
    Load test data and create dataset
    :param channels: number of channels (2, 3 or 4). For more details look at feature_engineering function
    :param crops: number of crops - int > 0. Usually less than 10.
    :return: test dataset to predict and targets
    """
    transformed_img_set, test_angles, ids = get_transformed_testsets(channels=channels, crops=crops)
    for transformed_imgs in transformed_img_set:
        transformed_imgs = feature_engineering(transformed_imgs, channels=channels, crops=crops)
        dataset = augment_create_datasets(X=transformed_imgs, angles=test_angles, y=np.zeros(len(test_angles)),
                                          phase = 'test', augmentation=[], channels=channels)
        yield dataset, ids

def get_models(models_path):
    """
    Load models from the specified location
    :param models_path: folder path with models
    :return: instances of models
    """
    for model_path in os.listdir(models_path):
        model = torch.load(models_path + '/' + model_path)
        yield model, model_path

def get_num_models(models_path):
    """
    Count number of models/files in the specified location
    :param models_path: folder path with models
    :return: number of models/files
    """
    return len(os.listdir(models_path))

def predict(models_path, channels, crops, augmentation):
    i = 0
    for i, (model, model_name) in enumerate(get_models(models_path), 1):
        print("Start prediction with model {}/{} - {}".format(i, get_num_models(models_path), model_name))
        if 'Transform8' in augmentation:
            for j, (test_dataset, ids) in enumerate(get_dataset(channels=channels, crops=crops), 1):
                dataloader = DataLoader(test_dataset, batch_size=66, shuffle=False, num_workers=1)
                predict = predict_util(model, dataloader, crops)
                if (j == 1):
                    predicts_transform = np.empty(len(predict), 8)
                    if (i == 1):
                        predicts = np.empty((len(predict), get_num_models(models_path)))
                predicts_transform[:, j - 1] = predict
                print('Predicted with {} transform(s) out of 8'.format(j))
            predicts[:, i - 1] = np.mean(predicts_transform, axis = 1)
        else:
            test_imgs, test_angles, ids = get_data('test')
            test_imgs = feature_engineering(test_imgs, channels, crops)
            dataset = augment_create_datasets(X=test_imgs, angles=test_angles, y=np.zeros(len(test_angles)),
                                              phase='test', augmentation=[], channels=channels)

            dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)
            predict = predict_util(model, dataloader, crops)
            if (i == 1):
                predicts = np.empty((len(predict), get_num_models(models_path)))
            predicts[:, i - 1] = predict
    ids = ids[:, np.newaxis]

    data = np.concatenate((ids, predicts), axis=1)
    num = get_num_models(models_path)
    columns = ['id', *list(map(lambda x: "is_iceberg_" + str(x), range(num)))]
    df_pred = pd.DataFrame(data=data, columns=columns)
    return df_pred

if __name__ == '__main__':
    torch.nn.Module.dump_patches = True
    path = "../models/one_best_leak"
    df_pred = predict(path, channels=3, crops=1, augmentation=[])
    df_pred.to_csv('../results/one_best_leak.csv', index=None)
    print('saved results')
