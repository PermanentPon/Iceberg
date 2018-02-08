"""
Data wrangling and NN training are done here.
Invoke train function in train_util when dataloaders are prepared.
Save model and make predictions.
@author: PermanentPon
"""

from utils.loggerer import Loggerer
import warnings
import argparse
from utils.custom_transforms import *
from models import my_models, vgg
import models.resnet as resnet
import models.densenet as densenet
import pandas as pd
import numpy as np
import predictor
import torch
from torch import nn
from torch.optim import Adam, lr_scheduler, SGD, RMSprop
from torch.utils.data import DataLoader
from utils.utils import train_util, lee_filter
from torchvision import transforms
from sklearn.model_selection import KFold, train_test_split
from os import path, makedirs
import gc

warnings.filterwarnings("ignore", category=UserWarning)

def get_model(model_name, channels, crops):
    """
    Create and return an instance of specified by model_name model.
    :param model_name: one of model names: MNIST, SimplestNet, VGG, ResNet, pretrained_VGG, DenseNet, trained_VGG, trained_DenseNet
    :param channels: number of channels (2, 3 or 4). For more details look at feature_engineering function
    :param crops: number of crops - int > 0. Usually less than 10.
    :return: ready to train model instance
    """
    if model_name == 'MNIST':
        model = my_models.MNIST().cuda()
    elif model_name == 'SimplestNet':
        model = my_models.SimplestNet().cuda()
    elif model_name == 'ResNet':
        model = resnet.resnet18().cuda()
    elif model_name == 'DenseNet':
        model = densenet.densnetXX_generic(1, channels).cuda()
    elif model_name == 'pretrained_ResNet':
        model = densenet.Pretrained_DenseNet(1).cuda()
    elif model_name == 'VGG':
        if crops == 3:
            model = vgg.vggnetcropsXX_generic(1, channels).cuda()
        else:
            model = vgg.vggnetXX_generic(1, channels).cuda()
    elif model_name == 'pretrained_VGG':
        model = vgg.vgg_pretrained(1, channels).cuda()
    elif model_name == 'trained_VGG':
        loaded_model = torch.load("../models/0.1973561350326539,pretrained_VGG,epochs=100,batch_size=64,optim=SGD,lr=0.001,moment=0.9,decay=1e-06,lr_sched=StepLR,step_size=20,factor=0.1,augmentation=['HFlip', 'VFlip', 'Rotation', 'Zoom'],channels=3,crops=1leak")
        model = my_models.Leak(loaded_model).cuda()
        for param in model.VGG.parameters():
            param.requires_grad = False
    elif model_name == 'trained_DenseNet':
        model = torch.load("../models/0.24378452465559045,pretrained_ResNet,epochs=50,batch_size=64,optim=SGD,lr=0.0001,moment=0.9,decay=1e-06,lr_sched=StepLR,step_size=10,factor=0.1,augmentation=['HFlip', 'VFlip', 'Rotation', 'Zoom'],channels=3,crops=1leak")
        for child in list(model.features.modules())[318:]:
            for param in child.parameters():
                param.requires_grad = True
    else:
        raise ValueError('You provided not supported model name - {}. '
                         'Please chose one of this models: MNIST, SimplestNet, ResNet'.format(model_name))
    model = nn.DataParallel(model)
    return model

def get_leak_features(angles):
    """
    Create ndarray with leak features based on inc_angle feature (Nx3: N - the number of images, 3 - the number of features):
    - inc_angle
    - the number of inc_angle occurrences in train and test set
    - the number of images with the same inc_angle and marked as icebergs
    :param angles: ndarray with inc_angle's (Nx1)
    :return: ndarray with enhanced features (Nx3)
    """
    train = pd.read_json("../data/processed/train.json")
    agg_df = train.groupby('inc_angle').agg({"is_iceberg": [len, np.sum]}).sort_values([('is_iceberg', 'len')],
                                                                                       ascending=False)
    test = pd.read_json('../data/processed/test.json')
    agg_df_test = test.groupby('inc_angle').agg('count').sort_values('id', ascending=False)
    agg_df = agg_df.join(agg_df_test)
    angles_df = pd.DataFrame(angles, columns=['inc_angle'])
    angles_df = angles_df.join(agg_df, on='inc_angle')
    angles_df = angles_df.fillna(-1.0)
    angles_df = angles_df.rename(columns={('is_iceberg', 'sum'): "sum", ('is_iceberg', 'len'): "len"})
    return angles_df[['inc_angle', 'len', 'id']].as_matrix()

def get_optimizer(model, optimizer_name, lr, decay, momentum):
    """
    Create and return an instance of specified by optimizer_name optimizer.
    :param model: model instance
    :param optimizer_name: one of optizers Adam, RMSProp, SGD
    :param lr: learning rate
    :param decay: weight decal (L2 regularization)
    :param momentum: momentum for SGD
    :return: optimizer instance
    """
    if optimizer_name == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=decay)
    elif optimizer_name == 'RMSProp':
        optimizer = RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    elif optimizer_name == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay, nesterov=True)
    else:
        raise ValueError('You/"ve provided not supported optimizer name - {}. '
                         'Please chose one of this optimizers: Adam, RMSprop, SGD'.format(optimizer_name))
    return optimizer

def get_scheduler(optimizer, scheduler_name, step_size, factor, patience):
    """
    Create and return an instance of the specified by scheduler_name scheduler.
    :param optimizer: optimizer instance
    :param scheduler_name: ReduceLROnPlateau or StepLR
    :param step_size: lr scheduler step_size
    :param factor: lr scheduler gamma
    :param patience: lr scheduler patience
    :return: scheduler instance
    """
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)
    elif scheduler_name == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size =step_size, gamma=factor)
    else:
        raise ValueError('You/"ve provided not supported scheduler name - {}. '
                         'Please chose one of this scheduler: '.format(scheduler_name))
    return scheduler

def normalize_data(imgs, angles):
    """
    Normalize test or train data
    :param imgs: image data
    :param angles: inc_angles
    :return: normalized image data(HH, HV bonds), inc_angles
    """
    imgs = (imgs - np.mean(imgs)) / np.std(imgs)
    angles = (angles - np.mean(angles))/np.std(angles)
    HH = imgs[:, :75*75]
    HV = imgs[:, 75*75:]
    return HH, HV, angles

def PCA_whitening_i(X):
    X_cov = np.dot(X.T, X)/X.shape[0]
    U, S, V = np.linalg.svd(X_cov)
    X_rot = np.dot(X, U)
    X_white = X_rot / np.sqrt(S + 1e-5)
    return X_white

def PCA_whitening(data, angles):
    data = (data - np.mean(data, axis=0))
    angles = (angles - np.mean(angles))
    data = PCA_whitening_i(data)
    HH = data[:, :75*75]
    HV = data[:, 75*75:]
    return HH, HV, angles

def get_data(phase, add_leak = False, normalize = False, PCA_whitening = False):
    """
    Load and return data(flatten images) from disk.
    Normalization and PCA_whitening can be applied.
    Leak features can be added.
    :param phase: train or test
    :param add_leak: flag if leak features should be added (True or False)
    :param normalize: flag if normalization should be applied (True or False)
    :param PCA_whitening: flag if PCA_whitening should be applied (True or False)
    :return: tuple of data(ndarrays) to train or test - images data, inc_angles and targets
    """
    if phase == 'train':
        data = pd.read_json('../data/processed/train.json')
        train_targets = data['is_iceberg'].values.astype(float)
    else:
        data = pd.read_json('../data/processed/test.json')
        ids = data['id'].values
    data['inc_angle'] = pd.to_numeric(data['inc_angle'], errors="coerce")
    data['inc_angle'] = data['inc_angle'].fillna(0.0)
    angles = data['inc_angle'].values.astype(float)
    HH = np.concatenate([im for im in data['band_1']]).reshape(-1, 75*75)
    HV = np.concatenate([im for im in data['band_2']]).reshape(-1, 75*75)

    if add_leak:
        angles = get_leak_features(angles)

    if normalize:
        angles = (angles - np.mean(angles)) / np.std(angles)
        HH, HV, angles = normalize_data(np.concatenate((HH, HV), axis = 1), angles)

    if PCA_whitening:
        HH = np.power(10, np.array(HH) / 10)
        HV = np.power(10, np.array(HV) / 10)
        HH, HV, angles = PCA_whitening(np.concatenate((HH, HV), axis = 1), angles)

    HH = HH.reshape(-1, 75, 75)
    HV = HV.reshape(-1, 75, 75)
    imgs = np.stack([HH, HV], axis=1)
    if phase == 'train':
        return (imgs, angles, train_targets)
    else:
        return (imgs, angles, ids)

def feature_engineering(imgs, channels, crops):
    """
    Enhance images with additional set of features:
    - channels == 4 adds pixels after Lee filter
    - channels == 3 adds combination of channels: (HH + HV) / 2
    - channels == 2 just HH and HV
    :param imgs: image data (ndarray)
    :param channels: number of channeks
    :param crops: number of crops - int > 0. Usually less than 10.
    :return: image data (ndarray)
    """
    imgs_list = []
    HH = imgs[:,0,:,:]
    HV = imgs[:,1, :, :]
    for i in range(crops):
        pix = 8
        HH_crop = HH[:, pix*i:75-pix*i, pix*i:75-pix*i]
        HV_crop = HV[:, pix*i:75-pix*i, pix*i:75-pix*i]
        #Lee filters
        HH_lee = np.zeros(HH_crop.shape)
        HV_lee = np.zeros(HH_crop.shape)
        for i in range(HH.shape[0]):
            HH_lee[i,:,:] = lee_filter(HH_crop[i,:,:], 2, 0.2)
            HV_lee[i, :, :] = lee_filter(HH_crop[i, :, :], 2, 0.2)
        if channels == 4:
            imgs = np.stack([HH_crop, HV_crop, HH_lee, HV_lee], axis=1)
        elif channels == 3:
            imgs = np.stack([HH_crop, HV_crop, (HH_crop+HV_crop)/2], axis=1)
        elif channels == 2:
            imgs = np.stack([HH_crop, HV_crop], axis=1)
        if crops == 1:
            imgs_list = imgs
        else:
            imgs_list.append(imgs)
    return imgs_list

def augment_create_datasets(X, angles, y, phase, augmentation, channels):
    """
    Create custom IceDataset with augmentations conteining in augmentation list.
    :param X:
    :param angles:
    :param y:
    :param phase:
    :param augmentation:
    :param channels:
    :return:
    """
    mutual_transforms = []
    mutual_transforms.append(ToTensor())
    all_transforms = mutual_transforms

    if phase == 'train':
        train_augments = []
        if 'Transform8' in augmentation:
            train_augments.append(RandomTransform8())
        if 'HFlip' in augmentation:
            train_augments.append(RandomHorizontalFlip())
        if 'VFlip' in augmentation:
            train_augments.append(RandomVerticallFlip())
        if 'Rotation' in augmentation:
            train_augments.append(RandomRotation(10))
        if 'Resize' in augmentation:
            train_augments.append(Resize((channels, 80, 80)))
        if 'FiveCrop' in augmentation:
            train_augments.append(FiveCrop(75))
        if 'Zoom' in augmentation:
            train_augments.append(Zoom((0.8, 1.2)))
        train_augments.extend(mutual_transforms)
        all_transforms = train_augments
    dataset = IceDataset(data=X, labels=y, angle=angles,
                                                 transform=transforms.Compose(all_transforms))
    return dataset

def train(train_dataset, val_dataset, model, epochs, batch_size, optimizer, scheduler, shuffle, num_workers, crops, params_str):
    """
    Create dataLoaders and invoke training function in train_util. Most of training logic can be found in train_util.
    """
    criterion = nn.BCELoss()#
    dataloaders = {'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
                   'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)}
    model, best_loss= train_util(model, criterion, optimizer, dataloaders, scheduler, epochs, crops, params_str)
    return model, best_loss

def save_model(model, str_path, model_name):
    """
    Save a model to disk
    :param model: model instance
    :param str_path: folder path to save
    :param model_name: name of file to save
    :return: None
    """
    torch.save(model, path.join(str_path, model_name))
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--model', type=str, default='VGG', help='chose one of models MNIST, SimplestNet, VGG, ResNet, pretrained_VGG, DenseNet, trained_VGG, trained_DenseNet')
    parser.add_argument('--kFolds', type=float, default=5, help='number of foldes for Cross Validation. '
                                                                '0 - no k-fold cross-validation')
    parser.add_argument('--epochs', type=int, default=90, help='Max number of epochs to train.')
    parser.add_argument('--batch_size', default=64, type=int, help='train batchsize')
    parser.add_argument('--optimizer', type=str, default='SGD', help='chose one of optizers Adam, RMSProp, SGD')
    parser.add_argument('--scheduler', default='StepLR', type=str, help='ReduceLROnPlateau or StepLR')
    parser.add_argument('--step_size', default=30, type=int, help='lr scheduler step_size')
    parser.add_argument('--factor', default='0.1', type=float, help='lr scheduler gamma ')
    parser.add_argument('--patience', default='20', type=int, help='patience ')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum if RMSprop or SGD')
    parser.add_argument('--decay', type=float, default=0.0001, help='Weight decay (L2 penalty).')
    parser.add_argument('--augmentation', type=str, nargs='+', default=['HFlip', 'VFlip', 'FiveCrop'],
                        help='Type of augmentation: HFlip, VFlip, Rotation, Zoom, Transform8, FiveCrop')
    parser.add_argument('--channels', default=2, type=int, help='number of Image channels')
    parser.add_argument('--crops', default=1, type=int, help='number of crops')
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 0)')
    parser.add_argument('--manualSeed', type=int, default=111, help='manual seed')
    parser.add_argument('--modelsPath', type=str, default='../models', help='Path to save models')
    parser.add_argument('--predict', type=int, default=True, help='predict')
    args = parser.parse_args()

    #Grid search of gyperparameters
    lrs = [0.001, 0.0005, 0.0001]
    decays = [0.0001, 0.00005, 0.00001]
    augmentation = ['HFlip', 'VFlip', 'Rotation', 'FiveCrop']
    step_size = [20, 30, 40]
    for j in range(100):
        args.lr = random.choice(lrs)
        args.decay = random.choice(decays)
        args.step_size = random.choice(step_size)
        args.augmentation = random.sample(augmentation, random.randint(2,4))
        args.manualSeed = random.randint(1,1000)
        train_imgs, train_angles, train_targets = get_data('train')

        params_str = args.model + ',epochs=' + str(args.epochs) + ',batch_size=' + str(args.batch_size) + ',optim=' + args.optimizer + ',lr=' + str(args.lr) \
                     + ',moment=' + str(args.momentum) + ',decay=' + str(args.decay) + ',lr_sched=' + args.scheduler + ',step_size=' + str(args.step_size) \
                     + ',factor='+ str(args.factor) + ',augmentation=' + str(args.augmentation) + ',channels='+ str(args.channels) + ',crops=' + str(args.crops)

        models_folder = path.join(args.modelsPath, "5fold_{}_{}".format(j, params_str))
        if not path.exists(models_folder):
            makedirs(models_folder)
        if args.kFolds > 1: # if cross-validation with several folds is used
            kf = KFold(n_splits=args.kFolds, shuffle=True, random_state=args.manualSeed)
            sum_loss = 0.0
            for i, (train_index, test_index) in enumerate(kf.split(train_imgs)):

                model = get_model(model_name=args.model, channels=args.channels, crops=args.crops)
                optimizer = get_optimizer(model=model, optimizer_name=args.optimizer, lr=args.lr, decay=args.decay,
                                          momentum=args.momentum)
                scheduler = get_scheduler(optimizer=optimizer, scheduler_name=args.scheduler, step_size=args.step_size,
                                          factor=args.factor, patience=args.patience)

                X_train, X_valid = train_imgs[train_index], train_imgs[test_index]
                y_train, y_valid = train_targets[train_index], train_targets[test_index]
                X_train = feature_engineering(X_train, args.channels, args.crops)
                X_valid = feature_engineering(X_valid, args.channels, args.crops)
                angles_train, angles_valid = train_angles[train_index], train_angles[test_index]
                train_dataset = augment_create_datasets(X_train, angles_train, y_train,
                                                        phase = 'train', augmentation=args.augmentation, channels=args.channels)
                val_dataset = augment_create_datasets(X_valid, angles_valid, y_valid,
                                                      phase = 'val', augmentation=args.augmentation, channels=args.channels)
                model, best_loss = train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, epochs=args.epochs, batch_size=args.batch_size,
                              optimizer=optimizer, scheduler=scheduler, shuffle=True, num_workers=args.workers, crops = args.crops, params_str=params_str)
                sum_loss += best_loss
                save_model(model, str_path=models_folder, model_name= str(best_loss) + ',fold=' + str(i))

            print("Avarage best loss - {}".format(sum_loss/args.kFolds))
            print("Model params - {}".format(params_str))

        else:
            model = get_model(model_name=args.model, channels=args.channels, crops=args.crops)
            optimizer = get_optimizer(model=model, optimizer_name=args.optimizer, lr=args.lr, decay=args.decay,
                                      momentum=args.momentum)
            scheduler = get_scheduler(optimizer=optimizer, scheduler_name=args.scheduler, step_size=args.step_size,
                                      factor=args.factor, patience=args.patience)

            X_train, X_valid, angles_train, angles_valid, y_train, y_valid = train_test_split(train_imgs, train_angles, train_targets, test_size=0.20, random_state=args.manualSeed)
            X_train = feature_engineering(X_train, args.channels, args.crops)
            X_valid = feature_engineering(X_valid, args.channels, args.crops)
            train_dataset = augment_create_datasets(X_train, angles_train, y_train,
                                                    phase = 'train', augmentation=args.augmentation, channels=args.channels)
            val_dataset = augment_create_datasets(X_valid, angles_valid, y_valid,
                                                  phase = 'val', augmentation=args.augmentation, channels=args.channels)
            model, best_loss = train(train_dataset=train_dataset, val_dataset=val_dataset, model=model, epochs=args.epochs,
                          batch_size=args.batch_size, optimizer=optimizer, scheduler=scheduler, shuffle=True,
                          num_workers=args.workers, crops = args.crops, params_str=params_str)
            save_model(model, str_path=args.modelsPath, model_name=str(best_loss) + ','+ params_str + "leak")
            print("Saved model - {}, {}".format(best_loss, params_str))

        if args.predict: # if predict right after training
            df_pred = predictor.predict(models_folder, channels = args.channels, crops = args.crops, augmentation =[])
            df_pred.to_csv('../results/ens/5fold_{}.csv'.format(best_loss), index=None)
            print("Predictions saved to {}". format('../results/ens/5fold_{}_{}.csv'.format(sum_loss/args.kFolds, params_str)))
            print("#"*20)
            print()
        gc.collect()