from utils.loggerer import Loggerer
import torch
from torch.autograd import Variable
import time
import numpy as np
from scipy.ndimage.filters import uniform_filter


def to_np(x):
    return x.data.cpu().numpy()

def decibel_to_linear(band):
    # convert to linear units
    return np.power(10, np.array(band) / 10)

def linear_to_decibel(band):
    return 10*np.log10(band)

def lee_filter(band, window, var_noise=0.25):
    # band: SAR data to be despeckled (already reshaped into image dimensions)
    # window: descpeckling filter window (tuple)
    # default noise variance = 0.25
    # assumes noise mean = 0
    band = decibel_to_linear(band)
    mean_window = uniform_filter(band, window)
    mean_sqr_window = uniform_filter(band ** 2, window)
    var_window = mean_sqr_window - mean_window ** 2

    weights = var_window / (var_window + var_noise)
    band_filtered = mean_window + weights * (band - mean_window)
    return linear_to_decibel(band_filtered)

def train_util(model, criterion, optimizer, data_loaders, scheduler, num_epochs, crops, params_str):

    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 10.0e8
    best_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            n = 0
            # Iterate over data.
            for data in data_loaders[phase]:
                # get the inputs
                if crops == 1:
                    inputs = data['images']
                    inputs = Variable(inputs.cuda().float())
                elif crops == 3:
                    inputs1, inputs2, inputs3 = data['images']
                    inputs1 = Variable(inputs1.cuda().float())
                    inputs2 = Variable(inputs2.cuda().float())
                    inputs3 = Variable(inputs3.cuda().float())
                labels = data['labels']
                angles = data['angle']
                # wrap them in Variable

                labels = Variable(labels.cuda().float())
                angles = Variable(angles.cuda().float())
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if crops == 1:
                    outputs = torch.squeeze(model(inputs, angles))
                elif crops == 3:
                    outputs = torch.squeeze(model(inputs1, inputs2, inputs3, angles))
                #print(labels)
                loss = criterion(outputs, torch.squeeze(labels))

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                n += 1
                predicts = (outputs.data > 0.5).float()
                accuracy = (predicts == torch.squeeze(labels).data).float().mean()
                running_corrects += accuracy
            epoch_loss = running_loss/n
            epoch_acc = running_corrects/n * 100

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val':
                scheduler.step(epoch_loss)
            if phase == 'train':
                train_loss = epoch_loss
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss and epoch_loss > train_loss:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

            # Set the logger
            logger = Loggerer('../logs/test/' + params_str)

            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            if phase == 'val':
                info = {
                    'loss': epoch_loss,
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, epoch + 1)

                # (2) Log values and gradients of the parameters (histogram)
                #for tag, value in model.named_parameters():
                #    tag = tag.replace('.', '/')
                #    logger.histo_summary(tag, to_np(value), epoch + 1)
                #    logger.histo_summary(tag + '/grad', to_np(value.grad), epoch + 1)

                 #(3) Log the images
                #images = inputs.view(-1, 75, 75)
                #info = {
                #    'images': to_np(images)
                #}
#
                #for tag, images in info.items():
                #    logger.image_summary(tag, images, epoch + 1)

    print()


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}, acc: {:4f}'.format(best_loss, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, best_loss

def predict_util(model, dataloader, crops):
    predicted = []
    for i, data in enumerate(dataloader):
        if crops == 1:
            inputs = data['images']
            inputs = Variable(inputs.cuda().float())
        elif crops == 3:
            inputs1, inputs2, inputs3 = data['images']
            inputs1 = Variable(inputs1.cuda().float())
            inputs2 = Variable(inputs2.cuda().float())
            inputs3 = Variable(inputs3.cuda().float())
        angles = Variable(data['angle'].cuda().float())
        if crops == 1:
            outputs_variable = torch.squeeze(model(inputs, angles))
        elif crops == 3:
            output = model(inputs1, inputs2, inputs3, angles)
            outputs_variable = torch.squeeze(output)
        outputs = outputs_variable.data.cpu().numpy()
        predicted.extend(outputs)
    return predicted