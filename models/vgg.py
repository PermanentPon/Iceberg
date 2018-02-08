'''VGG11/13/16/19 in Pytorch.'''
import torch.nn as nn
import torch
from torchvision import transforms, models

__all__ = ['vggnetXX_generic']

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, num_rgb):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], num_rgb)
        self.num_classes=num_classes
        self.dropout = nn.Dropout(0.3)#new
        self.classifier = nn.Linear(2049, num_classes)

        self.sig = nn.Sigmoid()

    def forward(self, x, angles):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print (out.data.size())
        out = torch.cat((out, angles), 1)#new
        out = self.dropout(out)#new
        out = self.classifier(out)
        if (self.num_classes == 1):
            out = self.sig(out)
        return out

    def _make_layers(self, cfg, num_rgb):
        layers = []
        in_channels = num_rgb
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_pretrained(nn.Module):
    def __init__(self, vgg_name, num_classes, num_rgb):
        super(VGG_pretrained, self).__init__()
        model = models.vgg16(pretrained=True).cuda()
        self.features = model.features
        self.num_classes=num_classes
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.linear1 = nn.Linear(2049, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, num_classes)
        self.sig = nn.Sigmoid()

    def forward(self, x, angles):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # print (out.data.size())
        out = torch.cat((out, angles), 1)
        out = self.linear1(out)
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)

        out = self.linear3(out)
        if (self.num_classes == 1):
            out = self.sig(out)
        return out


class VGG_pretrained_norm(nn.Module):
    def __init__(self, vgg_name, num_classes, num_rgb):
        super(VGG_pretrained_norm, self).__init__()
        model = models.vgg16_bn(pretrained=True).cuda()
        self.features = model.features
        self.classifier = nn.Sequential(
            nn.Linear(2049, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, angles):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, angles), 1)
        out = self.classifier(out)
        return out

class VGG_crops(nn.Module):
    def __init__(self, vgg_name, num_classes, num_rgb):
        super(VGG_crops, self).__init__()
        self.features1 = self._make_layers(cfg[vgg_name], num_rgb)
        self.features2 = self._make_layers(cfg[vgg_name], num_rgb)
        self.features3 = self._make_layers(cfg[vgg_name], num_rgb)
        self.num_classes=num_classes
        self.dropout = nn.Dropout(0.2)#new
        self.classifier = nn.Linear(3072 + 1, 512)
        self.classifier2 = nn.Linear(512, num_classes)
        #self.sig = nn.Sigmoid()

    def forward(self, x_75, x_65, x_60, angles):
        out1 = self.features1(x_75)
        out2 = self.features2(x_65)
        out3 = self.features3(x_60)
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        out3 = out3.view(out3.size(0), -1)
        out = torch.cat((out1, out2, out3), 1)
        out = torch.cat((out, angles), 1)#new
        out = self.dropout(out)#new
        out = self.classifier(out)
        out = self.dropout(out)
        out = self.classifier2(out)
        #if (self.num_classes == 1):
        #    out = self.sig(out)
        return out

    def _make_layers(self, cfg, num_rgb):
        layers = []
        in_channels = num_rgb
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)



def vggnetXX_generic(num_classes, num_rgb, type='VGG16'):
    model = VGG(type, num_classes, num_rgb)  # 56
    return model

def vggnetcropsXX_generic(num_classes, num_rgb, type='VGG16'):
    model = VGG_crops(type, num_classes, num_rgb)  # 56
    return model

def vgg_pretrained(num_classes, num_rgb, type='VGG16'):
    model = VGG_pretrained(type, num_classes, num_rgb)  # 56
    return model