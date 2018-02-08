
from torchvision import transforms, models

model = models.vgg16_bn(pretrained=True).cuda()
for child in model.features.parameters():
    print(child)

print(model.classifier)
