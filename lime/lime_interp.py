import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

from torchvision import transforms
from PIL import Image

import lime
from lime import lime_image

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift

# Hyperparameters
learning_rate = 0.0005
num_epochs = 200

# Architecture
NUM_CLASSES = 5
BATCH_SIZE = 256
GRAYSCALE = False


def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)

    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0),
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m / torch.max(m)
    return imp


class BookCoverDataset(Dataset):
    """Custom Dataset for loading AFAD face images"""

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_paths = df['filename']
        self.y = df['label'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        levels = [1] * label + [0] * (NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)

        return img, label, levels

    def __len__(self):
        return self.y.shape[0]


custom_transform = transforms.Compose([transforms.Resize(120),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])

train_dataset = BookCoverDataset(csv_path='train.csv',
                                 img_dir='images',
                                 transform=custom_transform)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes - 1).float())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** .5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return probas.numpy()


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock,
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


def loss_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits) * levels
                       + (F.logsigmoid(logits) - logits) * (1 - levels)) * imp,
                      dim=1))
    return torch.mean(val)


def cls(img):
    img = torch.tensor(img)
    ret = model(img)
    return ret


def seg(img):
    transposed = np.transpose(img, (1, 2, 0))
    quickshifted = quickshift(transposed,
                              kernel_size=4,
                              max_dist=200,
                              ratio=0.2,
                              random_seed=123)
    ret = np.tile(quickshifted, (3, 1, 1))
    return ret


RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

model = resnet34(NUM_CLASSES, GRAYSCALE)
model.load_state_dict(torch.load('model.pt'))

# Read Images
x = train_dataset.__getitem__(42878)[0]
y = train_dataset.__getitem__(283)[0]
z = train_dataset.__getitem__(7386)[0]
k = train_dataset.__getitem__(11767)[0]
s = train_dataset.__getitem__(6585)[0]

explainer = lime_image.LimeImageExplainer()
explainer_y = lime_image.LimeImageExplainer()
explainer_z = lime_image.LimeImageExplainer()
explainer_k = lime_image.LimeImageExplainer()
explainer_s = lime_image.LimeImageExplainer()

model.eval()
with torch.set_grad_enabled(False):
    explanation = explainer.explain_instance(x.numpy(), cls,
                                             top_labels=5, hide_color=0, num_samples=1000,
                                             segmentation_fn=seg)

model.eval()
with torch.set_grad_enabled(False):
    explanation_y = explainer_y.explain_instance(y.numpy(), cls,
                                                 top_labels=5, hide_color=0, num_samples=1000,
                                                 segmentation_fn=seg)

model.eval()
with torch.set_grad_enabled(False):
    explanation_z = explainer_z.explain_instance(z.numpy(), cls,
                                                 top_labels=5, hide_color=0, num_samples=1000,
                                                 segmentation_fn=seg)

model.eval()
with torch.set_grad_enabled(False):
    explanation_z = explainer_z.explain_instance(z.numpy(), cls,
                                                 top_labels=5, hide_color=0, num_samples=1000,
                                                 segmentation_fn=seg)

model.eval()
with torch.set_grad_enabled(False):
    explanation_k = explainer_k.explain_instance(k.numpy(), cls,
                                                 top_labels=5, hide_color=0, num_samples=1000,
                                                 segmentation_fn=seg)

model.eval()
with torch.set_grad_enabled(False):
    explanation_s = explainer_s.explain_instance(s.numpy(), cls,
                                                 top_labels=5, hide_color=0, num_samples=1000,
                                                 segmentation_fn=seg)

temp, mask = explanation.get_image_and_mask(0, positive_only=True, num_features=5,
                                            hide_rest=True)

temp_y, mask_y = explanation_y.get_image_and_mask(0, positive_only=True, num_features=5,
                                                  hide_rest=True)

temp_z, mask_z = explanation_z.get_image_and_mask(0, positive_only=True, num_features=5,
                                                  hide_rest=True)

temp_k, mask_k = explanation_k.get_image_and_mask(0, positive_only=True, num_features=5,
                                                  hide_rest=True)

temp_s, mask_s = explanation_s.get_image_and_mask(0, positive_only=True, num_features=5,
                                                  hide_rest=True)

plt.imshow(np.transpose(temp_y, (1, 2, 0)))
plt.imshow(np.transpose(y, (1, 2, 0)))
plt.imshow(np.transpose(z, (1, 2, 0)))
plt.imshow(np.transpose(temp_z, (1, 2, 0)))
plt.imshow(np.transpose(k, (1, 2, 0)))
plt.imshow(np.transpose(temp_k, (1, 2, 0)))
plt.imshow(np.transpose(s, (1, 2, 0)))
plt.imshow(np.transpose(temp_s, (1, 2, 0)))
