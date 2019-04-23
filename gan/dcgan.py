# coding: utf-8

#############################################
# DCGAN
#############################################

# Imports

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image

DATA_CSV_PATH = 'dataset_gan/all.csv'
IMAGE_PATH = 'images'
OUTPUT_PATH = 'output_gan'

CUDA = 0
RANDOM_SEED = 123

assert torch.cuda.is_available()
DEVICE = torch.device("cuda:%d" % 0)

if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
LOGFILE = os.path.join(OUTPUT_PATH, 'training.log')

# Logging

header = []

header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % DEVICE)
header.append('Random Seed: %s' % RANDOM_SEED)
header.append('Output Path: %s' % OUTPUT_PATH)

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()

##########################
# SETTINGS
##########################

# Hyperparameters
learning_rate_gen = 0.0005
learning_rate_dsc = 0.0005
num_epochs = 200

# Architecture
BATCH_SIZE = 128

nz = 100
ngf = 64
ndf = 64
nc = 3
ngpu = 1
device = torch.device("cuda:%d" % 0)


###################
# Dataset
###################


class BookCoverDataset(Dataset):
    """Custom Dataset for loading AFAD face images"""

    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_paths = df['filename']
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return self.img_paths.shape[0]


custom_transform = transforms.Compose([transforms.Resize(80),
                                       transforms.RandomCrop((64, 64)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = BookCoverDataset(csv_path=DATA_CSV_PATH,
                           img_dir=IMAGE_PATH,
                           transform=custom_transform)

dataloader = DataLoader(dataset=dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=4)


##########################
# MODEL
##########################


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


###########################################
# Initialize Cost, Model, and Optimizer
###########################################

def loss_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits) * levels
                       + (F.logsigmoid(logits) - logits) * (1 - levels)) * imp,
                      dim=1))
    return torch.mean(val)


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

netG = Generator(ngpu).to(device)
netG.apply(weights_init)
print(netG)

netD = Discriminator(ngpu).to(device)
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss()

fixed_noise = torch.randn(BATCH_SIZE, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = torch.optim.Adam(netD.parameters(), lr=learning_rate_dsc)
optimizerG = torch.optim.Adam(netG.parameters(), lr=learning_rate_gen)


def compute_mae_mse_acc(model, data_loader, device):
    mae, mse, correct_pred, num_examples = 0, 0, 0, 0
    for i, (features, targets, levels) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets) ** 2)
        assert predicted_labels.size() == targets.size()
        correct_pred += (predicted_labels == targets).sum()
    mae = mae.float() / num_examples
    mse = mse.float() / num_examples
    acc = correct_pred.float() / num_examples * 100
    return mae, mse, acc


start_time = time.time()
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data.to(device)
        batch_size = real_cpu.size(0)
        label = torch.full((batch_size,), real_label, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        if i % 50 == 0:
            s = '[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f' % (
                epoch, num_epochs, i, len(dataloader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

            vutils.save_image(real_cpu,
                              '%s/real_samples_epoch_%03d.png' % (OUTPUT_PATH, epoch),
                              normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(),
                              '%s/fake_samples_epoch_%03d.png' % (OUTPUT_PATH, epoch),
                              normalize=True)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time) / 60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

# do checkpointing
torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (OUTPUT_PATH, epoch))
torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (OUTPUT_PATH, epoch))
