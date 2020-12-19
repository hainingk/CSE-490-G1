import os

BASE_PATH = 'C:\\Users\\Kyle\\Downloads\\'
os.chdir(BASE_PATH)
DATA_PATH = BASE_PATH + 'csVOC\\'

import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
import torch.optim as optim
import sys
import time

class_values = [0, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39,
                43, 47, 50, 54, 58, 62, 66, 70, 74, 78]  # 1 is removed

# VOC class indices are stored in this tensor at specific locations. The
# index they are stored at represents a transformation of their color value.
# The color value 1 is not represented here.
class_indices = torch.FloatTensor([0, 0, 0, 1, 0,     0, 0, 2, 0, 0,
                                   0, 3, 0, 0, 0,     4, 0, 0, 0, 5,
                                   0, 0, 0, 6, 0,     0, 0, 7, 0, 0,
                                   0, 8, 0, 0, 0,     9, 0, 0, 0, 10,
                                   0, 0, 0, 11, 0,    0, 0, 12, 0, 0,
                                   13, 0, 0, 0, 14,   0, 0, 0, 15, 0,
                                   0, 0, 16, 0, 0,    0, 17, 0, 0, 0,
                                   18, 0, 0, 0, 19,   0, 0, 0, 20, 0])

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # images are around 3 * 375 * 500. 21 channels to count 20 classes + background.
        self.conv1 = nn.Conv2d(3, 21, 5, stride=4, padding=2)  # 21 * 94 * 125
        self.conv2 = nn.Conv2d(21, 32, 5, stride=4, padding=2)  # 32 * 24 * 32
        self.conv3 = nn.Conv2d(32, 64, 5, stride=4, padding=2)  # 64 * 6 * 8
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, stride=4, padding=2, output_padding=3)  # 32 * 24 * 32
        self.deconv2 = nn.ConvTranspose2d(32, 21, 5, stride=4, padding=2, output_padding=(1,0))  # 21 * 94 * 125
        self.deconv3 = nn.ConvTranspose2d(21, 21, 5, stride=4, padding=2, output_padding=(2,3))  # 21 * 375 * 500

    def adjust_tensor(self, x, x_new):
        width_dif = x.shape[2] - x_new.shape[2]
        x_new = F.pad(x_new, (0, 0, 0, width_dif))
        height_dif = x.shape[3] - x_new.shape[3]
        x_new = F.pad(x_new, (0, height_dif, 0, 0))
        return x_new

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.adjust_tensor(x2, self.deconv1(x3)) + x2)
        x5 = F.relu(self.adjust_tensor(x1, self.deconv2(x4)) + x1)
        x6 = self.adjust_tensor(x, self.deconv3(x5))
        return x6

    def loss(self, prediction, label, reduction='mean'):
        # prediction ~ (1, 21, 375, 500)
        prediction = torch.flatten(prediction.squeeze(), 1)  # (21, 187500)
        prediction = torch.transpose(prediction, 0, 1)  # (187500, 21)

        # need to convert label from color value to class index
        label = label.squeeze()
        shape = label.shape
        label = 1000 * label.flatten()
        label = torch.clamp(label, 0, 79)  # otherwise color val 1 will be 1000. Note this means boundaries are treated as background.
        label_indices = class_indices[torch.Tensor.long(label)]
        loss = F.cross_entropy(prediction, torch.Tensor.long(label_indices), reduction=reduction)
        return loss, torch.reshape(label_indices, shape)


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss, _ = model.loss(output, label)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        return np.mean(losses)

def test(model, device, test_loader, log_interval=None):
    model.eval()
    test_loss = 0
    accuracy = 0
    images = []

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss_on, label_indices = model.loss(output, label, reduction='sum')
            test_loss += test_loss_on.item()
            output = output.squeeze()  # 21 * 500 * 375
            output = output.argmax(dim=0)
            correct_mask = output.eq(label_indices)
            num_correct = correct_mask.sum().item()
            total_pixels = torch.numel(label_indices)
            accuracy += num_correct / total_pixels
            if batch_idx == 52 or batch_idx == 200 or batch_idx == 202:
                images.append((output, label_indices))
            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(data), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * accuracy / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy, len(test_loader.dataset), test_accuracy))
    return test_loss, test_accuracy, images


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

# set download to True first time around
data_train = datasets.VOCSegmentation(DATA_PATH, year='2007', \
        image_set='train', download=False, transform=transform_test, \
        target_transform=transform_test)
data_test = datasets.VOCSegmentation(DATA_PATH, year='2007', \
        image_set='val', download=False, transform=transform_test, \
        target_transform=transform_test)  # technically the val set, but we'll use it as test


BATCH_SIZE = 1  # do not change!
TEST_BATCH_SIZE = 1
EPOCHS = 80
LEARNING_RATE = 0.01
#MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
PRINT_INTERVAL = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using', device)
import multiprocessing
print('num cpus:', multiprocessing.cpu_count())
kwargs = {'num_workers': multiprocessing.cpu_count(), 'pin_memory': True} \
        if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(data_train, batch_size=BATCH_SIZE,
                                           shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(data_test, batch_size=BATCH_SIZE,
                                          shuffle=False, **kwargs)
model = Unet().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
train_losses = []
test_losses = []


for epoch in range(0, EPOCHS + 1):
    train_loss = train(model, device, train_loader, optimizer, epoch, PRINT_INTERVAL)
    test_loss, test_accuracy, images = test(model, device, test_loader)
    train_losses.append((epoch, train_loss))
    test_losses.append((epoch, test_loss))

    if (epoch <= 5 or epoch % 20 == 0):
        for image in images:
            pred = image[0]  # 21 * 500 * 375, max number is 20
            pred1 = pred * 20 % 256
            pred2 = pred * 53 % 256
            pred3 = pred * 107 % 256
            pred = torch.stack([pred1, pred2, pred3], dim=2)
            plt.imshow(torch.Tensor.long(pred))
            plt.show()

            # label_indices = image[1]
            # label_indices1 = label_indices * 20 % 256
            # label_indices2 = label_indices * 53 % 256
            # label_indices3 = label_indices * 107 % 256
            # label_indices = torch.stack([label_indices1, label_indices2, label_indices3], dim=2)
            # plt.imshow(torch.Tensor.long(label_indices))
            # plt.show()
