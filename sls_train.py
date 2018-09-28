from __future__ import print_function, division

import argparse

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets.folder import default_loader

matplotlib.use('agg')
import time
import os
from model import ft_net, ft_net_dense
from random_erasing import RandomErasing
import json
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils import get_gan_data

#######################################################n###############
# Options
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name', default='plsro_dense12', type=str, help='output model name')
parser.add_argument('--data_dir', default='/home/paul/datasets/viper/pytorch', type=str,
                    help='training dir path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0.8, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name

generated_image_size = 8000
n_clusters = 3
n_classes = 316
generated_images = get_gan_data(generated_size=generated_image_size, n_clusters=n_clusters,
                                generated_dir=os.path.join(data_dir, 'train_new', 'gen_0000'))
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    gid = int(str_id)
    if gid >= 0:
        gpu_ids.append(gid)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])

######################################################################
if opt.use_dense:
    width = 288
    height = 144
    random_width_crop = 256
    random_height_crop = 128
else:
    width = 256
    height = 256
    random_width_crop = 224
    random_height_crop = 224

transform_train_list = [
    # transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
    transforms.Resize((288, 144), interpolation=3),
    transforms.RandomCrop((256, 128)),
    #   transforms.Resize(256,interpolation=3),
    #   transforms.RandomCrop(224,224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(opt.erasing_p)]

# print(transform_train_list)

transform_val_list = [
    transforms.Resize(size=(256, 128), interpolation=3),  # Image.BICUBIC
    # transforms.Resize(256,interpolation=3),
    # transforms.RandomCrop(224,224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}


# read dcgan data
class dcganDataset(Dataset):
    def __init__(self, root, transform=None, targte_transform=None):
        super(dcganDataset, self).__init__()
        self.image_dir = os.path.join(opt.data_dir, root)
        self.samples = []  # train_data   xxx_label_flag_yyy.jpg
        self.img_label = []
        self.img_flag = []
        self.transform = transform
        self.targte_transform = targte_transform
        #   self.class_num=len(os.listdir(self.image_dir))   # the number of the class
        self.train_val = root  # judge whether it is used for training for testing
        if root == 'train_new':
            for folder in os.listdir(self.image_dir):
                fdir = self.image_dir + '/' + folder  # folder gen_0000 means the images are generated images, so their flags are 1
                if folder == 'gen_0000':
                    samples, img_labels, flags = generated_images
                    self.samples = self.samples + samples
                    self.img_label = self.img_label + img_labels
                    self.img_flag = self.img_flag + flags
                else:
                    for files in os.listdir(fdir):
                        temp = folder + '_' + files
                        lbl = int(folder)
                        label_vec = np.zeros(shape=n_classes)
                        label_vec[lbl] = 1
                        self.img_label.append(label_vec)
                        self.img_flag.append(0)
                        self.samples.append(temp)
        else:  # val
            for folder in os.listdir(self.image_dir):
                fdir = self.image_dir + '/' + folder
                for files in os.listdir(fdir):
                    temp = folder + '_' + files
                    lbl = int(folder)
                    label_vec = np.zeros(shape=n_classes)
                    label_vec[lbl] = 1
                    self.img_label.append(label_vec)
                    self.img_flag.append(0)
                    self.samples.append(temp)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        temp = self.samples[idx]  # folder_files
        # print(temp)
        if self.img_flag[idx] == 1:
            foldername = 'gen_0000'
            filename = temp[9:]
        else:
            foldername = temp[:4]
            filename = temp[5:]
        img = default_loader(self.image_dir + '/' + foldername + '/' + filename)
        if self.train_val == 'train_new':
            result = {'img': data_transforms['train'](img), 'label': self.img_label[idx],
                      'flag': self.img_flag[idx]}  # flag=0 for ture data and 0 for generated data
        else:
            result = {'img': data_transforms['val'](img), 'label': self.img_label[idx], 'flag': self.img_flag[idx]}
        return result


class SLSloss(nn.Module):
    def __init__(self):  
        super(SLSloss, self).__init__()


    def forward(self, input, target, flg): 
        if input.dim() > 2:  
            input = input.view(input.size(0), input.size(1), -1)  
            input = input.transpose(1, 2)  
            input = input.contiguous().view(-1, input.size(2))  
        maxRow, _ = torch.max(input.data, 1) 
        maxRow = maxRow.unsqueeze(1)
        input.data = input.data - maxRow
        flg = flg.view(-1, 1)
        flos = F.log_softmax(input)  
        flos = torch.sum(flos, 1) / flos.size(1)  
        logpt = F.log_softmax(input) 
        logpt = torch.mul(logpt, target)
        logpt = torch.sum(logpt, 1, True)
        logpt = logpt.view(-1) 
        flg = flg.view(-1)
        flg = flg.type(torch.cuda.FloatTensor)
        loss = -1 * logpt * (1 - flg) - flos * flg
        return loss.mean()


dataloaders = {}
dataloaders['train'] = DataLoader(dcganDataset('train_new', data_transforms['train']), batch_size=opt.batchsize,
                                  shuffle=True, num_workers=8)
dataloaders['val'] = DataLoader(dcganDataset('val_new', data_transforms['val']), batch_size=opt.batchsize,
                                shuffle=True, num_workers=8)

dataset_sizes = {}
dataset_train_dir = os.path.join(data_dir, 'train_new')
dataset_val_dir = os.path.join(data_dir, 'val_new')

dataset_sizes['val'] = sum(len(os.listdir(os.path.join(dataset_val_dir, i))) for i in os.listdir(dataset_val_dir))
dataset_sizes['train'] = 632 + generated_image_size - dataset_sizes['val']
print(dataset_sizes['train'])
print(dataset_sizes['val'])

use_gpu = torch.cuda.is_available()


y_loss = {}  # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs = data['img']
                labels = data['label']
                flags = data['flag']
                labels = labels.type(torch.cuda.FloatTensor)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    flags = Variable(flags.cuda())
                else:
                    inputs, labels, flags = Variable(inputs), Variable(labels), Variable(flags)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)  # outputs.data  return the index of the biggest value in each row
                loss = criterion(outputs, labels, flags)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item()
                print("Loss {} ".format(loss.item()))

                for temp in range(flags.size()[0]):
                    if flags.data[temp] == 1:
                        preds[temp] = -1

                indices = torch.argmax(labels, dim=1)
                running_corrects += torch.sum(preds == indices.data)
                # print('running_corrects: '+str(running_corrects))

            epoch_loss = running_loss / dataset_sizes[phase]
           
            if phase == 'train':
               
                epoch_acc = running_corrects / (
                        dataset_sizes[phase] - generated_image_size)  
            else:
                epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0 - epoch_acc)
           
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                if epoch >= 40:
                    save_network(model, epoch)
            #    draw_curve(epoch)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    save_network(model, 'best')
    return model

def save_network(network, epoch_label):
    save_filename = 'net_%s.pth'% epoch_label
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda(gpu_ids[0])

if opt.use_dense:
    model = ft_net_dense(n_classes)  
else:
    model = ft_net(n_classes)

if use_gpu:
    model = model.cuda()
criterion = SLSloss()

ignored_params = list(map(id, model.model.fc.parameters())) + list(map(id, model.classifier.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

optimizer_ft = optim.SGD([
    {'params': base_params, 'lr': 0.01},
    {'params': model.model.fc.parameters(), 'lr': 0.05},
    {'params': model.classifier.parameters(), 'lr': 0.05}
], momentum=0.9, weight_decay=5e-4, nesterov=True)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)

dir_name = os.path.join('./model', name)
if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

with open('%s/opts.json' % dir_name, 'w') as fp:
    json.dump(vars(opt), fp, indent=1)

if __name__ == '__main__':
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=130)
