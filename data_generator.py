from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
import argparse
from torch.utils.data import Dataset, DataLoader
import os
from random_erasing import RandomErasing


class DataGenerator(Dataset):
    def __init__(self, root, data_transform=None, image_dir=None, target_transform=None):
        super(DataGenerator, self).__init__()
        assert image_dir is not None
        self.image_dir = image_dir
        self.samples = []  # train_data   xxx_label_flag_yyy.jpg
        self.img_label = []
        self.img_flag = []
        self.data_transform = data_transform
        self.target_transform = target_transform
        #   self.class_num=len(os.listdir(self.image_dir))   # the number of the class
        self.train_val = root  # judge whether it is used for training for testing
        if root == 'train_new':
            for folder in os.listdir(self.image_dir):
                fdir = self.image_dir + '/' + folder  # folder gen_0000 means the images are generated images, so their flags are 1
                if folder == 'gen_0000':
                    for files in os.listdir(fdir):
                        temp = folder + '_' + files
                        self.img_label.append(int(folder[-4:]))
                        self.img_flag.append(1)
                        self.samples.append(temp)
                else:
                    for files in os.listdir(fdir):
                        temp = folder + '_' + files
                        self.img_label.append(int(folder))
                        self.img_flag.append(0)
                        self.samples.append(temp)
        else:  # val
            for folder in os.listdir(self.image_dir):
                fdir = self.image_dir + '/' + folder
                for files in os.listdir(fdir):
                    temp = folder + '_' + files
                    self.img_label.append(int(folder))
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
        # print(self.image_dir + '/' + foldername + '/' + filename)
        img = default_loader(self.image_dir + '/' + foldername + '/' + filename)

        if self.train_val == 'train_new':
            result = {'img': self.data_transform(img), 'label': self.img_label[idx],
                      'flag': self.img_flag[idx]}  # flag=0 for ture data and 0 for generated data
        else:
            result = {'img': self.data_transform(img), 'label': self.img_label[idx], 'flag': self.img_flag[idx]}
        return result
