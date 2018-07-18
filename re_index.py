# -*- coding: utf-8 -*-
# change name of the folder(e.g.  0002,0007,0010,0011...  to 0,1,2,3)
import os
from shutil import copyfile

original_path = '/home/paul/datasets/market1501/pytorch'


# copy folder tree from source to destination
def copyfolder(src, dst):
    files = os.listdir(src)
    if not os.path.isdir(dst):
        os.mkdir(dst)
    for tt in files:
        copyfile(src + '/' + tt, dst + '/' + tt)


new_folders = ['train_new', 'val_new']
old_folders = ['train_all', 'val']

for train, data in zip(new_folders, old_folders):
    train_save_path = os.path.join(original_path, train)
    if not os.path.exists(train_save_path):
        os.mkdir(train_save_path)

    data_path = os.path.join(original_path, data)
    if not os.path.isdir(train_save_path):
        os.mkdir(train_save_path)

    reid_index = 0
    folders = os.listdir(data_path)
    folders = sorted(folders)

    for foldernames in folders:
        copyfolder(data_path + '/' + foldernames, train_save_path + '/' + str(reid_index).zfill(4))
        reid_index = reid_index + 1
