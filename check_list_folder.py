import os
import numpy as np

image_dir = '/home/paul/datasets/market1501/pytorch/train_new'
img_label = []
image_pytorch = {}
for folder in os.listdir(image_dir):
    fdir = image_dir + '/' + folder
    if folder != 'gen_0000':
        for files in os.listdir(fdir):
            img_name = os.path.basename(files)
            lbl = int(folder)
            image_pytorch[img_name] = lbl



print(len(image_pytorch))

f = open('/home/paul/datasets/market1501/train.list', 'r')
image_list = {}
for line in f:
    line = line.strip()
    img, lbl = line.split()
    image_list[img] = int(lbl)

print(len(image_list))
for img in image_pytorch:
    if image_pytorch[img] != image_list[img]:
        print('%s doesnt have same id %s' % (img, image_pytorch[img]))
        exit(0)
    else:
        print(image_pytorch[img])

print('Finish')
from sklearn.cluster import KMeans