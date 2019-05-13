import os
import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import argparse
import utils
import glob
import pickle
from dataset import Dataset


dataset = Dataset(dataset='viper')
N_CLUSTER = 3


def generate_labels_for_gan():
    image_labels = {}
    f = open('/home/paul/datasets/viper/train.list', 'r')
    old_lbl = -1
    # Get a dictionnary of current ids in train set
    for line in f:
        line = line.strip()
        img, lbl = line.split()
        lbl = int(lbl)
        if lbl != old_lbl:
            splt = img.split('_')
            image_labels[splt[0]] = int(lbl)
            old_lbl = lbl
    f.close()
    for n_cluster in range(N_CLUSTER):
        cluster_path = os.path.join(dataset.cluster_path(), 'cluster_%s' % n_cluster)
        cluster_labels = {}
        cluster_imgs = glob.glob(os.path.join(cluster_path, '*.jpg'))
        cluster_imgs = sorted(cluster_imgs)
        for img in cluster_imgs:
            img = os.path.basename(img)
            splt = img.split('_')
            try:
                cluster_labels[splt[0]] += 1
            except KeyError:
                cluster_labels[splt[0]] = 1
        # build the idx
        f = open(os.path.join(dataset.cluster_path(), 'gan%s.list' % n_cluster), 'w')
        for i in cluster_labels:
            print(i)
            #if cluster_labels[i] > 4:
            f.write("%s\n" % image_labels[i])
        f.close()
        print(image_labels[i])
        # build the idx


def load_gan(gan_path, n_gan_images):
    return get_gan_data(n_gan_images)

def get_gan_data(n_gan_images):
    images = dict()
    labels = []
    # Get the labels for each gan set and save in array labels
    for i in range(N_CLUSTER):
        f = open(os.path.join(dataset.cluster_path(), 'gan%s.list' % i), 'r')
        tmp_labels = np.zeros(shape=dataset.n_classe(), dtype=np.int32)
        for line in f:
            lbl = line.strip()
            tmp_labels[int(lbl)] = 1
        f.close()
        labels.append(tmp_labels)
    labels = np.array(labels)
    n_gan = int(np.floor(n_gan_images / N_CLUSTER + 1))
    data_list = None
    for i in range(N_CLUSTER):
        gan_list = glob.glob(os.path.join(dataset.gan_path(), 'gan_%s*.jpg' % i))
        gan_list = gan_list[:n_gan]
        if data_list is None:
            data_list = gan_list
        else:
            data_list = np.concatenate((data_list, gan_list), axis=0)

    data_list = np.unique(data_list)
    np.random.shuffle(data_list)
    data_list = data_list[:n_gan_images]
    assert len(data_list) == n_gan_images

    for i, filename in enumerate(data_list):
        img_name = os.path.basename(filename)
        lbl = int(img_name.split('_')[1])
        try:
            images[str(lbl)].append(img_name)
        except KeyError:
            images[str(lbl)] = list()
            images[str(lbl)].append(img_name)

    labels = np.array(labels)

    assert np.sum([len(images[i]) for i in images]) == n_gan_images
    print(images)
    assert len(images) == labels.shape[0]
    assert labels.shape[1] == dataset.n_classe()
    return images, labels


if __name__ == '__main__':
    generate_labels_for_gan()
    #images, labels = get_gan_data(2000)
    #pickle.dump((images, labels), open('/home/fstu1/datasets/gan_data', 'wb'), protocol=2)
    print()
    print('Generated GAN data saved')
