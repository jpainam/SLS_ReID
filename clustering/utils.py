import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import sys
import keras
import tensorflow as tf
from keras import backend as K
import glob
import os
import pickle
from keras.metrics import top_k_categorical_accuracy
from keras.utils.np_utils import to_categorical


def random_crop(img, random_crop_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size, random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


# seems to achieve the same effect on an axis without losing rotated label support.
def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def load_data(dataset, gan_path='/home/fstu1/generated/market1501'):
    x_train = np.load(os.path.join(dataset, 'x_train.npy'))
    y_train = pickle.load(open(os.path.join(dataset, 'y_train.pkl'), 'rb'))
    x_gan = np.load(os.path.join(dataset, 'x_gan.npy'))
    y_gan = pickle.load(open(os.path.join(dataset, 'y_gan.pkl'), 'rb'))
    assert x_train.shape[0] == len(y_train)
    assert x_gan.shape[0] == len(y_gan)
    gan_list = os.listdir(gan_path)
    assert x_gan.shape[0] == len(gan_list)
    print(x_gan.shape)
    x_train = np.concatenate((x_train, x_gan), axis=0)
    y_train = dict(y_train, **y_gan)
    return x_train, y_train


def load_y_train_data(dataset):
    y_train = pickle.load(open(os.path.join(dataset, 'y_train.pkl'), 'rb'))
    keys = list(y_train.keys())
    y_train = [y_train[key] for key in keys]
    return y_train


def load_y_data(dataset):
    y_train = pickle.load(open(os.path.join(dataset, 'y_train.pkl'), 'rb'))
    keys = list(y_train.keys())
    y_train = [y_train[key] for key in keys]
    y_gan = pickle.load(open(os.path.join(dataset, 'y_gan.pkl'), 'rb'))
    keys = list(y_gan.keys())
    y_gan = [y_gan[key] / np.sum(y_gan[key], axis=0) for key in keys]
    assert np.array(y_gan).shape[1] == np.array(y_train).shape[1]
    y_train = np.concatenate((y_train, y_gan), axis=0)
    return y_train


def load_test_data(dataset, n_classes=751):
    test_list = glob.glob(os.path.join(dataset, '*.jpg'))
    sorted(test_list)
    x_test, y_test = [], []
    total = len(test_list)
    lbl = -1
    old_idx = -np.inf
    i = 0
    for idx, img in enumerate(test_list):
        progress(idx, total, status='Loading test set')
        fname = os.path.basename(img)
        splt = fname.split('_')
        if int(splt[0]) == -1:
            continue
        else:
            i = i + 1
            if int(splt[0]) != old_idx:
                lbl = lbl + 1
                old_idx = int(splt[0])
                if lbl == n_classes:
                    break
            img = image.load_img(img, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            x_test.append(img[0])
            y_test.append(lbl)

    x_test = np.array(x_test)
    y_test = to_categorical(y_test, n_classes)
    assert x_test.shape[0] == y_test.shape[0]
    assert y_test.shape[1] == n_classes
    return x_test, y_test


def load_train_data(dataset):
    x_train = np.load(os.path.join(dataset, 'x_train.npy'))
    y_train = pickle.load(open(os.path.join(dataset, 'y_train.pkl'), 'rb'))
    return x_train, y_train


def load_gan_data(dataset):
    x_gan = np.load(os.path.join(dataset, 'x_gan.npy'))
    y_gan = pickle.load(open(os.path.join(dataset, 'y_gan.pkl'), 'rb'))
    assert x_gan.shape[0] == len(y_gan)
    return x_gan, y_gan


def partial_lrso_loss(y_true, y_pred):
    k_size = np.sum(y_true, axis=0)
    assert k_size != 0
    y_true = y_true / k_size
    return keras.losses.categorical_crossentropy(y_true, y_pred)


def top_1_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


def top_10_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=10)


def top_20_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=20)


def safe_mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def extract_feature(dir_path, net):
    features = []
    infos = []
    i = 0
    total = len(os.listdir(dir_path))
    for image_name in sorted(os.listdir(dir_path)):
        arr = image_name.split('_')
        # Avoid Thumbs.db found hidden in some image directory such as Market1501
        if len(arr) > 2:
            person = int(arr[0])
            camera = int(arr[1][1])
            image_path = os.path.join(dir_path, image_name)
            img = image.load_img(image_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feature = net.predict(x)
            features.append(np.squeeze(feature))
            infos.append((person, camera))
            progress(i, total=total, status='Feature extraction')
            i = i + 1
    return features, infos


def smooth_labels(y, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

     # Arguments
         y: matrix of one-hot row-vector labels to be smoothed
         smooth_factor: label smoothing factor (between 0 and 1)

     # Returns
         A matrix of smoothed labels.
     '''
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[0]
    else:
        raise Exception('Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def write(path, content):
    with open(path, "a+") as dst_file:
        dst_file.write(content)


if __name__ == '__main__':
    load_y_train_data('/home/fstu1/datasets')
