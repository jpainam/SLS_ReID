import os
import numpy as np
import glob

n_classes = 316
import json


def get_gan_data(generated_size, n_clusters=3, generated_dir=None):
    assert generated_dir is not None
    labels = []
    # Get the labels for each gan set and save in array labels
    for i in range(n_clusters):
        f = open(os.path.join('/home/paul/clustering', 'gan%s.list' % i), 'r')
        tmp_labels = np.zeros(shape=n_classes, dtype=np.float)
        for line in f:
            lbl = line.strip()
            tmp_labels[int(lbl)] = 1.0
        f.close()
        tmp_labels = tmp_labels / np.sum(tmp_labels)
        labels.append(tmp_labels)
    labels = np.array(labels)
    n_gan = int(np.floor(generated_size / n_clusters + 1))
    data_list = None
    for i in range(n_clusters):
        gan_list = glob.glob(os.path.join(generated_dir, 'gan_%s*.jpg' % i))
        gan_list = gan_list[:n_gan]
        if data_list is None:
            data_list = gan_list
        else:
            data_list = np.concatenate((data_list, gan_list), axis=0)

    data_list = np.unique(data_list)
    np.random.shuffle(data_list)
    assert data_list.shape[0] >= generated_size
    data_list = data_list[:generated_size]
    img_labels = []
    images = []
    flags = []
    for i, filename in enumerate(data_list):
        img_name = os.path.basename(filename)
        lbl = int(img_name.split('_')[1])
        img_labels.append(labels[lbl])
        temp = 'gen_0000' + '_' + img_name
        images.append(temp)
        flags.append(1)
    assert len(images) == generated_size

    assert len(images) == len(img_labels) == len(flags)
    return images, img_labels, flags


class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def write_json(obj, fpath):
    mkdir_if_missing(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))