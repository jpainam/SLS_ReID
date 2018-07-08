import os
import os.path as osp
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import pairwise_distances
from torch.autograd import Variable
from torchvision import datasets, transforms
import scipy.io
from model import ft_net, ft_net_dense
from re_ranking import re_ranking
'''
Command
#python test_viper.py --use_dense --which_epoch 59 --name viper_dense

'''
parser = argparse.ArgumentParser(description='Testing arguments')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--re_rank', action='store_true', help='use reranking')
parser.add_argument('--name', default='resnet', type=str, help='save model path')
parser.add_argument('--which_epoch', default='59', type=str, help='0,1,2,3...or last')
opt = parser.parse_args()
model_path = opt.name

data_transforms = transforms.Compose([
    transforms.Resize((288, 144), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


data_dir = '/home/paul/datasets/viper/pytorch'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                  ['gallery', 'query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=8,
                                              shuffle=False, num_workers=4) for x in
               ['gallery', 'query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()


def load_network(network):
    save_path = os.path.join('./viper', model_path, 'net_%s.pth' % opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def extract_feature(model, dataloaders):
    features = torch.FloatTensor()
    count = 0
    labels = torch.LongTensor()
    for data in dataloaders:
        img, label = data
        labels = torch.cat((labels, label), 0)
        n, c, h, w = img.size()
        count += n
        #print(count)
        if opt.use_dense:
            ff = torch.FloatTensor(n, 1024).zero_()
        else:
            ff = torch.FloatTensor(n, 2048).zero_()
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            input_img = Variable(img.cuda())
            outputs = model(input_img)
            f = outputs.data.cpu()
            ff = ff + f
        # norm feature
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, f), 0)
    return features.numpy(), labels.numpy()


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def _cmc_core(D, G, P, k):
    order = np.argsort(D, axis=0)
    res = np.zeros((k, D.shape[1]))
    for i in range(k):
        for j in range(D.shape[1]):
            if G[order[i][j]] == P[j]:
                res[i][j] += 1
    return (res.sum(axis=1) * 1.0 / D.shape[1]).cumsum()


def _re_assign_labels(q_pids, g_pids):
    # Reassign the labels to make them sequentially numbered from zero
    unique_labels = np.unique(np.r_[q_pids, g_pids])
    labels_map = {l: i for i, l in enumerate(unique_labels)}
    q_pids = np.asarray([labels_map[l] for l in q_pids])
    g_pids = np.asarray([labels_map[l] for l in g_pids])
    return q_pids, g_pids


def test(query_feature, query_label, gallery_feature, gallery_label, method='cosine'):
    D = pairwise_distances(gallery_feature, query_feature, metric=method, n_jobs=-2)
    query_label, gallery_label = _re_assign_labels(query_label, gallery_label)
    gallery_labels_set = np.unique(gallery_label)

    if opt.re_rank:
        q_g_dist = np.dot(query_feature, np.transpose(gallery_feature))
        q_q_dist = np.dot(query_feature, np.transpose(query_feature))
        g_g_dist = np.dot(gallery_feature, np.transpose(gallery_feature))

    for label in query_label:
        if label not in gallery_labels_set:
            print('Probe-id is out of Gallery-id sets.')

    Times = 100
    k = 110

    res = np.zeros(k)

    gallery_labels_map = [[] for i in range(gallery_labels_set.size)]
    for i, g in enumerate(gallery_label):
        gallery_labels_map[g].append(i)

    for __ in range(Times):
        # Randomly select one gallery sample per label selected
        newD = np.zeros((gallery_labels_set.size, query_label.size))
        print(newD.shape)
        for i, g in enumerate(gallery_labels_set):
            j = np.random.choice(gallery_labels_map[g])
            newD[i, :] = D[j, :]
        # Compute CMC
        print(newD.shape)

        res += _cmc_core(newD, gallery_labels_set, query_label, k)

    if opt.re_rank:
        newD = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        newD = np.transpose(newD)
        res += _cmc_core(newD, gallery_labels_set, query_label, k)
    res /= Times
    return res


if __name__ == '__main__':
    # Load Collected data Trained model
    print('-------test-----------')
    if opt.use_dense:
        model_structure = ft_net_dense(316)
    else:
        model_structure = ft_net(316)

    model = load_network(model_structure)

    # Remove the final fc layer and classifier layer
    model.model.fc = nn.Sequential()
    model.classifier = nn.Sequential()
    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    # Extract feature
    gallery_feature, gallery_label = extract_feature(model, dataloaders['gallery'])
    query_feature, query_label = extract_feature(model, dataloaders['query'])
    res = test(query_feature, query_label, gallery_feature, gallery_label)
    scipy.io.savemat('./viper/'+model_path.split('_')[1]+'.mat', {'CMC': res})
    for topk in [1, 5, 10, 20]:
        print("{:8}{:8.2%}".format('rank-' + str(topk), res[topk - 1]))
