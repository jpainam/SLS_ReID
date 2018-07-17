import torch
from utils import AverageMeter
import time
import numpy as np

import os
import data_manager
from model import ft_net, ft_net_dense
from torch.utils.data import DataLoader
from torchvision import transforms

import scipy.io
from dataset_loader import ImageDataset


import argparse
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--model_path', default='resnet', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet')
parser.add_argument('--n_classe', default=1367, help='n classes')
parser.add_argument('--dataset', default='/home/paul/datasets', type=str, help='Path to the dataset')

opt = parser.parse_args()
n_classe = opt.n_classe

def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20]):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        end = time.time()
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu: imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, 32))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()
    # Save to Matlab for check
    result = {'distmat':distmat, 'q_pids': q_pids, 'g_pids':g_pids,
              'q_camids':q_camids, 'g_camids': g_camids,
              'query_feature': qf.numpy(), 'gallery_feature': gf.numpy()}
    print(qf.numpy())
    print(gf.numpy())
    scipy.io.savemat('./result.mat', result)



def load_network(network):
    save_path = os.path.join(opt.model_path)
    network.load_state_dict(torch.load(save_path))
    return network

# --------

use_dense = opt.use_dense
if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    data_transforms = transforms.Compose([
        transforms.Resize((288, 144), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    if opt.use_dense:
        model_structure = ft_net_dense(n_classe)
    else:
        model_structure = ft_net(n_classe)
    model = load_network(model_structure)
    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    dataset = data_manager.init_img_dataset(
        root=opt.dataset, name='cuhk03', split_id=0, cuhk03_classic_split=True)

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=data_transforms),
        batch_size=32, shuffle=False, num_workers=4, drop_last=False,
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=data_transforms),
        batch_size=32, shuffle=False, num_workers=4, drop_last=False,
    )
    test(model, queryloader, galleryloader, use_gpu)