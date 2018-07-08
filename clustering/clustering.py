import argparse
import glob
import os
import shutil

# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.models import load_model
from keras.preprocessing import image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabaz_score
import utils
import random
import matplotlib as mpl



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", default='market1501', help="path to the dataset")
ap.add_argument("-t", "--type", type=int, default=1,
                help="Type of the operation")
args = vars(ap.parse_args())

N_CLUSTER = 5
RANGE_CLUSTERS = [2, 3, 4, 5]
# Market 1501
N_CLASSES = 751
# Duke
N_CLASSES = 702
# CUHK03
N_CLASSES = 1367
# VIPeR
N_CLASSES = 316
DATASET = '/home/paul/datasets/viper'
CHECKPOINT = './pretrain/viper_softmax_pretrain.h5'

LIST = os.path.join(DATASET, 'train.list')
TRAIN = os.path.join(DATASET, 'bounding_box_train')


def feature_extraction():
    model = load_model(CHECKPOINT)
    model = Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    model.summary()
    image_labels = []
    resnet_feature_list = []
    image_paths = []
    filenames = glob.glob(os.path.join(TRAIN, '*.jpg'))
    random.shuffle(filenames)
    total = len(filenames)
    i = 0
    print('-----Start feature extraction----')
    for fname in filenames:
        img_path = fname
        filename = os.path.basename(fname)
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        resnet_feature = model.predict(img_data)
        resnet_feature_np = np.array(resnet_feature)
        resnet_feature_list.append(resnet_feature_np.flatten())
        image_labels.append(filename)
        image_paths.append(img_path)
        utils.progress(i, total=total, status='Feature extraction')
        i = i + 1


    # Save the extracted feature
    np.savez('/home/paul/clustering/viper/features.npz', resnet_feature_list, image_paths)
    print('----End of feature extraction----')


def load_features():
    # If a saved feature extraction does not exists, extract feacture first
    #npz_file = '/home/paul/clustering/market1501/features_3_100.npz'
    npz_file = '/home/paul/clustering/viper/features.npz'
    if not os.path.exists(npz_file):
        feature_extraction()
    npz_file = np.load(npz_file)
    resnet_feature_list = npz_file['arr_0']
    image_paths = npz_file['arr_1']
    resnet_feature_list_np = np.array(resnet_feature_list)
    return resnet_feature_list_np, image_paths


# Arg type = 1
def separate_files(n_clusters=3):
    resnet_feature_list_np, image_paths = load_features()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, verbose=1).fit(resnet_feature_list_np)
    labels = np.array(kmeans.labels_)
    total = labels.shape[0]
    for idx, current_cluster in enumerate(labels):
        utils.progress(idx, total=total, status='Clustering file with n_clusters=%s' % n_clusters)
        img_path = image_paths[idx]
        path = os.path.join("/home/paul/clustering/viper", "cluster_%s" % current_cluster)
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(img_path, path)


def plot_clusters(n_cluster=5):
    resnet_feature_list_np, _ = load_features()
    reduced_data = PCA(n_components=2).fit_transform(resnet_feature_list_np)
    #for n_cluster in RANGE_CLUSTERS:
    kmeans = KMeans(n_clusters=n_cluster, random_state=10, verbose=1).fit(reduced_data)
    plt.gca().set_xticks([]) # clear y and x axis
    plt.gca().set_yticks([])
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=1, c=kmeans.labels_,
                s=200, edgecolors='k', cmap='rainbow', marker='.')
    centers = kmeans.cluster_centers_
    # Draw white circles at cluster center
    plt.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
    for i, c in enumerate(centers):
        plt.scatter(c[0], c[1], marker='$%d$' % (i+1), alpha=1, s=50, edgecolor='k')
    utils.simpleaxis(plt.gca())
    plt.ylabel("{} clusters".format(n_cluster))
    plt.savefig('/home/paul/clustering/duke/plot_clusters_%s.eps' % n_cluster)
    plt.savefig('/home/paul/clustering/duke/plot_clusters_%s.png' % n_cluster)
    plt.show()
    #plt.close()


def compare_silhouette():
    resnet_feature_list_np, _ = load_features()
    X = PCA(n_components=2).fit_transform(resnet_feature_list_np)
    bar_values = []
    for n_clusters in RANGE_CLUSTERS:
        kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(X, cluster_labels)
        calinski_score = calinski_harabaz_score(X, cluster_labels)
        print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
        print("For n_clusters =", n_clusters, "The calinsky_harabaz_score is :", calinski_score)
        #bar_values.append(silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)
    '''bar = plt.bar(np.arange(len(bar_values)), bar_values, width=1.0)
    bar[0].set_color('r')
    bar[1].set_color('b')
    bar[2].set_color('g')
    bar[3].set_color('y')
    plt.savefig('/home/paul/clustering/duke/silhouettes.png')
    plt.close()'''


def get_train_list():
    # load data
    image_dict = {}
    data_list = os.path.join(DATASET, 'train.list')
    with open(data_list, 'r') as f:
        for line in f:
            line = line.strip()
            img, lbl = line.split()
            image_dict[img] = lbl
        f.close()
    # create clustering list of ids
    cluster_path = '/home/paul/clustering/viper'
    for idx in range(3):
        cluster_img_path = os.path.join(cluster_path, "cluster_%s" % idx)
        cluster_list = os.path.join(cluster_path, 'cluster.list')
        f = open(cluster_list, 'w')
        filelist = glob.glob(os.path.join(cluster_img_path, '*.jpg'))
        sorted(filelist)
        for file in filelist:
            key = os.path.basename(file)
            f.write('%s\n' % image_dict.get(key))
        f.close()


def main():

    if args['type'] == 1:
        separate_files(n_clusters=3)
    elif args['type'] == 2:
        plot_clusters(n_cluster=3)
    elif args['type'] == 3:
        compare_silhouette()
    elif args['type'] == 4:
        get_train_list()


if __name__ == '__main__':
    main()