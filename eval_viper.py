import numpy as np
import scipy.io
from collections import defaultdict

if __name__ == '__main__':
    import json
    import os

    '''print("Number of identities: {}".format(num_pids))
    num_train_pids = num_pids // 2
    old_splits = json.load(open('/home/paul/datasets/viper/oldsplits.json'))
    meta = json.load(open('/home/paul/datasets/viper/oldmeta.json'))
    identities = meta['identities']
    splits = []
    for split in old_splits:
        trainval = split['trainval']
        query = split['query']
        gallery = split['gallery']
        train = []
        for idx in trainval:
            imgs = identities[idx]
            cam_a_img = os.path.join('/home/paul/datasets/viper/images', imgs[0][0])
            cam_b_img = os.path.join('/home/paul/datasets/viper/images', imgs[1][0])
            if not (os.path.exists(cam_a_img) and os.path.exists(cam_b_img)):
                print("%s %s not exists" % (cam_a_img, cam_b_img))
            train.append((cam_a_img, idx, 0))
            train.append((cam_b_img, idx, 1))
        test = []
        for idx in query:
            imgs = identities[idx]
            cam_a_img = os.path.join('/home/paul/datasets/viper/images', imgs[0][0])
            cam_b_img = os.path.join('/home/paul/datasets/viper/images', imgs[1][0])
            if not (os.path.exists(cam_a_img) and os.path.exists(cam_b_img)):
                print("%s %s not exists" % (cam_a_img, cam_b_img))
            test.append((cam_a_img, idx, 0))
            test.append((cam_b_img, idx, 1))
        '''
    import glob
    train_list = sorted(glob.glob('/home/paul/datasets/viper/bounding_box_train/*.jpg'))
    train = []
    test = []
    query = []
    num_train_pids = 316
    num_pids = 632
    for filename in train_list:
        img = os.path.basename(filename)
        pid = int(img.split('_')[0])
        camera = int(img.split('c')[1][0])
        train.append((filename, pid, camera))

    test_list=sorted(glob.glob('/home/paul/datasets/viper/bounding_box_test/*.jpg'))
    for filename in test_list:
        img = os.path.basename(filename)
        pid = int(img.split('_')[0])
        camera = int(img.split('c')[1][0])
        test.append((filename, pid, camera))

    query_list = sorted(glob.glob('/home/paul/datasets/viper/query/*.jpg'))
    for filename in query_list:
        img = os.path.basename(filename)
        pid = int(img.split('_')[0])
        camera = int(img.split('c')[1][0])
        query.append((filename, pid, camera))

    spl = {'train': train, 'query': query, 'gallery': test,
           'num_train_pids': num_train_pids,
           'num_train_imgs': len(train),
           'num_query_pids': num_pids - num_train_pids,
           'num_query_imgs': len(query),
           'num_gallery_pids': num_pids - num_train_pids,
           'num_gallery_imgs': len(test)}
    splits = [spl]

    from utils import write_json
    write_json(splits, '/home/paul/datasets/viper/splits.json')
