import os
import glob

class Dataset():
    def __init__(self, root='/home/paul/datasets', dataset='market1501'):
        self.dataset = dataset
        self.root = root

    def train_path(self):
        if self.dataset == 'market1501' or self.dataset == 'duke':
            return os.path.join(self.root, self.dataset, 'bounding_box_train')
        elif self.dataset == 'cuhk03':
            return os.path.join(self.root, self.dataset, 'bounding_box_train')
        elif self.dataset == 'viper':
            return os.path.join(self.root, self.dataset, 'bounding_box_train')
        else:
            raise ValueError('Unknown train set for %s' % self.dataset)

    def test_path(self):
        if self.dataset == 'market1501' or self.dataset == 'duke':
            return os.path.join(self.root, self.dataset, 'bounding_box_test')
        elif self.dataset == 'cuhk03' or self.dataset == 'viper':
            return os.path.join(self.root, self.dataset, 'bounding_box_test')
        else:
            raise ValueError('Unknown test set for %s' % self.dataset)

    def gallery_path(self):
        return self.testset()

    def query_path(self):
        if self.dataset == 'market1501' or self.dataset == 'duke':
            return os.path.join(self.root, self.dataset, 'query')
        elif self.dataset == 'cuhk03' or self.dataset == 'viper':
            return os.path.join(self.root, self.dataset, 'query')
        else:
            raise ValueError('Unknown query set for %s' % self.dataset)

    def gan_path(self):
        return os.path.join('/home/paul/generated', self.dataset)

    def dataset_path(self):
        return os.path.join(self.root, self.dataset)

    def n_classe(self):
        if self.dataset == 'market1501':
            return 751
        elif self.dataset == 'duke':
            return 702
        elif self.dataset == 'cuhk03':
            return 767
        elif self.dataset == 'viper':
            return 316
        else:
            raise ValueError('Unknown n_classe set for %s' % self.dataset)

    def root_path(self):
        return self.root

    def gt_set(self):
        if self.dataset == 'market1501':
            return os.path.join(self.root, self.dataset, 'gt_bbox')
        else:
            raise ValueError('Unknown hand-drawn bounding boxes for %s' % self.dataset)

    def train_list(self):
        if self.dataset == 'market1501' or self.dataset == 'duke' or self.dataset == 'cuhk03':
            train_list = os.path.join(self.root, self.dataset, 'train.list')
        elif self.dataset == 'viper':
            train_list = os.path.join(self.root, self.dataset, 'train.list')
        else:
            raise ValueError('Unknown train bounding boxes for %s' % self.dataset)
        if not os.path.exists(train_list):
            raise FileNotFoundError('%s not found' % train_list)
        return train_list

    def cluster_path(self):
        if self.dataset == 'market1501' or self.dataset == 'duke' or \
                self.dataset == 'cuhk03' or self.dataset == 'viper':
            return os.path.join('/home/paul', 'clustering', self.dataset)

        else:
            raise ValueError('Unknown cluster path for %s' % self.dataset)

    def n_training_set(self):
        if self.dataset == 'market1501':
            data_list = glob.glob(os.path.join(self.train_path(), '*.jpg'))
            n = len(data_list)
            assert n == 12936
        elif self.dataset == 'duke':
            n = 16522
        else:
            raise ValueError("Unknow training set size for %s" % self.dataset)
        return n

    def n_gan_set(self):
        if self.dataset == 'market1501':
            data_list = glob.glob(os.path.join(self.gan_path(), '*.jpg'))
            n = len(data_list)
        else:
            raise ValueError('Unknow generated set size for %s' % self.dataset)
        return n

    def test_num(self):
        if self.dataset == 'market1501':
            return 19732
        elif self.dataset == 'duke':
            return 17661
        elif self.dataset == 'cuhk03':
            return 6751
        elif self.dataset == 'viper':
            return 316
        else:
            raise ValueError('Unknown test num for % dataset' % self.dataset)

    def query_num(self):
        if self.dataset == 'market1501':
            return 3368
        elif self.dataset == 'duke':
            return 2228
        elif self.dataset == 'cuhk03':
            return 6751
        elif self.dataset == 'viper':
            return 316
        else:
            raise ValueError('Unknown query num for % dataset' % self.dataset)