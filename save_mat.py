from scipy.io import savemat, loadmat
import numpy as np
dense = loadmat('./viper/dense.mat')
resnet = loadmat('./viper/resnet.mat')
sls_dense = loadmat('./viper/dense8.mat')
sls_resnet = loadmat('./viper/resnet8.mat')

dense = dense['CMC']
resnet = resnet['CMC']
sls_dense = sls_dense['CMC']
sls_resnet = sls_resnet['CMC']

dense = np.squeeze(dense)
resnet = np.squeeze(resnet)
sls_dense = np.squeeze(sls_dense)
sls_resnet = np.squeeze(sls_resnet)
y = 110
dense = dense[:y]*100
resnet = resnet[:y]*100
sls_dense = sls_dense[:y]*100
sls_resnet = sls_resnet[:y]*100

'''dense[99] = 1
resnet[99] = 1
sls_dense[99] = 1
sls_resnet[99] = 1'''

'''dense = np.zeros(y)
resnet = np.zeros(y)
sls_dense = np.zeros(y)
sls_resnet = np.zeros(y)'''


dense = np.expand_dims(dense, axis=0)
dense = np.transpose(dense)
resnet = np.expand_dims(resnet, axis=0)
resnet = np.transpose(resnet)
sls_dense = np.expand_dims(sls_dense, axis=0)
sls_dense = np.transpose(sls_dense)
sls_resnet = np.expand_dims(sls_resnet, axis=0)
sls_resnet = np.transpose(sls_resnet)

mat = {'resnet': resnet, 'dense': dense, 'sls_dense': sls_dense, 'sls_resnet': sls_resnet}

savemat('./viper/viper.mat', mat)