# DCGAN in Tensorflow for SLS

## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12) (Notice that it is not the latest version)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- CUDA 8.0

Add Cuda Path to bashrc first
```bash
export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"
```

We recommend you to install anaconda. Here we write a simple script for you to install the dependence by anaconda.
```python
# install env (especially for old version Tensorflow)
conda env create -f dcgan.yml
# activate env, then you can run code in this env without downgrading the outside Tensorflow.
source activate dcgan
```

### Training
```bash
mkdir data
ln -rs your_dataset_path/market1501/bounding_box_train ./market_train
```
1. Train with the all training set
```bash
python main.py --dataset market_train --train
```
`market_train` is the dir path which contains images. Here I use the [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html) training set. You can change it to your dataset path.
2. Train each clusters with `parent_checkpoint` the saved checkpoint of 1.
```bash
python main.py --dataset cluster_0 --train --parent_checkpoint market_train_64_128_128
python main.py --dataset cluster_1 --train --parent_checkpoint market_train_64_128_128
python main.py --dataset cluster_2 --train --parent_checkpoint market_train_64_128_128
```
### Use pre-trained
Or you can download the pre-trained model and saved them in `./checkpoint`

| Dataset | checkpoint |
| --- | --- |
| Market1501`--parent_checkpoint` | [market_train_64_128_128](#)|
| 4 clusters for market1501 | [cluster_0_64_128_128](#), [cluster_1_64_128_128](#), [cluster_2_64_128_128](#), [cluster_4_64_128_128](#)
| 3 clusters for market1501 | [cluster_0_64_128_128](#), [cluster_1_64_128_128](#), [cluster_2_64_128_128](#) |
| 2 clusters for market1501 | [cluster_0_64_128_128](#), [cluster_1_64_128_128](#) |
| DukeMTMC-reID `--parent_checkpoint` | [duke_train_64_128_128](#) |
| 3 clusters for Duke | [cluster_0_64_128_128](#), [cluster_1_64_128_128](#), [cluster_2_64_128_128](#) |


### Generate samples
```bash
python main.py --dataset cluster_0 --options 5  --output_path generated  --sample_size 12036
python main.py --dataset cluster_1 --options 5  --output_path generated  --sample_size 12036
python main.py --dataset cluster_2 --options 5  --output_path generated  --sample_size 12036
```
It will use the trained model of each cluster and generate `sample_size` images for the following semi-supervised training.

### Download samples
[Download samples - Google Drive](https://drive.google.com/open?id=139vpswFge7S50_ccsnnw4Hy4RRK7a-KY)
