### SLS ReID
![](./images/cmc_curve.jpg)
### Datasets
- Download [Market1501 Dataset](http://www.liangzheng.org/Project/project_reid.html)
- Download [DukeMTMC-reID Dataset](https://github.com/layumi/DukeMTMC-reID_evaluation)
- Download [CUHK03 Dataset](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- Download [VIPeR Dataset](https://vision.soe.ucsc.edu/node/178)

### Training
- Train the baseline
```bash
python train.py --use_dense
```
- Train SLS_ReID
```bash
python sls_train.py --use_dense
```

### Pre-trained models

| Dataset | Dense Baseline | ResNet Baseline |Dense SLS_ReID | ResNet SLS_ReID |
| --- | --- | --- | --- | --- | 
| Market-1501 | [market/dense.pth](https://drive.google.com/open?id=18_rb1c3m8YohQVv0ecWL1sgSKiaRNLur) | [market/resnet.pth]() | [cuhk03/dense_slsreid.pth]() | [market/resnet_slsreid.pth]() | 
| CUHK03 | [cuhk03/dense.pth]() | [cuhk03/resnet.pth](https://drive.google.com/open?id=1F53kR_L2bk4ePUWpzRPM8lus9l1JTbkI) | [cuhk03/dense_slsreid.pth]() | [cuhk03/resnet_slsreid.pth](https://drive.google.com/open?id=1D6cEuUmA9KcZ38d715XBdtT5kagSLZtc) | 
| VIPeR | [viper/dense.pth](https://drive.google.com/open?id=15MToMvqenWW7XmygATfm0WdjXIk6kU2J) | [viper/resnet.pth]() | [viper/dense_slsreid.pth]() | [viper/resnet_slsreid.pth]() |  
| DukeMTMCReID | [duke/dense.pth]() | [duke/resnet.pth]() | [duke/dense_slsreid.pth](https://drive.google.com/open?id=139ngCD9PuHIQvqV4X5bZ-4_Y8G-ZQ03S)| [duke/resnet_slsreid.pth]() |

### Testing

```bash
python test_cuhk03.py --model_path ./cuhk03/model.pth --use_dense
python eval_cuhk03.py
----------
python test_viper.py --model_path ./viper/model.pth --use_dense
%Add --re_rank to get re-ranking with k-reciprocal encoding
----------
python test.py --model_path ./market/model.pth --use_dense
python evaluate.py
python evaluate_rerank.py 
%Add --multi for multi-query evaluation
```

### Currents results

| Dataset | Rank 1 | Rank 5 | Rank 10 | Rank 20 | mAP |
| --- | --- | --- | --- | --- | --- |
| `CUHK03-Dense Baseline` | 67.92% | 90.94% | 95.35% | 97.78% | 78.10% |
| `CUHK03-Dense SLS_ReID` | 84.32% | 97.13% | 98.92% | 99.63% | 89.92% |
| `CUHK03-ResNet Baseline` |75.02% | 95.08% | 97.92% | 99.11% | 83.87% |
| `CUHK03-ResNet SLS_ReID` | 90.99% | 98.24% | 99.25% | 99.74% | 94.18% |
| `VIPeR-Dense Baseline` | 63.45% | 72.78% | 79.11% | 86.23% | - |
| `VIPeR-Dense SLS_ReID` | 67.41% | 81.01% | 88.61% | 93.51% | - |
| `Market-1501-Dense Baseline` |
| `Market-1501-Dense SLS_ReID` |
| `Market-1501-ResNet Baseline` |
| `Market-1501-ResNet SLS_ReID` |
| `DukeMTMC-ReID-Dense Baseline` |
| `DukeMTMC-ReID-Dense SLS_ReID` |
| `DukeMTMC-ReID-ResNet Baseline` |
| `DukeMTMC-ReID-ResNet SLS_ReID` |




