# Debiased Cross Contrastive Quantization for Unsupervised Image Retrieval

## 1. Preparation

- numpy 1.17.0
- pandas 1.1.5
- pytorch 1.7.1
- torchvision 0.8.2
- pillow 8.2.0
- opencv-python 4.5.5.64
- tqdm 4.62.2

## 2. Download the image datasets 

The `data/` folder is the collection of data splits for Flickr25K and NUS-WIDE datasets. The raw images of Flickr25K and NUS-WIDE datasets should be downloaded additionally and arranged in `datasets/Flickr25K/` and `datasets/NUS-WIDE/` respectively. Here we provide copies of these image datasets, you can download them via  [Baidu Wangpan (password: **1111**)](https://pan.baidu.com/s/1mc-ZLuNvHy3BpX94pjA4Lg?pwd=1111). For experiments on CIFAR-10 dataset, you can use the option `--download_cifar10` when running `main.py`.

## 3. Train and then evaluate

If you want to train and evaluate a 64-bit DCCQ model on  CIFAR10, you can do:

```bash
cd CIFAR10
python main.py  
```

If you want to train and evaluate a 64-bit DCCQ model on  FLICKR25K, you can do:

```bash
cd Flickr25K
python main.py 
```

If you want to train and evaluate a 64-bit DCCQ model on NUS-WIDE, you can do:

```bash
cd NUSWIDE
python main.py 
```

## 4. Acknowledgements

Our code is based on the implementation of  [SPQ](https://github.com/youngkyunJang/SPQ) and [MeCoQ](https://github.com/gimpong/AAAI22-MeCoQ).



 

 
