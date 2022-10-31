# [Official] Understanding Cross-Domain Few-Shot Learning Based on Domain Similarity and Few-shot Difficulty

This repo contains the implementation of our [paper](https://arxiv.org/abs/2202.01339) accepted at NeurIPS 2022.

## Abstract
Cross-domain few-shot learning (CD-FSL) has drawn increasing attention for handling large differences between the source and target domains--an important concern in real-world scenarios. To overcome these large differences, recent works have considered exploiting small-scale unlabeled data from the target domain during the pre-training stage. This data enables self-supervised pre-training on the target domain, in addition to supervised pre-training on the source domain. In this paper, we empirically investigate which pre-training is preferred based on domain similarity and few-shot difficulty of the target domain. We discover that the performance gain of self-supervised pre-training over supervised pre-training becomes large when the target domain is dissimilar to the source domain, or the target domain itself has low few-shot difficulty. We further design two pre-training schemes, mixed-supervised and two-stage learning, that improve performance. In this light, we present six findings for CD-FSL, which are supported by extensive experiments and analyses on three source and eight target benchmark datasets with varying levels of domain similarity and few-shot difficulty.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Data Preparation](#data-preparation)
  * [BSCD-FSL](#bscd-fsl)
  * [Cars](#cars)
  * [CUB](#cub (caltech-ucsd birds-200-2011))
  * [Places](#places)
  * [Plantae](#plantae)
* [Usage](#usage)
* [Model Checkpoints](#model-checkpoints)
* [License](#license)
* [Attribution](#attribution)

## Prerequisites

Our code works on `torch>=1.8`. Install the required Python packages via

```sh
pip install -r requirements.txt
```

## Data Preparation

Prepare and place all dataset folders in `/data/cdfsl/`. You may specify custom locations in `configs.py`.

### BSCD-FSL

Refer to the original BSCD-FSL [repository](https://github.com/IBM/cdfsl-benchmark).

The dataset folders should be organized in `/data/cdfsl/` as follows:

```
CropDiseases/train
├── Apple___Apple_scab
│  ├── 00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG
│  ├── 0208f4eb-45a4-4399-904e-989ac2c6257c___FREC_Scab 3037.JPG

EuroSAT
├── AnnualCrop
│  ├── AnnualCrop_1.jpg
│  ├── AnnualCrop_2.jpg

ISIC
├── ATTRIBUTION.txt
├── ISIC2018_Task3_Training_GroundTruth.csv
├── ISIC2018_Task3_Training_LesionGroupings.csv
├── ISIC_0024306.jpg
├── ISIC_0024307.jpg

chestX/images
├── 00000001_000.png
├── 00000001_001.png
```

Note: `chestX/images/` should contain **all** images from the ChestX dataset (the dataset archive provided online will
typically split these images across multiple folders).

### Cars

https://ai.stanford.edu/~jkrause/cars/car_dataset.html

We use all images from both training and test sets for CD-FSL experiments.
For convenience, we pre-process the data such that each image goes into its respective class folder.

1. Download [`car_ims.tgz`](http://ai.stanford.edu/~jkrause/car196/car_ims.tgz) (the tar of all images) and [`cars_annos.mat`](http://ai.stanford.edu/~jkrause/car196/cars_annos.mat) (all bounding boxes and labels for both training and test).
2. Copy `cars_annos.mat` and unzip `car_ims.tgz` into `./data_preparation/input/`. The directory should contain the following:
```
data_preparation/input
├── cars_annos.mat
├── car_ims
│  ├── 000001.jpg
│  ├── 000002.jpg
```
3. Run `./data_preparation/cars.py`, to generate the cars dataset folder at `./data_preparation/output/cars_cdfsl/`.
4. Move the `cars_cdfsl` directory to `/data/cdfsl/`. You may specify a custom location in `configs.py`.

### <a name="cub"></a> CUB (Caltech-UCSD Birds-200-2011)

http://www.vision.caltech.edu/datasets/cub_200_2011/

We use all images for CD-FSL experiments.

1. Download [`CUB_200_2011.tgz`](https://data.caltech.edu/records/20098).
2. Unzip the archive and copy the *enclosed* `CUB_200_2011/` folder to `/data/cdfsl/`. You may specify a custom location in `configs.py`. The directory should contain the following:
```
CUB_200_2011/
├── attributes/
├── bounding_boxes.txt
├── classes.txt
├── image_class_labels.txt
├── images/
├── images.txt
├── parts/
├── README
└── train_test_split.txt
```

### <a name="places"></a> Places (Places 205)

http://places.csail.mit.edu/user/

Due to the size of the original dataset, we only use a subset of the training set for CD-FSL experiments.
We use 27,440 images from 16 classes. Please refer to the paper for details or refer to the subset sampling code at
`data_prepratation/places_plantae_subset_sampler.ipynb`.

1. Download [places365standard_easyformat.tar](http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar).
2. Unzip the archive into `./data_preparation/input/`. The directory should contain the following:
```
data_preparation/input/
├── places365_standard/
│  ├── train/
│  │  ├── airfield/
│  │  ├── airplane_cabin/
│  │  ├── aiport_terminal/
```
3. Run `./data_preparation/places.py` to generate the places dataset folder at `./data_preparation/outuput/places_cdfsl/`.
4. Move the `places_cdfsl` directory to `/data/cdfsl/`. You may specify a custom location in `configs.py`.

### <a name="plantae"></a> Plantae (from iNaturalist 2018)

https://github.com/visipedia/inat_comp/tree/master/2018#Data

Due to the size of the original dataset, we only use a subset of the training set (of the Plantae super category) for CD-FSL experiments.
We use 26,650 images from 69 classes. Please refer to the paper for details or refer to the subset sampling code at
`data_prepratation/places_plantae_subset_sampler.ipynb`

1. Download [`train_val2018.tar.gz`](https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz) (~120GB).
2. Unzip the archive and copy the enclosed `Plantae/` folder (~43GB) to `./data_preparation/input/`. The directory should contain the following:
```
data_preparation/input
└── Plantae/
   ├── 5221/
   ├── 5222/
   ├── 5223/
```
3. Run `./data_preparation/plantae.py` to generate the plantae dataset folder at `./data_preparation/outuput/plantae_cdfsl`.
4. Move the `plantae_cdfsl` directory to `/data/cdfsl/`. You may specify a custom location in `configs.py`.

## Usage

The main training scripts are `pretrain.py` and `finetune.py`. Refer to `pretrain.sh` and `finetune.sh` on example
usages for the main results in our paper, e.g., SL, SSL, MSL, two-stage SSL and two-stage MSL.
To see all CLI arguments, refer to `io_utils.py`.

## Model Checkpoints

| Backbone  | Pretraining  | Augmentation | Model Checkpoints |
| :-------- | :------------: |:---------: |:--------------:|
| ResNet10  | miniImageNet (SL) | default (strong) | [google drive](https://drive.google.com/file/d/1J4weUMgMhdjYe0sbPBNavaf5D7aRkAog/view?usp=sharing) |
| ResNet10  | miniImageNet (SL) | base | [google drive](https://drive.google.com/file/d/11HSAg85vlS67sVsEgd-RYgksnlX61WOj/view?usp=sharing) |
| ResNet18  | tieredImageNet (SL) | default (strong) | [google drive](https://drive.google.com/file/d/1hRbE5VwDvgsKV6E7okOgqJaNOVsSfitP/view?usp=share_link) |
| ResNet18  | tieredImageNet (SL) | base | [google drive](https://drive.google.com/file/d/1-UOzOG-NkqRFXng-Zu3RAbuMhZJZfEhN/view?usp=share_link) |
| ResNet18  | ImageNet (SL) | base | [torchvision](https://pytorch.org/vision/stable/models.html) |

## License

Distributed under the MIT License.

## Attribution

Below, we provide the licenses, attribution, citations, and URL of the datasets considered in our paper (if applicable).

- **ImageNet** is available at https://image-net.org/. miniImageNet and tieredImageNet are subsets of ImageNet.
  1. *Deng, Jia, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. "Imagenet: A large-scale hierarchical image database." In 2009 IEEE conference on computer vision and pattern recognition, pp. 248-255. Ieee, 2009.*
- **CropDisease** refers to the Plant Disease dataset on Kaggle, licensed under  GPL 2. It is available at https://www.kaggle.com/saroz014/plant-disease/.
- **EuroSAT** is available at https://github.com/phelber/eurosat.
  1. *Helber, Patrick, Benjamin Bischke, Andreas Dengel, and Damian Borth. "Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12, no. 7 (2019): 2217-2226.*
  2. *Helber, Patrick, Benjamin Bischke, Andreas Dengel, and Damian Borth. "Introducing eurosat: A novel dataset and deep learning benchmark for land use and land cover classification." In IGARSS 2018-2018 IEEE international geoscience and remote sensing symposium, pp. 204-207. IEEE, 2018.*
- **ISIC** refers to the ISIC 2018 Challenge Task 3 dataset. It is available at https://challenge.isic-archive.com.
  1. *Codella, Noel, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba et al. "Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration (isic)." arXiv preprint arXiv:1902.03368 (2019).*
  2. *Tschandl, Philipp, Cliff Rosendahl, and Harald Kittler. "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions." Scientific data 5, no. 1 (2018): 1-9.*
- **ChestX** refers to the Chest-Xray8 provided by the NIH Clinical Center. The dataset is available at https://nihcc.app.box.com/v/ChestXray-NIHCC.
  1. *Wang, Xiaosong, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M. Summers. "Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2097-2106. 2017.*
- **Places** is licensed under CC BY. It is available at http://places.csail.mit.edu/user.
  1. *Zhou, Bolei, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. "Places: A 10 million image database for scene recognition." IEEE transactions on pattern analysis and machine intelligence 40, no. 6 (2017): 1452-1464.*
- **Plantae** refers to the Plantae super category of the iNaturalist 2018 dataset. It is available at https://github.com/visipedia/inat_comp/tree/master/2017.
  1. *Van Horn, Grant, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. "The inaturalist species classification and detection dataset." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8769-8778. 2018.*
- **CUB** refers to the Caltech-UCSD Birds-200-2011 dataset, licensed under CC0 (public domain). It is available at https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images.
  1. *Wah, Catherine, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. "The caltech-ucsd birds-200-2011 dataset." (2011).*
- **Cars** is available at https://ai.stanford.edu/~jkrause/cars/car_dataset.html.
  1. *Krause, Jonathan, Michael Stark, Jia Deng, and Li Fei-Fei. "3d object representations for fine-grained categorization." In Proceedings of the IEEE international conference on computer vision workshops, pp. 554-561. 2013.*
  
## Contact
* Jaehoon Oh: jhoon.oh@kaist.ac.kr
* Sungnyun Kim: ksn4397@kaist.ac.kr
* Namgyu Ho: itsnamgyu@kaist.ac.kr
