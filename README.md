# Understanding Cross-Domain Few-Shot Learning: An Experimental Study

This repo contains the implementation of the paper under review, **Understanding Cross-Domain Few-Shot Learning: An Experimental Study**.

## Table of Contents

* [Prerequisites](#prerequisites)
* [Data Setup](#data-setup)
* [Usage](#usage)
  * [Pretraining](#pretraining)
  * [Finetuning](#finetuning)
* [License](#license)

## Prerequisites

Our code works on `torch>=1.8`. Install the required Python packages via

```sh
pip install -r requirements.txt
```

## Data Setup

Make sure that that the dataset directories are correctly specified in `configs.py`, and that your dataset directories contain the correct files. Each directory should contain the files as such:

```sh
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

Note: `chestX/images` should contain **all** images from the ChestX dataset (the dataset archive provided online will typically store these images across multiple folders).

## Usage

The main training files are `pretrain_new.py` and `finetune_new.py`. To see all CLI arguments, refer to `io_utils.py`.

### 1. Pretraining <a name="pretraining"></a>

To pretrain the model in a **supervised learning (SL)** with miniImageNet dataset, run the following command.
```sh
python ./pretrain_new.py --ls --source_dataset miniImageNet --backbone resnet10 --model base --tag [TAG]
```
`[TAG]` is an optional tag for distinguishing each experiment. The default name is `default`.

To change the above command to a **self-supervised manner (SSL)**, use the following command.    
(Caution: You have to specify the target dataset and model, such as simclr or byol.)
```sh
python ./pretrain_new.py --ut --source_dataset miniImageNet --target_dataset [TARGET] --backbone resnet10 --model [MODEL] --tag [TAG]
```

Also, `--ls --ut` will make the training **mixed-supervised learning (MSL)** setting, while adding the `--pls` option will make the **two-stage training**. For example, `--pls --ut` indicates two-stage SSL (SL -> SSL). In the case of two-stage training, the SL-pretrained model should exist in prior, and `--pls_tag` should match with the `[TAG]` you used during the SL pretraining.

The output files, including model checkpoints and training parameters and history, are saved under the directory `./logs/output/[SOURCE]/[EXP_NAME]/[TARGET]`.

### 2. Finetuning <a name="finetuning"></a>

After the pretraining, the finetuning and few-shot evaluation can be conducted via `finetune_new.py`.    
The basic rule is, match the options you used in pretraining (e.g., `--ls`, `--source_dataset`, `--target_dataset`, `--model`, **AND `--tag`**).

For example, to finetune and evaluate in 5-shot with the MSL (SimCLR) pretrained model, run the following command.
```sh
python ./finetune_new.py --ls --ut --source_dataset miniImageNet --target_dataset [TARGET] --backbone resnet10 --model simclr --n_shot 5 --tag [TAG]
```

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
- **Plantae** refers to the Plantae super category of the iNaturalist 2017 dataset. It is available at https://github.com/visipedia/inat_comp/tree/master/2017.
  1. *Van Horn, Grant, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. "The inaturalist species classification and detection dataset." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8769-8778. 2018.*
- **CUB** refers to the Caltech-UCSD Birds-200-2011 dataset, licensed under CC0 (public domain). It is available at https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images.
  1. *Wah, Catherine, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. "The caltech-ucsd birds-200-2011 dataset." (2011).*
- **Cars** is available at https://ai.stanford.edu/~jkrause/cars/car_dataset.html.
  1. *Krause, Jonathan, Michael Stark, Jia Deng, and Li Fei-Fei. "3d object representations for fine-grained categorization." In Proceedings of the IEEE international conference on computer vision workshops, pp. 554-561. 2013.*
