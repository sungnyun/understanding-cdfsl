### Pre-train
#python ./pretrain_simsiam.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn --pretrain_type 1 --aug_mode base
python ./pretrain_simsiam.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn --pretrain_type 3 --dataset_names CropDisease --aug_mode base
python ./pretrain_simsiam.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn --pretrain_type 4 --dataset_names CropDisease --aug_mode base
python ./pretrain_simsiam.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn --pretrain_type 5 --dataset_names CropDisease --aug_mode base
python ./pretrain_simsiam.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn --pretrain_type 2 --aug_mode base
