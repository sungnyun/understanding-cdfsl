# Unlabeled training (SimCLR)

# Transfer + RR + SimCLR
# python ./train_unlabeled.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
# --save_freq 100 --stop_epoch 1000 --use_base_classes --aug_mode strong --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX

# python ./train_unlabeled.py --dataset tieredImageNet --model ResNet18 --method baseline --track_bn \
# --save_freq 100 --stop_epoch 1000 --use_base_classes --aug_mode strong --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX

# python ./train_unlabeled.py --dataset ImageNet --model çResNet18 --method baseline --track_bn \
# --save_freq 100 --stop_epoch 1000 --use_base_classes --aug_mode strong --dataset_names CropDisease EuroSAT ISIC ChestX

# python ./train_unlabeled.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
# --save_freq 100 --stop_epoch 1000 --aug_mode strong --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX

# python ./train_unlabeled.py --dataset tieredImageNet --model ResNet18 --method baseline --track_bn \
# --save_freq 100 --stop_epoch 1000 --aug_mode strong --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX

# python ./train_unlabeled.py --dataset ImageNet --model ResNet18 --method baseline --track_bn \
# --save_freq 100 --stop_epoch 1000 --aug_mode strong --dataset_names CropDisease EuroSAT ISIC ChestX


python ./train_unlabeled.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
--save_freq 100 --stop_epoch 1000 --use_base_classes_as_unlabeled --aug_mode strong --dataset_names miniImageNet


# SimCLR
# python ./train_unlabeled.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
# --save_freq 100 --stop_epoch 1000 --aug_mode strong --no_base_pretraining --no_rerand --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX

# Transfer+SimCLR
# python ./train_unlabeled.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
# --save_freq 100 --stop_epoch 1000 --aug_mode strong --no_rerand--dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX