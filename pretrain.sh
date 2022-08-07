export CUDA_VISIBLE_DEVICES=0


SOURCES=("miniImageNet" "tieredImageNet" "ImageNet")
SOURCE=${SOURCES[2]}

TARGETS=("CropDisease" "ISIC" "EuroSAT" "ChestX" "places" "cub" "plantae" "cars")
TARGET=${TARGETS[0]}

# BACKBONE=resnet10  # for mini
BACKBONE=resnet18  # for tiered and full imagenet


# Source SL (note, we adapt the torchvision pre-trained model for ResNet18 + ImageNet. Do not use this command as-is.)
python pretrain.py --ls --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "base" --tag "default"

# Target SSL
python pretrain.py --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "simclr" --tag "default"
python pretrain.py --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "byol" --tag "default"

# MSL (Source SL + Target SSL)
python pretrain.py --ls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "simclr" --tag "gamma78"
python pretrain.py --ls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "byol" --tag "gamma78"

# Two-Stage SSL (Source SL -> Target SSL)
python pretrain.py --pls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "simclr" --tag "default" --previous_tag "default"
python pretrain.py --pls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "byol" --tag "default" --previous_tag "default"

# Two-Stage MSL (Source SL -> Source SL + Target SSL)
python pretrain.py --pls --ls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "simclr" --tag "gamma78" --previous_tag "default"
python pretrain.py --pls --ls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "byol" --tag "gamma78" --previous_tag "default"