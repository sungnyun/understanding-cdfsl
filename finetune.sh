export CUDA_VISIBLE_DEVICES=0


SOURCES=("miniImageNet" "tieredImageNet" "ImageNet")
SOURCE=${SOURCES[0]}

TARGETS=("CropDisease" "ISIC" "EuroSAT" "ChestX" "places" "cub" "plantae" "cars")
TARGET=${TARGETS[0]}

# BACKBONE=resnet10  # for mini
BACKBONE=resnet18  # for tiered and full imagenet

N_SHOT=5


# Source SL
for TARGET in "CropDisease" "ISIC" "EuroSAT" "ChestX" "places" "cub" "plantae" "cars"; do
  python finetune.py --ls --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "base" --tag "default" --n_shot $N_SHOT
done

# Target SSL
for TARGET in "CropDisease" "ISIC" "EuroSAT" "ChestX" "places" "cub" "plantae" "cars"; do
  python finetune.py --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "simclr" --tag "default" --n_shot $N_SHOT
  python finetune.py --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "byol" --tag "default" --n_shot $N_SHOT
done

# MSL (Source SL + Target SSL)
for TARGET in "CropDisease" "ISIC" "EuroSAT" "ChestX" "places" "cub" "plantae" "cars"; do
  python finetune.py --ls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "simclr" --tag "gamma78" --n_shot $N_SHOT
  python finetune.py --ls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "byol" --tag "gamma78" --n_shot $N_SHOT
done

# Two-Stage SSL (Source SL -> Target SSL)
for TARGET in "CropDisease" "ISIC" "EuroSAT" "ChestX" "places" "cub" "plantae" "cars"; do
  python finetune.py --pls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "simclr" --tag "default" --n_shot $N_SHOT
  python finetune.py --pls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "byol" --tag "default" --n_shot $N_SHOT
done

# Two-Stage MSL (Source SL -> Source SL + Target SSL)
for TARGET in "CropDisease" "ISIC" "EuroSAT" "ChestX" "places" "cub" "plantae" "cars"; do
  python finetune.py --pls --ls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "simclr" --tag "gamma78" --n_shot $N_SHOT
  python finetune.py --pls --ls --ut --source_dataset $SOURCE --target_dataset $TARGET --backbone $BACKBONE --model "byol" --tag "gamma78" --n_shot $N_SHOT
done
