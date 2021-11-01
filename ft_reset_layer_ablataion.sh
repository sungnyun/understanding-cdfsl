### Layerwise Re-Randomization Ablation

declare -a ResNet18_84x84Layers=(
  "4.1.b3 4.1.c3"
  "4.1.b3 4.1.c3 4.1.b2 4.1.c2"
  "4.1.b3 4.1.c3 4.1.b2 4.1.c2 4.1.b1 4.1.c1"
  "4.1.b3 4.1.c3 4.1.b2 4.1.c2 4.1.b1 4.1.c1 4.0.bs 4.0.cs"
  "4.1.b3 4.1.c3 4.1.b2 4.1.c2 4.1.b1 4.1.c1 4.0.bs 4.0.cs 4.0.b3 4.0.c3"
  "4.1.b3 4.1.c3 4.1.b2 4.1.c2 4.1.b1 4.1.c1 4.0.bs 4.0.cs 4.0.b3 4.0.c3 4.0.b2 4.0.c2"
  "4.1.b3 4.1.c3 4.1.b2 4.1.c2 4.1.b1 4.1.c1 4.0.bs 4.0.cs 4.0.b3 4.0.c3 4.0.b2 4.0.c2 4.0.b1 4.0.c1"
  "4.1.b3"
  "4.1.b3 4.1.c3 4.1.b2"
  "4.1.b3 4.1.c3 4.1.b2 4.1.c2 4.1.b1"
)

declare -a ResNet18Layers=(
  "4.1.b2 4.1.c2"
  "4.1.b2 4.1.c2 4.1.b1 4.1.c1"
  "4.1.b2 4.1.c2 4.1.b1 4.1.c1 4.0.bs 4.0.cs"
  "4.1.b2 4.1.c2 4.1.b1 4.1.c1 4.0.bs 4.0.cs 4.0.b2 4.0.c2"
  "4.1.b2 4.1.c2 4.1.b1 4.1.c1 4.0.bs 4.0.cs 4.0.b2 4.0.c2 4.0.b1 4.0.c1"
  "4.1.b2"
  "4.1.b2 4.1.c2 4.1.b1"
)

SHOTS="5"

export CUDA_VISIBLE_DEVICES=3

#DATASET_NAMES="miniImageNet"
#DATASET_NAMES="CropDisease"
#DATASET_NAMES="EuroSAT"
#DATASET_NAMES="ISIC"
#DATASET_NAMES="ChestX"
DATASET_NAMES="miniImageNet CropDisease EuroSAT ISIC ChestX"


#for S in $SHOTS; do
#    # mini
#    NOT IMPLEMENTED YET
#    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
#     --dataset_names "$DATASET_NAMES" --n_shot $S --no_tracking
#done

for S in $SHOTS; do
    # tiered
    for LAYERS in "${ResNet18_84x84Layers[@]}"; do
      ho "Reset START - $S Shot - Tiered - [$DATASET_NAMES] - [$LAYERS]"
      python ./finetune.py --dataset tieredImageNet --model ResNet18_84x84 --method baseline --track_bn \
       --dataset_names $DATASET_NAMES --n_shot $S --no_tracking --reset_layers $LAYERS
      ho "Reset END - $S Shot - Tiered - [$DATASET_NAMES] - [$LAYERS]"
     done
done

for S in $SHOTS; do
    # imagenet
    for LAYERS in "${ResNet18Layers[@]}"; do
      python ./finetune.py --dataset ImageNet --model ResNet18 --method baseline --train_aug --track_bn \
       --dataset_names $DATASET_NAMES --n_shot $S --no_tracking --reset_layers $LAYERS
     done
done
