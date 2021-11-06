# Finetune base - RR - Source + SimCLR - RR - FT

export CUDA_VISIBLE_DEVICES=0
SHOTS="1"

DATASET_NAMES="miniImageNet CropDisease EuroSAT ISIC ChestX"

for S in $SHOTS; do
    ho "DOUBLE RR START - $S Shot - MINI - [$DATASET_NAMES]"
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
     --dataset_names $DATASET_NAMES --n_shot $S --simclr_finetune --simclr_finetune_source --simclr_epochs 1000 --no_tracking \
      --reset_layers "4.b2 4.c2 4.bs 4.cs"
    ho "DOUBLE RR END - $S Shot - MINI - [$DATASET_NAMES]"
done
