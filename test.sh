# New arguments
# --startup_split: use 80% of data, as in Startup (default=False)
# --partial_reinit: reinit {Conv2, BN2, ShortCutConv, ShortCutBN} from last block (default=False)
# --mv_init: reinit all blocks with normal dist while maintaining mean-var
# --method startup_both_body
# --method startup_student_body
# --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX

# python ./finetune.py --dataset miniImageNet --model ResNet10 --startup_split \
#  --method baseline_body --train_aug --reinit_blocks 1 2 3 4
# python ./finetune.py --dataset miniImageNet --model ResNet10 --startup_split \
#  --method baseline --train_aug --reinit_blocks 1 2 3 4

# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --n_shot 5 --freeze_backbone --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --n_shot 5 --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --n_shot 5 --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --n_shot 1 --freeze_backbone --train_aug 
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --n_shot 1 --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --n_shot 1 --train_aug

### Pre-train
python ./train.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug
python ./train.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug
