# python ./finetune.py --dataset miniImageNet --model ResNet10 \
#  --method baseline_body --train_aug --reinit_blocks 1 2 3 4
# python ./finetune.py --dataset miniImageNet --model ResNet10 \
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
