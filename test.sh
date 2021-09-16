python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug
python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug

# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --freeze_backbone --train_aug
# python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --freeze_backbone --train_aug

# python ./train.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug
# python ./train.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug
