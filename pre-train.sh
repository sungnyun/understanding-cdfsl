### Pre-train
# miniImageNet
python ./train.py --dataset miniImageNet --model ResNet10 --method baseline --stop_epoch 400 --train_aug --track_bn
python ./train.py --dataset miniImageNet --model ResNet10 --method baseline --stop_epoch 400 --train_aug

python ./train.py --dataset miniImageNet --model ResNet10 --method baseline_body --stop_epoch 400 --train_aug --track_bn
python ./train.py --dataset miniImageNet --model ResNet10 --method baseline_body --stop_epoch 400 --train_aug

# tieredImageNet
python ./train.py --dataset tieredImageNet --model ResNet12 --method baseline --stop_epoch 90 --track_bn
python ./train.py --dataset tieredImageNet --model ResNet12 --method baseline --stop_epoch 90

python ./train.py --dataset tieredImageNet --model ResNet12 --method baseline_body --stop_epoch 90 --track_bn
python ./train.py --dataset tieredImageNet --model ResNet12 --method baseline_body --stop_epoch 90