### Fine-tune
# Arguments
# --startup_split: use 80% of data, as in Startup (default=False)
# --reinit_blocks 1 2 3 4
# --partial_reinit: reinit {Conv2, BN2, ShortCutConv, ShortCutBN} from last block (default=False)
# --mv_init: reinit all blocks with normal dist while maintaining mean-var
# --method startup_both_body
# --method startup_student_body
# --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX
# --no_tracking

python ./train_unlabeled.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
--save_freq 100 --stop_epoch 1000 --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX

python ./train_unlabeled.py --dataset tieredImageNet --model ResNet12 --method baseline --track_bn \
--save_freq 100 --stop_epoch 1000 --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX