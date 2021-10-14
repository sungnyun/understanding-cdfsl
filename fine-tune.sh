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

shot_lst="1 5"

# ###################### All testset #######################
for s in $shot_lst
do
    # Without Re-initialization
    # miniImageNet
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking

    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug --track_bn \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking

    # tieredImageNet
    python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline --track_bn \
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking
    python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline\
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking

    python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline_body --track_bn \
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking
    python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline_body\
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking
     
    ##### With Re-initialization
    # miniImageNet
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --no_tracking
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --no_tracking

    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug --track_bn \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --no_tracking
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --no_tracking

    # tieredImageNet
    python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline --track_bn \
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --no_tracking
    python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline\
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --no_tracking

    python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline_body --track_bn \
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --no_tracking
    python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline_body\
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --no_tracking
done

####################### 80% testset (following STARTUP) #######################
# for s in $shot_lst
# do
#     ##### Without Re-initialization
#     # miniImageNet
#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking
#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking

#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug --track_bn \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking
#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking

#     # tieredImageNet
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline --track_bn \
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline\
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking

#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline_body --track_bn \
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline_body\
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking

#     ##### With Re-initialization
#     # miniImageNet
#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --startup_split --no_tracking
#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --startup_split --no_tracking

#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug --track_bn \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --startup_split --no_tracking
#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline_body --train_aug \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --startup_split --no_tracking

#     # tieredImageNet
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline --track_bn \
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --startup_split --no_tracking
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline\
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --startup_split --no_tracking

#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline_body --track_bn \
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --startup_split --no_tracking
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline_body\
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --partial_reinit --startup_split --no_tracking
# done