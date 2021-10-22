### Fine-tune
# Arguments
# --startup_split: use 80% of data, as in Startup (default=False)
# --reinit_blocks 1 2 3 4
# --partial_reinit: [C0, BN0, C1, BN1, C2, BN2, shortcut, BNshortcut]
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

    # tieredImageNet
    python ./finetune.py --dataset tieredImageNet --model ResNet18 --method baseline --track_bn \
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking
    python ./finetune.py --dataset tieredImageNet --model ResNet18 --method baseline\
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking

    ##### With Re-initialization
    # miniImageNet
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking \
     --partial_reinit C2 BN2 shortcut BNshortcut
    python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug \
     --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking \
     --partial_reinit C2 BN2 shortcut BNshortcut

    # tieredImageNet
    python ./finetune.py --dataset tieredImageNet --model ResNet18 --method baseline --track_bn \
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking \
     --partial_reinit C3 BN3
    python ./finetune.py --dataset tieredImageNet --model ResNet18 --method baseline\
     --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --no_tracking \
     --partial_reinit C3 BN3
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

#     # tieredImageNet
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline --track_bn \
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline\
#      --dataset_names tieredImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking

#     ##### With Re-initialization
#     # miniImageNet
#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug --track_bn \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking \
#      --partial_reinit C2 BN2 shortcut BNshortcut
#     python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --train_aug \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking \
#      --partial_reinit C2 BN2 shortcut BNshortcut
#
#     # tieredImageNet
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline --track_bn \
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking \
#      --partial_reinit C2 BN2 shortcut BNshortcut
#     python ./finetune.py --dataset tieredImageNet --model ResNet12 --method baseline\
#      --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX --n_shot $s --startup_split --no_tracking \
#      --partial_reinit C2 BN2 shortcut BNshortcut
# done
