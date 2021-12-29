### Fine-tune
# Arguments
# --startup_split: use 80% of data, as in Startup (default=False)
# --dataset_names miniImageNet CropDisease EuroSAT ISIC ChestX
# --finetune_parts: ['head', 'body', 'full']
# --no_tracking

##### Fine-tuning from scratch #####
python ./finetune.py --dataset none --model ResNet10 --method baseline --track_bn \
--finetune_parts head --dataset_names CropDisease EuroSAT ISIC ChestX --n_shot 1

python ./finetune.py --dataset none --model ResNet10 --method baseline --track_bn \
--finetune_parts body --dataset_names CropDisease EuroSAT ISIC ChestX --n_shot 1

python ./finetune.py --dataset none --model ResNet10 --method baseline --track_bn \
--finetune_parts full --dataset_names CropDisease EuroSAT ISIC ChestX --n_shot 1

python ./finetune.py --dataset none --model ResNet10 --method baseline --track_bn \
--finetune_parts head --dataset_names CropDisease EuroSAT ISIC ChestX --n_shot 5

python ./finetune.py --dataset none --model ResNet10 --method baseline --track_bn \
--finetune_parts body --dataset_names CropDisease EuroSAT ISIC ChestX --n_shot 5

python ./finetune.py --dataset none --model ResNet10 --method baseline --track_bn \
--finetune_parts full --dataset_names CropDisease EuroSAT ISIC ChestX --n_shot 5


python ./finetune.py --dataset none --model ResNet10 --method baseline --track_bn \
--finetune_parts haed --dataset_names miniImageNet --n_shot 5

python ./finetune.py --dataset none --model ResNet10 --method baseline --track_bn \
--finetune_parts full --dataset_names miniImageNet --n_shot 5

# NIL-testing
python ./nil_testing.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 3 --aug_mode strong --n_shot 5 --dataset_names ISIC --startup_split


##### Fine-tuning from pre-trained models #####
python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode base --finetune_parts head --n_shot 1 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode base --finetune_parts body --n_shot 1 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode base --finetune_parts full --n_shot 1 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode base --finetune_parts head --n_shot 5 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode base --finetune_parts body --n_shot 5 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode base --finetune_parts full --n_shot 5 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode strong --finetune_parts head --n_shot 1 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode strong --finetune_parts body --n_shot 1 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode strong --finetune_parts full --n_shot 1 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode strong --finetune_parts head --n_shot 5 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode strong --finetune_parts body --n_shot 5 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--pretrain_type 1 --aug_mode strong --finetune_parts full --n_shot 5 \
--dataset_names CropDisease EuroSAT ISIC ChestX --no_tracking --startup_split

python ./finetune_fusion.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--fusion_method concat --finetune_parts head --n_shot 5 \
--dataset_names CropDisease --no_tracking --startup_split

python ./finetune_fusion.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--fusion_method adaptive_weight_vectorwise --finetune_parts head --n_shot 5 \
--dataset_names CropDisease --no_tracking --startup_split

python ./finetune_fusion.py --dataset miniImageNet --model ResNet10 --method baseline --track_bn \
--fusion_method adaptive_weight_elementwise --finetune_parts head --n_shot 5 \
--dataset_names CropDisease --no_tracking --startup_split