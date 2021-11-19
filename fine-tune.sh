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