## Type 1
for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts head --split_seed 1
done

## Type 2
for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --us --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts head --split_seed 1
done

## Type 3
for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py ---ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts head --split_seed 1
done

## Type 4
for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --ls --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts head --split_seed 1
done

## Type 5
for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./finetune_new.py --us --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr  --ft_parts head --split_seed 1
done
