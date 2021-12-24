## Type 1
python ./pretrain_new.py --ls --source_dataset miniImageNet --backbone resnet10 --model simclr --epochs 200

## Type 2
python ./pretrain_new.py --us --source_dataset miniImageNet --backbone resnet10 --model simclr --epochs 200

## Type 3
for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./pretrain_new.py --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
done

## Type 4
for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./pretrain_new.py --ls --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
done

# Type 5
for TARGET in "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./pretrain_new.py --us --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
done
