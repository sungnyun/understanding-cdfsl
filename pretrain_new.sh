## Type 1
python ./pretrain_new.py --ls --source_dataset miniImageNet --backbone resnet10 --model simclr --epochs 200

## Type 2
python ./pretrain_new.py --us --source_dataset miniImageNet --backbone resnet10 --model simclr --epochs 200

## Type 3
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./pretrain_new.py --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
done

## Type 4
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./pretrain_new.py --ls --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
done

# Type 5
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./pretrain_new.py --us --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
done

## PLS pretrain (note 1, should use --model base for ls pretrain) (note 2, should use same --tag for ls and pls)
python ./pretrain_new.py --ls --source_dataset miniImageNet --backbone resnet10 --model base

## Type 6
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./pretrain_new.py --pls --ls --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
done

# Type 7
for TARGET in "miniImageNet_test" "CropDisease" "EuroSAT" "ISIC" "ChestX"; do
  python ./pretrain_new.py --pls --us --ut --source_dataset miniImageNet --target_dataset $TARGET --backbone resnet10 --model simclr
done
