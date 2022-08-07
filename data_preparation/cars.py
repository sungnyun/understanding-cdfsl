import os
import shutil
from collections import defaultdict

import scipy.io
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
METADATA_PATH = os.path.join(ROOT, "input", "cars_annos.mat")
SOURCE_DIR = os.path.join(ROOT, "input", "car_ims")
TARGET_DIR = os.path.join(ROOT, "output", "cars_cdfsl")

if not os.path.isfile(METADATA_PATH):
    raise Exception("Could not find metadata file at `{}`".format(METADATA_PATH))
if not os.path.isdir(SOURCE_DIR):
    raise Exception("could not find image folder at `{}`".format(SOURCE_DIR))

metadata = scipy.io.loadmat(METADATA_PATH)
metadata = metadata["annotations"][0]

paths_by_class = defaultdict(list)
total = len(metadata)
for m in metadata:
    path, _, _, _, _, cls, test = m
    cls = cls.item()
    path = path.item()
    path = os.path.basename(path)
    paths_by_class[str(cls)].append(path)

print("Copying images to `{}`...".format(TARGET_DIR))
with tqdm(total=total) as pbar:
    for cls, paths in paths_by_class.items():
        for path in paths:
            source_path = os.path.join(SOURCE_DIR, path)
            target_path = os.path.join(TARGET_DIR, cls, path)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(source_path, target_path)
            pbar.update()
print("Complete")
