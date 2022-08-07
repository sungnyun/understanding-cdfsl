import json
import os
import shutil

from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
SUBSET_PATH = 'places_cdfsl_subset_16_class_1715_sample_seed_0.json'
SOURCE_DIR = os.path.join(ROOT, "input", "places365_standard/train")
TARGET_DIR = os.path.join(ROOT, "output", "places_cdfsl")

if not os.path.isfile(SUBSET_PATH):
    raise Exception("Could not find subset file at `{}`".format(SUBSET_PATH))
if not os.path.isdir(SOURCE_DIR):
    raise Exception("could not find image folder at `{}`".format(SOURCE_DIR))

with open(SUBSET_PATH) as f:
    subset_data = json.load(f)
all_paths = []
for paths in subset_data.values():
    all_paths.extend(paths)

print("Copying images to {}...".format(TARGET_DIR))
for p in tqdm(all_paths):
    source_path = os.path.join(SOURCE_DIR, p)
    target_path = os.path.join(TARGET_DIR, p)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy(source_path, target_path)
print("Complete")
