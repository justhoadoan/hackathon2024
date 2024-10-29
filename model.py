import pandas as pd
from collections import Counter
from pathlib import Path
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO
import shutil
import os
import gc

dataset_path = Path("datasets/yolo_data")
labels = sorted(dataset_path.rglob("*labels/*.txt"))

# Assuming your classes are defined in a YAML file
with open("dataset.yaml", "r") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = list(range(len(classes)))

labels_df = pd.DataFrame([], columns=cls_idx, index=[l.stem for l in labels])

for label in labels:
    lbl_counter = Counter()
    with open(label, "r") as lf:
        lines = lf.readlines()
    for l in lines:
        lbl_counter[int(l.split(" ")[0])] += 1
    labels_df.loc[label.stem] = lbl_counter

ksplit = 5
kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)
kfolds = list(kf.split(labels_df))

folds = [f'split_{n}' for n in range(1, ksplit + 1)]
fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
    train_totals = labels_df.iloc[train_indices].sum()
    val_totals = labels_df.iloc[val_indices].sum()

    # To avoid division by zero, we add a small value (1E-7) to the denominator
    ratio = val_totals / (train_totals + 1E-7)
    fold_lbl_distrb.loc[f'split_{n}'] = ratio
kfold_base_path = Path('hackathon/kfold')
shutil.rmtree(kfold_base_path) if kfold_base_path.is_dir() else None # Remove existing folder
os.makedirs(str(kfold_base_path)) # Create nww folder
yaml_paths = list()
train_txt_paths = list()
val_txt_paths = list()

image_paths = [str(l).replace("labels", "images").replace(".txt", ".jpg") for l in labels]
labels_paths= [str(l) for l in labels]


for i, (train_idx, val_idx) in enumerate(kfolds):
    # Get image paths for train-val split
    train_paths = [image_paths[j] for j in train_idx]
    val_paths = [image_paths[j] for j in val_idx]
    # Create text files to store image paths
    train_txt = kfold_base_path / f"train_{i}.txt"
    val_txt =  kfold_base_path / f"val_{i}.txt"

    # Write images paths for training and validation in split i
    with open(str(train_txt), 'w') as f:
        f.writelines(s + '\n' for s in train_paths)
    with open(str(val_txt), 'w') as f:
        f.writelines(s + '\n' for s in val_paths)

    train_txt_paths.append(str(train_txt))
    val_txt_paths.append(str(val_txt))

    # Create yaml file
    yaml_path = kfold_base_path / f'data_{i}.yaml'
    with open(yaml_path, 'w') as ds_y:
        yaml.safe_dump({
            'nc': 4,
            'train': str(train_txt.name),
            'val': str(val_txt.name),
            'names': classes
        }, ds_y)
    yaml_paths.append(str(yaml_path))
if __name__ == '__main__':
    
    gc.collect()
    batch = 64
    project = 'kfold_training'
    epochs = 100
    model = YOLO('yolov8n.pt', task="detect")
    results = {}

    for i in range(ksplit):
        model.train(data=str(yaml_paths[i]), imgsz=640, batch=batch, epochs=epochs, project=project, workers=8)
        results[i]= model.metrics
        
       
   
   