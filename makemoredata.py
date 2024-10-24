import os
import shutil

file_img = os.listdir('data_augmentation/train_augmented_images')
file_label = os.listdir('data_augmentation/train_augmented_labels')
train_path_img = "datasets/yolo_data/images/train/"
train_path_label = "datasets/yolo_data/labels/train/"
val_path_img = "datasets/yolo_data/images/val/"
val_path_label = "datasets/yolo_data/labels/val/"


for filex in file_img:
    file_path = os.path.join('data_augmentation/train_augmented_images', filex)
    file_name, file_ext = os.path.splitext(filex)
    new_file_name = f"{file_name}aug{file_ext}"
    new_file_path = os.path.join(train_path_img, new_file_name)
    shutil.copy(file_path, new_file_path)

for filex in file_label:
    file_path = os.path.join('data_augmentation/train_augmented_labels', filex)
    file_name, file_ext = os.path.splitext(filex)
    new_file_name = f"{file_name}aug{file_ext}"
    new_file_path = os.path.join(train_path_label, new_file_name)
    shutil.copy(file_path, new_file_path)

