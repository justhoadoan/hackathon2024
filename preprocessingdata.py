import os
import shutil

files_train = os.listdir('./train/')
file_test= os.listdir('./public test/')
train_path_img = "./yolo_data/images/train/"
train_path_label = "./yolo_data/labels/train/"
val_path_img = "./yolo_data/images/val/"
val_path_label = "./yolo_data/labels/val/"
test_path = "./yolo_data/test"

# train_files = os.listdir(train_path_img)
# val_files = os.listdir(val_path_img)
# print(len(train_files), len(val_files))
split=0.2
os.makedirs(train_path_img, exist_ok = True)
os.makedirs(train_path_label, exist_ok = True)
os.makedirs(val_path_img, exist_ok = True)
os.makedirs(val_path_label, exist_ok = True)

for filex in files_train[int(len(files_train)*split):]:
      file_path = os.path.join('./train/', filex)

      if filex.endswith(".txt"):
          shutil.copy(file_path, train_path_label)
      else:
            shutil.copy(file_path, train_path_img)
for filex in files_train[:int(len(files_train)*split)]:
      file_path = os.path.join('./train/', filex)

      if filex.endswith(".txt"):
          shutil.copy(file_path, val_path_label)
      else:
            shutil.copy(file_path, val_path_img)
