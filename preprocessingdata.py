import os
import shutil
import pandas as pd

def modify_txt_file(file_path):
    
    df = pd.read_csv(file_path, header=None, delim_whitespace=True)
    df.iloc[:, 0] = df.iloc[:, 0].apply(lambda x: x - 4 if x >= 4 else x)
    df.to_csv(file_path, header=False, index=False, sep=' ')
        
files_train = os.listdir('./train/')
file_test= os.listdir('./public test/')
train_path_img = "./yolo_data/images/train/"
train_path_label = "./yolo_data/labels/train/"
val_path_img = "./yolo_data/images/val/"
val_path_label = "./yolo_data/labels/val/"
test_path = "./yolo_data/test"

split=0.2
os.makedirs(train_path_img, exist_ok = True)
os.makedirs(train_path_label, exist_ok = True)
os.makedirs(val_path_img, exist_ok = True)
os.makedirs(val_path_label, exist_ok = True)

path= "/train/"

for filex in files_train[:int(len(files_train)*(1-split))]:
      
      file_path=os.path.join('./train/', filex)
      
      if filex.endswith(".txt"):
         
          modify_txt_file(file_path)        
          shutil.copy(file_path, train_path_label)
      else:
          shutil.copy(file_path, train_path_img)
for filex in files_train[int(len(files_train)*(1-split)):]:
      file_path=os.path.join('./train/', filex)
      
      if filex.endswith(".txt"):
          modify_txt_file(file_path)
          shutil.copy(file_path, val_path_label)
      else:
            shutil.copy(file_path, val_path_img)
