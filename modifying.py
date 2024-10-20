import os
import shutil

night_files_path = "yolo_data/labels/train"
night_files = os.listdir(night_files_path)
for night_file in night_files:
    if night_file.endswith(".txt"):
        night_file_path = os.path.join(night_files_path, night_file)
        with open(night_file_path, 'r') as file:
            lines = file.readlines()
        with open(night_file_path, 'w') as file:
            for line in lines:
                if line[0].isdigit() and int(line[0]) >= 4:
                    line = str(int(line[0]) - 4) + line[1:]
                file.write(line)

night_files_path = "yolo_data/labels/val"
night_files = os.listdir(night_files_path)
for night_file in night_files:
    if night_file.endswith(".txt"):
        night_file_path = os.path.join(night_files_path, night_file)
        with open(night_file_path, 'r') as file:
            lines = file.readlines()
        with open(night_file_path, 'w') as file:
            for line in lines:
                if line[0].isdigit() and int(line[0]) >= 4:
                    line = str(int(line[0]) - 4) + line[1:]
                file.write(line)


