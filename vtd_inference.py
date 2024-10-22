import os

output_dir = 'runs/detect/predict/labels/'
output_file = 'predict.txt'

with open(output_file, 'w') as f_out:
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(output_dir, filename), 'r') as f_in:
                for line in f_in:
                    f_out.write(line)
