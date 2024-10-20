from ultralytics import YOLO
import numpy as np
from PIL import Image
import os

if __name__ == '__main__':
    model = YOLO("runs/detect/train/weights/best.pt")

    # test_files = os.listdir("public_test")
    # test_files_path = [os.path.join("public_test", test_files)]
    #
    # results = model(test_files_path)
    # for i, result in enumerate(results):
    #     result.show()
    #     if i == 5:
    #         break

    result = model("5.jpg")
    for r in result:
        r.show()