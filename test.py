from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO('best.pt')
    test_files = os.listdir('./public test')
    
    with open("predict.txt", 'w') as file:
        for test_file in test_files:
            predictions = model(f"./public test/{test_file}", save_txt=None)
            for idx, prediction in enumerate(predictions[0].boxes.xywhn):
                cls = int(predictions[0].boxes.cls[idx].item())
                conf = predictions[0].boxes.conf[idx].item()
                file.write(f"{test_file} {cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()} {conf}\n")