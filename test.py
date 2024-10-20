from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO('best.pt')
    test_files = os.listdir('./public test')
    
    with open("predict.txt", 'w') as file:
        for test_file in test_files:
            predictions = model(f"./public test/{test_file}", save_txt=None)
            prediction_strings = []
            for idx, prediction in enumerate(predictions[0].boxes.xywhn):
                cls = int(predictions[0].boxes.cls[idx].item())
                prediction_strings.append(f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}")
            file.write(f"{test_file} {' '.join(prediction_strings)}\n")