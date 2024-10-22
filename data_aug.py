import albumentations as A
import cv2
import os

images_path = 'yolo_data/images/train'
labels_path = 'yolo_data/labels/train'

augmented_images_path = 'data_augmentation/train_augmented_images'
augmented_labels_path = 'data_augmentation/train_augmented_labels'

os.makedirs(augmented_images_path, exist_ok=True)
os.makedirs(augmented_labels_path, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.4),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=10, p=0.4),
    A.Affine(scale=(0.85, 1.1), shear=10),
    A.ColorJitter(brightness=0.3, contrast=0.8, saturation=0.5, hue=0.1)
], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

for image_file in os.listdir(images_path):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(images_path, image_file)
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(labels_path, label_file)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(label_path, 'r') as f:
            bboxes = []
            category_ids = []
            for line in f.readlines():
                bbox = list(map(float, line.strip().split()))
                category_ids.append(int(bbox[0]))
                bboxes.append(bbox[1:])

        try:
            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        except:
            augmented_image_path = os.path.join(augmented_images_path, image_file)
            cv2.imwrite(augmented_image_path, image)
            augmented_label_path = os.path.join(augmented_labels_path, label_file)
            with open(augmented_label_path, 'w') as f:
                for i, bbox in enumerate(bboxes):
                    bbox_str = ' '.join(map(str, [int(category_ids[i]), *bbox]))
                    f.write(bbox_str + '\n')
            continue

        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_category_ids = transformed['category_ids']

        augmented_image_path = os.path.join(augmented_images_path, image_file)
        cv2.imwrite(augmented_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

        augmented_label_path = os.path.join(augmented_labels_path, label_file)
        with open(augmented_label_path, 'w') as f:
            for i, bbox in enumerate(transformed_bboxes):
                bbox_str = ' '.join(map(str, [int(transformed_category_ids[i]), *bbox]))
                f.write(bbox_str + '\n')
