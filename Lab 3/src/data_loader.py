import os
import cv2
import numpy as np

def load_data(folder_path):
    X = []
    y = []

    for label_name in ['NORMAL', 'PNEUMONIA']:
        label_value = -1 if label_name == 'NORMAL' else 1
        sub_folder = os.path.join(folder_path, label_name)
        
        for img_name in os.listdir(sub_folder):
            img_path = os.path.join(sub_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                X.append(img.flatten() / 255.0)
                y.append(label_value)
                
    return np.array(X), np.array(y)

