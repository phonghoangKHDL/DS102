import numpy as np
import os

def load_mnist_ubyte(digit1=0, digit2=1):
    train_img_name = 'train-images.idx3-ubyte' 
    train_lbl_name = 'train-labels.idx1-ubyte'

    path_img = os.path.join('data', train_img_name)
    path_lbl = os.path.join('data', train_lbl_name)

    if not os.path.exists(path_img) or not os.path.exists(path_lbl):
        print("Lỗi: Không tìm thấy file trong folder data. Kiểm tra lại tên file!")
        return None, None

    with open(path_lbl, 'rb') as f:
        y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with open(path_img, 'rb') as f:
        X = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(y), 784)

    mask = (y == digit1) | (y == digit2)
    X_bin, y_bin = X[mask], y[mask]

    y_bin = np.where(y_bin == digit1, 0, 1)
    X_bin = X_bin / 255.0

    return X_bin, y_bin


def load_mnist_full():
    train_img_name = 'train-images.idx3-ubyte' 
    train_lbl_name = 'train-labels.idx1-ubyte'
    
    path_img = os.path.join('data', train_img_name)
    path_lbl = os.path.join('data', train_lbl_name)

    with open(path_lbl, 'rb') as f:
        y = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    with open(path_img, 'rb') as f:
        X = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(y), 784)

    X_all = X / 255.0
    return X_all, y

