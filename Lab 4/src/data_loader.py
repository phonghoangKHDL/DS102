import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_wine_data(file_path, test_size=0.2, random_state=42):
    df = pd.read_csv(file_path, sep=';')

    # Chuyển điểm số thành nhãn nhị phân: 'quality' > 5 là tốt (1), ngược lại là tệ (0)
    df['target'] = (df['quality'] > 5).astype(int)

    X = df.drop(['quality', 'target'], axis=1).values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test
