import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_mnist_ubyte
from src.models import LogisticRegressionGD
from src.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = load_mnist_ubyte(0, 1)

model = LogisticRegressionGD(lr=3, epochs=1000)
model.fit(X, y)

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print(f"\n--- KẾT QUẢ ---")
print(f"Độ chính xác (Accuracy): {acc * 100:.2f}%")

plt.plot(model.losses)
plt.title("Biểu đồ giảm lỗi (Loss Curve)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()