import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_mnist_full
from src.models import SoftmaxRegressionGD
from src.metrics import accuracy_score
import matplotlib.pyplot as plt

X, y = load_mnist_full()

model = SoftmaxRegressionGD(lr=1, epochs=500)
model.fit(X, y)

y_pred = model.predict(X)
print(f"Accuracy on 10 classes: {accuracy_score(y, y_pred)*100:.2f}%")

plt.plot(model.losses)
plt.title("Softmax Regression Loss Curve")
plt.show()