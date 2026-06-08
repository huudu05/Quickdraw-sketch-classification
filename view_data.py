import matplotlib.pyplot as plt
import numpy as np
from src.config import CLASSES

data = np.load(f"data/{CLASSES[0]}.npy")
print(data.shape)

plt.imshow(data[0].reshape(28,28), cmap='gray')
plt.show()