# coding=utf-8

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 100)
y1 = 1 / (1 + np.exp(-0.5 * x))
y2 = 1 / (1 + np.exp(-1.5*x))
y3 = 1 / (1 + np.exp(-3*x))
plt.figure(figsize=(10,4))
plt.plot(x, y1, label='a=0.5', linewidth=1, linestyle=':')
plt.plot(x, y2, label='a=1.5', linewidth=2, linestyle='--')
plt.plot(x, y3, label='a=3', linewidth=3, linestyle='-.')
plt.legend(loc='lower right')
plt.show()
end = 1