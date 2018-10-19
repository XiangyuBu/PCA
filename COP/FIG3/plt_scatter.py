import numpy as np
import matplotlib.pyplot as plt

A = np.loadtxt('PCA.txt')
y1 = A[:, 0]
y2 = A[:, 1]
T = A[:, 2]
plt.figure(figsize=(8, 6), dpi=100)
plt.xlim(0, 21)
plt.ylim(0, 21)
plt.xlabel("x", fontsize=16, color='black', horizontalalignment='center', fontname='Brush Script MT')
plt.ylabel("y", fontsize=16, color='black', horizontalalignment='center', fontname='Brush Script MT')  
plt.scatter(y1, y2, marker=',', s=450, c=T)
plt.savefig('PCA.png')
