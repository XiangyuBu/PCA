import numpy as np
import matplotlib.pyplot as plt

'''
This python program can execute the PCA method to reduce dimensions.
And this program use KMeans in sklearn to cluster.
'''
A = np.loadtxt('PCA.txt')
print(A.shape)
A = np.reshape(A, (1600, 6400))
mean = np.mean(A, axis=0)
X = A - mean
#print(A)
X = np.dot(X.T, X)
#print(A)
U, S, V = np.linalg.svd(X)
S = S / S.sum()
U_reduce1 = -U[:, 0]
U_reduce2 = -U[:, 1]
#print(U_reduce.shape)
#print(U_reduce1)
#print(S)
y1 = np.dot(A, U_reduce1)
y2 = np.dot(A, U_reduce2)

N = np.vstack((y1, y2))
N = N.T
#print(N)

from sklearn.cluster import KMeans
n_cluster = 3
kmean = KMeans(n_cluster)
kmean.fit(N)
labels = kmean.labels_
centers = kmean.cluster_centers_


xx = np.arange(1, 11)
plt.figure(figsize=(8, 6), dpi=200)
plt.xlim(1, 10)
plt.ylim(0.001, 1)
plt.semilogy()
plt.xlabel("l", fontsize=16, color='black', horizontalalignment='center', fontname='Brush Script MT')
plt.ylabel(r'$\bar {\lambda}_{l}$', fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Brush Script MT')  
plt.plot(xx, S[xx-1], linestyle="-", marker="o", color="b", linewidth=2)
plt.savefig('111.png')

T = np.arange(len(y1))
plt.figure(figsize=(8, 6), dpi=200)
plt.xlim(-100, 100)
plt.ylim(-50, 50)
plt.xlabel("$y_1$", fontsize=16, color='black', horizontalalignment='center', fontname='Brush Script MT')
plt.ylabel("$y_2$", fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Brush Script MT')  
plt.scatter(y1, y2, s=75, c=T)
cbar = plt.colorbar()
#cbar.set_label('$T_B(K)$',fontdict=font)
cbar.set_ticks(np.linspace(0, 1600, 9))
#cbar.set_ticklabels(('1.4', '1.5', '1.6', '1.7', '1.8', '1.9', '2.0', '2.1',
#	                 '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9'))
cbar.set_ticklabels(('1.4', '1.6', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8'))
plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='white', s=300)
plt.savefig('PCA.png')


