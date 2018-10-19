import numpy as np
import matplotlib.pyplot as plt

'''
This python program can execute the PCA method to reduce dimensions.
And this program use KMeans in sklearn to cluster.
'''
#load the dataset
A = np.loadtxt('PCA.txt')
A = A[:, 2]
print(A.shape)
A = np.reshape(A, (1400, 1600))
#normalize the array and reduce the dimsion of the array
mean = np.mean(A, axis=0)
X = A - mean
X = np.dot(X.T, X)
U, S, V = np.linalg.svd(X)


S = S / S.sum()
U_reduce1 = -U[:, 0]
U_reduce2 = -U[:, 1]
U_reduce3 = -U[:, 2]
U_reduce4 = -U[:, 3]
#print(U_reduce1.shape)
#print(U_reduce2)
#print(S)
y1 = np.dot(A, U_reduce1)
y2 = np.dot(A, U_reduce2)
y3 = np.dot(A, U_reduce3)
y4 = np.dot(A, U_reduce4)
######This function will be used in fig.7################################
yy = y1*y1 + y2*y2 + y3*y3 + y4*y4
yy = np.reshape(yy, (14, 100))
yy = yy.sum(1)
yy = yy/10
#########################################################################
########This section will be used in fig.6(a) and kmeans cluster#########
yy1 = y1*y1+y2*y2
yy2 = y3*y3+y4*y4
N = np.vstack((yy1, yy2))
N = N.T
#print(N)
#########################################################################
#ues kmeans to cluster the dataset####################################### 
from sklearn.cluster import KMeans
n_cluster = 3
kmean = KMeans(n_cluster)
kmean.fit(N)
labels = kmean.labels_
centers = kmean.cluster_centers_
##########################################################################
##############draw the figures from here beginning########################
#plot the leading principal components with fig.4#########################
xx = np.arange(1, 11)
plt.figure(figsize=(8, 6), dpi=200)
plt.xlim(1, 10)
plt.ylim(0.00, 0.12)
#plt.semilogy()
plt.xlabel("l", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel(r'$\bar {\lambda}_{l}$', fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.plot(xx, S[xx-1], linestyle="-", marker="o", color="b", linewidth=2)
plt.savefig('fig4.png')
##########################################################################
#plot the weight of the four leading principal components over the lattice sites.fig.4 inset.
plt.figure(figsize=(8, 6), dpi=200)
plt.subplot(2, 2, 1)
U_reduce1 = U_reduce1.T.reshape(40, 40)
U_reduce1 = U_reduce1.T
plt.title('w1', fontname='Times New Roman')
plt.xticks(())
plt.yticks(())
plt.imshow(U_reduce1, cmap='jet', interpolation='bilinear')
plt.colorbar()

plt.subplot(2, 2, 2)
U_reduce2 = U_reduce2.reshape(40, 40)
U_reduce2 = U_reduce2.T
plt.title('w2', fontname='Times New Roman')
plt.xticks(())
plt.yticks(())
plt.imshow(U_reduce2, cmap='jet', interpolation='bilinear')
plt.colorbar()

plt.subplot(2, 2, 3)
U_reduce3 = U_reduce3.reshape(40, 40)
U_reduce3 = U_reduce3.T
plt.title('w3', fontname='Times New Roman')
plt.xticks(())
plt.yticks(())
plt.imshow(U_reduce3, cmap='jet', interpolation='bilinear')
plt.colorbar()

plt.subplot(2, 2, 4)
U_reduce4 = U_reduce4.reshape(40, 40)
U_reduce4 = U_reduce4.T
plt.title('w4', fontname='Times New Roman')
plt.xticks(())
plt.yticks(())
plt.imshow(U_reduce4, cmap='jet', interpolation='bilinear')
plt.colorbar()
plt.savefig('w.png')
#############################################################################
###############################################fig.1 inset##############################################
#In this program, this section is not used
#xx = np.arange(0, 1600)
#plt.figure(figsize=(8, 6), dpi=200)
#plt.xlim(1, 10)
#plt.ylim(0.00, 0.05)
#plt.semilogy()
#plt.xlabel("i", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
#plt.ylabel(r'${\omega}_{1}$', fontsize=16, color='black', horizontalalignment='center',
#	 	   fontname='Times New Roman')  
#plt.scatter(xx, U_reduce1[xx], marker="o", color="r", s=10)
#plt.savefig('w1.png')

#plt.figure(figsize=(8, 6), dpi=200)
#plt.xlim(1, 10)
#plt.ylim(-0.10, 0.10)
#plt.semilogy()
#plt.xlabel("i", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
#plt.ylabel(r'${\omega}_{2}$', fontsize=16, color='black', horizontalalignment='center',
#	 	   fontname='Times New Roman')  
#plt.scatter(xx, U_reduce2[xx], marker="o", color="r", s=10)
#plt.savefig('w2.png')
########################################################################################################
T = np.arange(len(y1))
#plot fig.5#############################################################################################
plt.figure(figsize=(18, 10), dpi=200)
#(2,3,1)
plt.subplot(2, 3, 1)
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xlabel("$y_1$", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel("$y_2$", fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.scatter(y1, y2, s=10, cmap='jet', c=T)
cbar = plt.colorbar()
#cbar.set_label('$T_B(K)$',fontdict=font)
cbar.set_ticks(np.linspace(0, 1400, 8))
#cbar.set_ticklabels(('1.4', '1.51.6', '1.7', '1.8', '1.9', '2.0', '2.1',
#	                 '2.2', '2.3', '2.4', '2.5', '2.6', '2.7', '2.8', '2.9'))
cbar.set_ticklabels(('1.6', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8'))
#plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='white', s=300)

plt.subplot(2, 3, 2)
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xlabel("$y_1$", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel("$y_3$", fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.scatter(y1, y3, s=10, cmap='jet', c=T)
cbar = plt.colorbar()
#cbar.set_label('$T_B(K)$',fontdict=font)
cbar.set_ticks(np.linspace(0, 1400, 8))
cbar.set_ticklabels(('1.6', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8'))

plt.subplot(2, 3, 3)
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xlabel("$y_1$", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel("$y_4$", fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.scatter(y1, y4, s=10, cmap='jet', c=T)
cbar = plt.colorbar()
#cbar.set_label('$T_B(K)$',fontdict=font)
cbar.set_ticks(np.linspace(0, 1400, 8))
cbar.set_ticklabels(('1.6', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8'))

plt.subplot(2, 3, 4)
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xlabel("$y_2$", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel("$y_3$", fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.scatter(y2, y3, s=10, cmap='jet', c=T)
cbar = plt.colorbar()
#cbar.set_label('$T_B(K)$',fontdict=font)
cbar.set_ticks(np.linspace(0, 1400, 8))
cbar.set_ticklabels(('1.6', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8'))

plt.subplot(2, 3, 5)
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xlabel("$y_2$", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel("$y_4$", fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.scatter(y2, y4, s=10, cmap='jet', c=T)
cbar = plt.colorbar()
#cbar.set_label('$T_B(K)$',fontdict=font)
cbar.set_ticks(np.linspace(0, 1400, 8))
cbar.set_ticklabels(('1.6', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8'))

plt.subplot(2, 3, 6)
plt.xlim(-50, 50)
plt.ylim(-50, 50)
plt.xlabel("$y_3$", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel("$y_4$", fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.scatter(y3, y4, s=10, cmap='jet', c=T)
cbar = plt.colorbar()
#cbar.set_label('$T_B(K)$',fontdict=font)
cbar.set_ticks(np.linspace(0, 1400, 8))
cbar.set_ticklabels(('1.6', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8'))
plt.savefig('fig5.png')
########################################################################################
#################plot fig.6#############################################################
#plot fig.6(a) 
plt.figure(figsize=(18, 12), dpi=200)
plt.xlabel("$y_1^2 + y_2^2$", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel("$y_3^2 + y_4^2$", fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.scatter((y1*y1+y2*y2), (y3*y3+y4*y4), s=75, cmap='jet', c=T)
cbar = plt.colorbar()
#cbar.set_label('$T_B(K)$',fontdict=font)
cbar.set_ticks(np.linspace(0, 1400, 8))
cbar.set_ticklabels(('1.6', '1.8', '2.0', '2.2', '2.4', '2.6', '2.8'))
plt.scatter(centers[:, 0], centers[:, 1], marker='x', c='black', s=500)
plt.savefig('fig6.png')
########################################################################################
#############################plot fig.7#################################################
ss = np.arange(1.6, 3.0, 0.1)
plt.figure(figsize=(8, 6), dpi=200)
#plt.xlim(1, 10)
#plt.ylim(0.00, 0.12)
#plt.semilogy()
plt.xlabel("T/J", fontsize=16, color='black', horizontalalignment='center', fontname='Times New Roman')
plt.ylabel('S', fontsize=16, color='black', horizontalalignment='center',
	 	   fontname='Times New Roman')  
plt.plot(ss, yy, linestyle="-", marker="o", color="b", linewidth=2)
plt.savefig('fig7.png')
########################################################################################