import Perceptrone
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data', header=None)


y = df.iloc[0:100, 4].values # what a f@ck is this , anyway ?
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0,2]].values

plt.scatter(X[:50,0], X[:50,1],color='red',marker='o',label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue',marker='x',label='versicolor')

plt.xlabel('petal length')
plt.ylabel('petal length')
plt.legend(loc='upper left')

plt.show()
savefig('foo1.png') # i hope to save pics like this

ppn = Perceptrone(eta=0.1, n_iter=10) #initializing perceptrone with parameters
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) +1) , ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')

plt.show()
plt.savefig('myfig1')


from matplotlib.colors import ListedColormap

# marker generator and color map

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('s','x','o','^','v')
    colors = ('red', 'blue', 'lightgreen', 'gray','cyan')
cmap = ListedColormap(colors[:len(np.unique(y))])

# plot the decision surface

x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
x2_min, x2_max = X[:,1].min() -1 , X[:,1].max() +1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arrange(x2_min, x2_max, resolution))

Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T) #hmm.. who is ravel ?
Z = Z.reshape(xx1.shape)

plt.contourf(xx1,xx2,Z, alpha=0.4, cmap=cmap) #why cmap equals cmap
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())

for idx, cl in enumerate (np.unique(y)):
    plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.show()
plt.savefig('myfig2')


