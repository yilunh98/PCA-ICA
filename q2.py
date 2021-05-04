import numpy as np
import matplotlib.pyplot as plt
import time

def validate_PCA(states, train_data):
  from sklearn.decomposition import PCA
  pca = PCA(n_components=10)
  pca.fit(train_data)
  true_matrix = pca.components_.T
  true_ev = pca.explained_variance_
  
  output_matrix = states['transform_matrix']
  error = np.mean(np.abs(np.abs(true_matrix) - np.abs(output_matrix)) / np.abs(true_matrix))
  if error > 0.01:
    print('Matrix is wrong! Error=',error)
  else:
    print('Matrix is correct! Error=', error)

  output_ev = states['eigen_vals']
  error = np.mean(np.abs(true_ev - output_ev) / true_ev)
  if error > 0.01:
    print('Variance is wrong! Error=', error)
  else:
    print('Variance is correct! Error=', error)

def train_PCA(train_data):
  states = {
      'transform_matrix': np.identity(train_data.shape[-1]),
      'eigen_vals': np.ones(train_data.shape[-1])
  }

  average = np.mean(train_data, axis=0)
  diff = train_data - average
  eigenval,eigenvec = np.linalg.eig(diff.T@diff/train_data.shape[0])
  sortindex = np.argsort(-eigenval)
  sortval = eigenval[sortindex]
  sortvec = eigenvec[:, sortindex]
  
  plt.scatter(sortindex, sortval)
  plt.show()
  plt.savefig('eigenvalues.png')

  K = 10
  pcaval = sortval[0:K]
  pcavec = sortvec[:, 0:K]
  print('eigenvalues corresponding to the first 10 principal component: ', pcaval) 

  fig, axes = plt.subplots(2, 5)
  count = 0
  for i in np.arange(2):
    for j in np.arange(5):
        if (i==0)&(j==0):
          axes[i, j].imshow(np.reshape(average, (48, -1)))
        else:
          axes[i, j].imshow(np.reshape(pcavec[:, count], (48, -1)))
          count = count + 1
  plt.savefig('eigenvectors.png')

  reconstruct = train_data@pcavec@pcavec.T

  states['transform_matrix'] = pcavec
  states['eigen_vals'] = pcaval

  return states

# Load data
start = time.time()
images = np.load('q2data/q2.npy')
num_data = images.shape[0]
train_data = images.reshape(num_data, -1)

states = train_PCA(train_data)
print('training time = %.1f sec'%(time.time() - start))

validate_PCA(states, train_data)
