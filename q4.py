import numpy as np
from scipy.io.wavfile import write
Fs = 11025
    
def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('q4data/q4.dat')
    return mix

def sigmoid(X):
    return 1 / (1+np.exp(-X))

def unmixer(X):
    N, M = X.shape
    W = np.eye(M)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
                    0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')

    W_temp = np.zeros_like(W)
    for alpha in anneal:
        for i in range(N):
            x = X[i, :].reshape(-1, 1)
            Z = 1 - 2 * sigmoid(W @ x)
            W = W + alpha * (Z @ x.T + np.linalg.inv(W.T))   
    return W

def unmix(X, W):
    # S = np.zeros(X.shape)
    # average = np.mean(X,axis=0)
    # diff = X - average[np.newaxis, :]
    # eigenval, eigenvec = np.linalg.eig(diff.T@diff / X.shape[0])
    # val2 = np.linalg.inv(np.diag(eigenval))
    # val = np.sqrt(val2)
    # pca = val@eigenvec.T@X.T
    # S = W @ pca 
    # return S.T
    S = X @ W.T
    return S

X = normalize(load_data())
print(X.shape)
print('Saving mixed track 1')
write('q4_mixed_track_1.wav', Fs, X[:, 0])

import time
t0 = time.time()
W = unmixer(X) # This will take around 2min
print('time=', time.time()-t0)
S = normalize(unmix(X, W))

for track in range(5):
    print(f'Saving unmixed track {track}')
    write(f'q4_unmixed_track_{track}.wav', Fs, S[:, track])

print('W solution:')
print(W)
