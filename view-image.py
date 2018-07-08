import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
np.set_printoptions(threshold=np.nan)

train_dir = '/Users/mac/anaconda3/envs/image-classifier/train_images/train_images_2.csv'
data = pd.read_csv(train_dir, nrows=1, header=None)
print(data.shape)
X = data.values.reshape((1, 28, 28, 4))
print(X.shape)
#print(X)
X = X.astype(np.uint8)
plt.imshow(X[0,:,:,:3]) # Muestra la primera imagen todos los canales excepto la distancia (el cuarto)
plt.show()
Y = X[0,:,:,:3]
#print(Y.shape)
#print(Y)
