import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import random
import math
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

#Definiendo constantes, paths, diccionarios y métodos útiles
CLASSNAMES = {
    0 : 'buildings',
    1 : 'barren-land',
    2 : 'water',
    3 : 'grassland', 
    4 : 'roads',
    5 : 'trees', 
}

n_classes = 6
img_shape = 28
n_channels = 3

#Estos métodos son para contar filas, contar archivos en determinados directorios, 
#revolver archivos para darle aleatoriedad al input, y una serie de wrappers que 
#facilitan la implementación del código como su lectura

def count_csv_rows(csvfile):
    total_rows = 0
    with open(csvfile, 'rt') as file:
        row_count = sum(1 for row in csv.reader(file))
        total_rows = row_count
    file.close()    
    return total_rows

def count_files(dir_path):
    total_files = 0
    for file in os.listdir(dir_path):
        #avoid invisible files to be counted
        if not file.startswith('.'):
            total_files += 1

    return total_files

def shuffle_files(file1, file2):
    filenames = list(zip(file1, file2))
    random.shuffle(filenames)
    file1, file2 = zip(*filenames)
    return file1, file2

def ConvLayer(input_layer,n_filters,f_shape, name):
    return tf.layers.conv2d(inputs=input_layer, filters=n_filters, kernel_size=f_shape,
        strides=1, padding="same", name=name, activation=tf.nn.relu)

def Pooling(input_layer,name):
    return tf.layers.max_pooling2d(inputs=input_layer, pool_size=2,
        strides=2, padding="same", name=name)

def OutputLayer(dense_layer):
    return tf.layers.dense(inputs=dense_layer, units=n_classes)


## Cargando nuestros datos

# paths for training images and labels, testing images and labels
train_images_dir = '/Users/mac/code/custom-cnn/train_images/'
train_labels_dir = '/Users/mac/code/custom-cnn/train_labels/'
test_images_dir  = '/Users/mac/code/custom-cnn/test_images/'
test_labels_dir = '/Users/mac/code/custom-cnn/test_labels/'

# counting files in those directories (easier to count respective labels)
n_training_labels = count_files(train_labels_dir)
n_testing_labels = count_files(test_labels_dir)
n_training_files = n_training_labels
n_testing_files = n_testing_labels
print('n training files: ',n_training_files)
print('n training labels: ',n_training_labels)
print('n testing files: ',n_testing_files)
print('n testig labels: ',n_testing_labels)

#Hemos definidos las constantes extract1 y extract2, que disminuyen la cantidad de archivos de images 
#y labels, respectivamente, para hacer una menor cantidad de computaciones (a costa de precisión)

extract1 = 5 # maximo 9
extract2 = 1 # maximo 2

# Lists of all img files and labels
training_filenames = [(train_images_dir+'X_train_%d.csv' % i) for i in range(1,n_training_files-extract1 +1)]
training_labels = [(train_labels_dir+'train_labels_%d.csv' % i) for i in range(1,n_training_labels-extract1 +1)]

testing_filenames = [(test_images_dir+'X_test_%d.csv' % i) for i in range(1,n_testing_files-extract2 +1)]
testing_labels = [(test_labels_dir+'test_labels_%d.csv' % i) for i in range(1,n_testing_labels-extract2 +1)]

print('number of training files we are using:', len(training_filenames))
print('number of training labels we are using:', len(training_labels))
print('number of testing files we are using:', len(testing_filenames))
print('number of testing labels we are using:', len(testing_labels))

# Shuffling files
training_filenames, training_labels = shuffle_files(training_filenames, training_labels)
testing_filenames, testing_labels = shuffle_files(testing_filenames, testing_labels)

# Just one file with its respective label for a quick review 
train_file_path = train_images_dir+'X_train_7.csv'
train_label_path = train_labels_dir+'train_labels_7.csv'

# and a testing case
test_file_path = test_images_dir+'X_test_3.csv'
test_label_path =test_labels_dir+'test_labels_3.csv'

# Counting examples per file 
n_rows_train_file = count_csv_rows(train_label_path)
n_rows_test_file = count_csv_rows(test_label_path)

print('number of rows in training file:', n_rows_train_file)
print('number of rows in testing file:', n_rows_test_file)

# Métodos de Preprocessing

# este metodo tiene como proposito facilitar la visualizacion de las siguientes imagenes
def process_data_for_visualization(train_path, label_path, img_shape, n_rows):
    # Pre-processing data. Returns numpy arrays
    img_df = pd.read_csv(train_path, nrows=n_rows, header=None) 
    labels_df = pd.read_csv(label_path, nrows=n_rows, header=None)
    np_x = img_df.values.astype(np.uint8)
    np_x = np_x.reshape(-1, img_shape, img_shape, 4)
    np_x = np_x[:, :, :, :3]
    np_x = np_x/255

    y = labels_df.values.astype(np.uint8)
    y = np.argmax(y, axis=1)

    return np_x, y

# est emetodo tiene como proposito el procesamiento de los datos para el modelo
def process_data(img_df, labels_df, img_shape):
    # Pre-processing data. Returns numpy arrays
    np_x = img_df.values.astype(np.uint8)
    np_x = np_x.reshape(-1, img_shape, img_shape, 4)
    np_x = np_x[:, :, :, :3]
    np_x = np_x/255

    np_y = labels_df.values.astype(np.uint8)

    return np_x, np_y



#Verificamos las dimensiones de nuestros datos una vez procesados
images, labels = process_data_for_visualization(test_file_path, test_label_path, img_shape, 50)
print(images.shape, labels.shape)



def feed_format(images):
    new_images = np.array(images).reshape(-1,image_size, image_size, 1)
    return new_images


#Verifiquemos su uso, y el formato que nos entrega
new_img_vector = feed_format(img_vector)
print(new_img_vector.shape)
print(new_img_vector[5:6].shape)


tf.reset_default_graph()
n_filters = 32
learning_rate = 0.001
# Input and target placeholders
inputs_ = tf.placeholder(tf.float32, (None, 28,28,1), name="input")
targets_ = tf.placeholder(tf.float32, (None, 28,28,1), name="target")

# Encoder
conv_1 = ConvLayer(inputs_, n_filters, f_shape, 'conv_1') # 28x28x1 -> 28X28X32
pool_1 = Pooling(conv_1, 'pool_1') # 28X28X32 ->14x14x32
conv_2 = ConvLayer(pool_1, n_filters, f_shape, 'conv_2') # 14x14x32 -> 14x14x32
pool_2 = Pooling(conv_2, 'pool_2')# 14x14x32 -> 7x7x32
conv_3 = ConvLayer(pool_2, n_filters/2, f_shape, 'conv_3') # 7x7x32 -> 7x7x16 
encoded = Pooling(conv_3, 'encoded') # 7x7x16 -> 4x4x16

#Decoder
upsample1 = tf.image.resize_images(encoded, size=(7,7), 
method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 4x4x16 -> 7x7x16 

decoder_conv_1 = ConvLayer(upsample1, n_filters, f_shape, 'decoder_conv_1') # 7x7x16 -> 7x7x32

upsample2 = tf.image.resize_images(decoder_conv_1, size=(14,14), 
method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 7x7x32 -> 14x14x32 

decoder_conv_2 = ConvLayer(upsample2, n_filters, f_shape, 'decoder_conv_2') # 14x14x32  -> 14x14x32

upsample3 = tf.image.resize_images(decoder_conv_2, size=(28,28), 
method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 14x14x32 -> 28x28x32 

decoder_conv_3 = ConvLayer(upsample3, n_filters, f_shape, 'decoder_conv_3') # 28x28x32  -> 28x28x32


#logits = tf.layers.conv2d(inputs=conv6, filters=1, kernel_size=(3,3), padding='same', activation=None)
#Now 28x28x1
logits = tf.layers.conv2d(inputs=decoder_conv_3, filters=1, kernel_size=(3,3), padding='same', activation=None)

# Pass logits through sigmoid to get reconstructed image
decoded = tf.nn.sigmoid(logits)

# Pass logits through sigmoid and calculate the cross-entropy loss
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets_, logits=logits)
cost = tf.reduce_mean(loss)
opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess= tf.Session() 
sess.run(tf.global_variables_initializer())

epochs = 1
print("Starting 1st Autoencoder Training...")
for i in range(epochs+1):
    cnt=1
    for data_chunk in pd.read_csv(train_path, chunksize= 100):
        batch_x, _ = preprocess_data(data_chunk, image_size)
        # Run optimization op (backprop).
        training_loss, _ = sess.run([cost, opt ], feed_dict={inputs_: batch_x,
                                                                    targets_: batch_x})
        print("Epoch " + str(i) + " Batch "+str(cnt)+": Loss= " + "{:.6f}".format(training_loss))
        print("Processing next epoch...")
        cnt+=1
print(" 1st Autoencoder training has finished.")



# plotting a Sample
# Get 28x28 image
sample_1 = images[0]
# Get corresponding integer label from one-hot encoded data
sample_label_1 = labels[0]
# Plot sample
print("y = {label_index} ({label})".format(label_index=sample_label_1, label=CLASSNAMES[sample_label_1]))
plt.imshow(sample_1, cmap='gray')


formatted_input = feed_format(sample_1)
reconstructed = sess.run(decoded,feed_dict={inputs_: formatted_input})
print(reconstructed.shape)
# of this 4-dim tensor.
img = reconstructed[0, :, :, 0]
# Plot image.
plt.imshow(img, cmap='gray')










