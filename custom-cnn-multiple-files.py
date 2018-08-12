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


n_classes = 6
shape = 28
n_channels = 3
epochs = 2
extract1 = 8
extract2 = 1

# defining useful methods
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
        if not file.startswith('.'):
            total_files += 1

    return total_files

def shuffle_files(file1, file2):
    filenames = list(zip(file1, file2))
    random.shuffle(filenames)
    file1, file2 = zip(*filenames)
    return file1, file2

def process_data(img_df, labels_df, img_shape):
    # Pre-processing data. Returns numpy arrays
    np_x = img_df.values.astype(np.uint8)
    np_x = np_x.reshape(-1, img_shape, img_shape, 4)
    np_x = np_x[:, :, :, :3]

    np_y = labels_df.values.astype(np.uint8)

    return np_x, np_y

def getActivations(layer,stimuli, name):
    img = pd.read_csv(stimuli,nrows=1,header=None).values.astype(np.uint8)
    img = img.reshape(-1, 28, 28, 4)
    img = img[:, :, :, :3]
    units = sess.run(layer,feed_dict={x: img,keep_prob:1.0})
    plotNNFilter(units, name)

def plotNNFilter(units, name):
    filters = units.shape[3]
    fig = plt.figure(1, figsize=(15,15))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        fig.add_subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i])
        plt.tight_layout()
    fig.savefig(name+'.png')    


def ConvLayer(input_layer,n_filters,f_shape, name):
    return tf.layers.conv2d(inputs=input_layer, filters=n_filters, kernel_size=f_shape,
        strides=1, padding="same", name=name, activation=tf.nn.relu)

def Pooling(input_layer,name):
    return tf.layers.max_pooling2d(inputs=input_layer, pool_size=2,
        strides=2, padding="same", name=name)

def OutputLayer(dense_layer):
    return tf.layers.dense(inputs=dense_layer, units=n_classes)


#defining model
def my_cnn_arch(input_layer):
    # Defining filters
    n_filters = 32
    f_shape = 3

    # input -> first layer
    conv_layer1 = ConvLayer(input_layer, n_filters, f_shape, 'conv_layer1')
    conv_layer2 = ConvLayer(conv_layer1, n_filters, f_shape, 'conv_layer2')
    pool_layer1 = Pooling(conv_layer2, 'pool_layer1')

    # first layer output -> second layer
    conv_layer3 = ConvLayer(pool_layer1, 2*n_filters, f_shape, 'conv_layer3')
    conv_layer4 = ConvLayer(conv_layer3, 2*n_filters, f_shape, 'conv_layer4')
    pool_layer2 = Pooling(conv_layer4, 'pool_layer2')

    flat_layer = tf.layers.flatten(inputs=pool_layer2)

    dense_layer1 = tf.layers.dense(inputs=flat_layer, units=1024, activation= tf.nn.relu)
    
    dropout = tf.layers.dropout(inputs= dense_layer1, rate=0.4)

    output = OutputLayer(dropout)

    return output


# paths for training images and labels, testing images and labels
train_images_dir = '/Users/mac/code/custom-cnn/train_images/'
train_labels_dir = '/Users/mac/code/custom-cnn/train_labels/'
test_images_dir  = '/Users/mac/code/custom-cnn/test_images/'
trest_labels_dir = '/Users/mac/code/custom-cnn/test_labels/'

# counting files in those directories (easier to count respective labels)
n_training_labels = count_files(train_labels_dir)
n_testing_labels = count_files(trest_labels_dir)
n_training_files = n_training_labels
n_testing_files = n_testing_labels

# Lists of all img files and labels
training_filenames = [(train_images_dir+'X_train_%d.csv' % i) for i in range(1,n_training_files-extract1)]
training_labels = [(train_labels_dir+'train_labels_%d.csv' % i) for i in range(1,n_training_labels-extract1)]

testing_filenames = [(test_images_dir+'X_test_%d.csv' % i) for i in range(1,n_testing_files-extract2)]
testing_labels = [(trest_labels_dir+'test_labels_%d.csv' % i) for i in range(1,n_testing_labels-extract2)]

# Shuffling files
training_filenames, training_labels = shuffle_files(training_filenames, training_labels)
testing_filenames, testing_labels = shuffle_files(testing_filenames, testing_labels)


# Just one file with its respective label for a quick review 
train_file_path = train_images_dir+'X_train_1.csv'
train_label_path = train_labels_dir+'train_labels_1.csv'

# and a testing case
test_file_path = test_images_dir+'X_test_1.csv'
test_label_path =trest_labels_dir+'test_labels_1.csv'

# Counting examples per file 
n_rows_train_file = count_csv_rows(train_label_path)
n_rows_test_file = count_csv_rows(test_label_path)

# Hyperparameters and constants regarding each dataset
learning_rate = 0.001 
batch_size = n_rows_train_file//100
test_batch_size = n_rows_test_file//100

# Defining data placeholders
x = tf.placeholder(tf.float32, [None, 28, 28, 3])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder("float")

# Conv Net and Optimization
# prediction = my_cnn_arch(x)

n_filters = 32
f_shape = 3

# input -> first layer
conv_layer1 = ConvLayer(x, n_filters, f_shape, 'conv_layer1')
conv_layer2 = ConvLayer(conv_layer1, n_filters, f_shape, 'conv_layer2')
pool_layer1 = Pooling(conv_layer2, 'pool_layer1')

# first layer output -> second layer
conv_layer3 = ConvLayer(pool_layer1, 2*n_filters, f_shape, 'conv_layer3')
conv_layer4 = ConvLayer(conv_layer3, 2*n_filters, f_shape, 'conv_layer4')
pool_layer2 = Pooling(conv_layer4, 'pool_layer2')

flat_layer = tf.layers.flatten(inputs=pool_layer2)

dense_layer1 = tf.layers.dense(inputs=flat_layer, units=1024, activation= tf.nn.relu)

dropout = tf.layers.dropout(inputs= dense_layer1, rate=keep_prob)

prediction = OutputLayer(dropout)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init) 
    train_loss = []
    train_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
    print("Starting Training...")
    for i in range(epochs):

        for j in range(len(training_filenames)):
            print("Training File " + str(j) + ": ")
            cnt = 1
            for image_chunk,label_chunk in zip(pd.read_csv(training_filenames[j], header=None, chunksize= batch_size), pd.read_csv(training_labels[j],header=None,chunksize=batch_size)):
                batch_x, batch_y = process_data(image_chunk, label_chunk, shape)

                # Run optimization op (backprop).
                opt = sess.run(optimizer, feed_dict={x: batch_x,y: batch_y, keep_prob: 0.5})
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
                train_loss.append(loss)
                train_accuracy.append(acc)
                print("Batch  " + "{:6d}".format(cnt)  + ": Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                print("Processing next training Batch...")
                cnt+=1

        print("Epoch " + str(i) + ": Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Processing next epoch...")

    print("Training has finished.")


    print("Starting Testing...")

    total_test_acc = 0 
    for k in range(len(testing_filenames)):
        print('Testing File {:.2f}: ', k)
        cnt=1
    # TODO: shuffle those dataframes    
        for image_chunk,label_chunk in zip(pd.read_csv(test_file_path, header=None, chunksize= test_batch_size), pd.read_csv(test_label_path,header=None,chunksize=test_batch_size)):
            test_batch_x, test_batch_y = process_data(image_chunk, label_chunk, shape)

            test_acc, test_loss = sess.run([accuracy,cost], feed_dict={x: test_batch_x, y : test_batch_y, keep_prob: 1.0})
            total_test_acc += test_acc
            print("Batch " + "{:6d}".format(cnt) + ": Loss= " + \
                          "{:.6f}".format(test_loss) + ", Testing Accuracy= " + \
                          "{:.5f}".format(test_acc))
            print("Processing next testing Batch...")
            cnt+=1
    
    n_test_batches = n_rows_test_file//test_batch_size
    avg_test_acc = total_test_acc/n_test_batches
    print("Testing Accuracy:","{:.5f}".format(avg_test_acc))
    
    fig1 = plt.figure()
    plt.plot(range(len(train_loss)), train_loss, 'b', label='Training loss')
    plt.title('Training  loss')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.legend()
    fig1.savefig('training-loss.png')
    plt.show()

    fig2 = plt.figure()
    plt.plot(range(len(train_loss)), train_accuracy, 'b', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.legend()
    fig2.savefig('training-acc.png')
    plt.show()



    imageToUse = test_file_path
    reshaped_img = pd.read_csv(imageToUse, nrows = 1, header=None)
    reshaped_img = reshaped_img.values.astype(np.uint8)
    reshaped_img = reshaped_img.reshape(1, 28, 28, 4)
    reshaped_img = reshaped_img[0, :, :, :3]
    plt.imsave('img1.png', reshaped_img)
    plt.imshow(reshaped_img)
    plt.show()


    getActivations(conv_layer1,imageToUse,'conv1-filters')
    plt.tight_layout()
    plt.show()

    getActivations(conv_layer1,imageToUse,'conv1-filters')
    plt.tight_layout()
    plt.show()

    getActivations(conv_layer2,imageToUse, 'conv2-filters')
    plt.tight_layout()
    plt.show()

    getActivations(conv_layer3,imageToUse, 'conv3-filters')
    plt.tight_layout()
    plt.show()

    getActivations(conv_layer3,imageToUse, 'conv3-filters')
    plt.tight_layout()
    plt.show()
    summary_writer.close()














