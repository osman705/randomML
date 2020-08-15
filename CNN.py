import os
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu,floatX=float32'
import theano as th
import cv2
import numpy as np
import pandas as pd
from sklearn import datasets,preprocessing
from sklearn.preprocessing import LabelEncoder
from skimage import io,transform
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam
from keras import backend as K


"""
Info 

Sample: one element of a dataset.
Example: one image is a sample in a convolutional network
Example: one audio file is a sample for a speech recognition model

Batch: a set of N samples. The samples in a batch are processed independently, in parallel. If training, a batch results in only one update to the model.
A batch generally approximates the distribution of the input data better than a single input. 
The larger the batch, the better the approximation; however, it is also true that the batch will take longer to processes and will still result in only one update.
For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluating/prediction).

Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
When using evaluation_data or evaluation_split with the fit method of Keras models, evaluation will be run at the end of every epoch.
Within Keras, there is the ability to add callbacks specifically designed to be run at the end of an epoch. Examples of these are learning rate changes and model checkpointing (saving).

"""

train_DIR = "/modules/cs342/Assignment2/Data/train/" #"/modules/cs342/Assignment2/Data/sampleTrain/"
#test_DIR = "/modules/cs342/Assignment2/Data/test/"

FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 128  #256
COLS = 128 #256
CHANNELS = 3


def get_images(fish):
    """Load files from train folder"""
    fish_dir = train_DIR+'{}'.format(fish)
    images = [fish+'/'+im for im in os.listdir(fish_dir)]
    return images

def read_image(src):
    """Read and resize individual images"""
    im = io.imread(src)
    im = transform.resize(im, (ROWS,COLS), order=3)
    return im


files = []
y_all = []

for fish in FISH_CLASSES:
    fish_files = get_images(fish)
    files.extend(fish_files)
    
    y_fish = np.tile(fish, len(fish_files))
    y_all.extend(y_fish)
    print("{0} photos of {1}".format(len(fish_files), fish))
    
X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

for i, im in enumerate(files): 
    X_all[i] = read_image(train_DIR+im)
    if i%1000 == 0: print('Processed {} of {}'.format(i, len(files)))

print(X_all.shape)


    
# Encode labels and then split the training and test data  
y_all = LabelEncoder().fit_transform(y_all)
y_all = np_utils.to_categorical(y_all)
X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size = 0.2, random_state = 23, stratify = y_all)



"""" dropout regularization between the fully connected layers. 
Note: I set the epochs to 1 to avoid timing out - change it to around 20.
"""


optimizer = RMSprop(lr=1e-4)
objective = 'categorical_crossentropy'

def center_normalize(x):
    return (x - K.mean(x)) / K.std(x)
  
# Channel row column...
# Input: 100 * 100 images with 3 channels ---> (128,128,3) tensors

model = Sequential()
                          
model.add(Activation(activation = center_normalize, input_shape = (ROWS, COLS, CHANNELS)))

# This applies to 32 cov filts of size 5 * 5 , 
model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='th'))
model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='th'))
model.add(MaxPooling2D( pool_size= (2 , 2), dim_ordering='th')) 

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(FISH_CLASSES)))
model.add(Activation('sigmoid'))

model.compile(loss=objective, optimizer=optimizer)


# Early stopping regualisation to prevent the cnn overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience = 4, verbose=1, mode='auto')        

# Train the model, iterating on the data in batches of 64 samples
#validation data used will be the last 20 % of the data.
model.fit(X_train, y_train, batch_size = 64, nb_epoch = 50 , validation_split = 0.2, verbose = 1, shuffle = True, callbacks = [early_stopping]) # 50 epochs


# calculate predictions
predictions = model.predict(X_all)




