 
import os
import glob
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
#import theano as th
import cv2
import numpy as np
import pandas as pd
from sklearn import datasets,preprocessing, model_selection
from sklearn.preprocessing import LabelEncoder,StandardScaler
from skimage import io,transform,feature, exposure, filters
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#import tensorflow as tf
#from keras.models import Sequential
#from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
#from keras.utils import np_utils
#from keras.callbacks import EarlyStopping
#from keras.optimizers import RMSprop, Adam
#from keras import backend as keras
import subprocess


def load_training_set():
  x = []
  y = []
  types = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
  for fish in types:
    files = glob.glob('/../modules/cs342/Assignment2/Data/train/' + fish + '/*.jpg')
    for images in files:
      grey_image = io.imread(images, as_grey = True)
      resize_image = transform.resize(grey_image, (128 , 128))
      x += [resize_image]
      y += [fish]
  return np.array(x),y

def load_test_set1():
  x = []
  files = glob.glob('/../modules/cs342/Assignment2/Data/test/*.jpg')
  
  for images in files:
    grey_image = io.imread(images, as_grey = True)
    resize_image = transform.resize(grey_image, (128 , 128))
    x += [resize_image]  
  return np.array(x)


def label_binarizer(target):
  lb = preprocessing.LabelBinarizer(neg_label = 0, pos_label = 1, sparse_output = False)
  result = lb.fit_transform(target)
  return result

def flatten(features):
  result = []
  for n in features:
    result += [np.ravel(n)]
  return result

def normalize(features):
  n = preprocessing.Normalizer()
  result = n.fit_transform(features)
  return result



def hog(features):
  hog_images = []
  for image in features:
    arr = feature.hog(image, orientations= 8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True)
    hog_images += [arr]
  return hog_images  
    
def model():
  x_train, y_train = load_training_set()
  x_train = hog(x_train)
  x_train = flatten(x_train)
  y_train = label_binarizer(y_train)
  x_test1 = load_test_set1()



# combineTestSet = x_test1 + x_test2
#  combineTestSet = combineTestSet.append(x_test1)
#  combineTestSet = combineTestSet.append(x_test2)
#  combineTestSet = x_test1.append(x_test2 ) # axis = 0   #np.concatenate
#  print (combineTestSet)

 
  x_test1 = hog(x_test1)
  x_test1 = flatten(x_test1)
 
# combineTestSet = otsu(combineTestSet)
# combineTestSet = flatten(combineTestSet)  
  #mlp = MLPClassifier(alpha = 1 , activation = 'relu' , hidden_layer_sizes = (256,) , solver = 'sgd' , learning_rate_init = 0.04 , momentum = 0.9 , verbose = True)
  mlp = MLPClassifier( hidden_layer_sizes = (16,)  , activation = 'tanh' , solver = 'adam', verbose = True, learning_rate_init = 0.001 )
  mlp.fit(x_train,y_train)
  y1 = mlp.predict_proba(x_test1)
  df = pd.DataFrame(y1 ,columns = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
  return df


#'alpha': 1, 'activation': 'tanh', 'solver': 'adam', 'learning_rate_init': 0.001, 'hidden_layer_sizes': (16,)



m = model()
m.to_csv('HoGUMLPSubmission1.csv', index = False)

