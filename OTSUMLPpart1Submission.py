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
      resize_image = transform.resize(grey_image, (256 , 256))
      x += [resize_image]
      y += [fish]
  return np.array(x),y

def load_test_set1():
  x = []
  files = glob.glob('/../modules/cs342/Assignment2/Data/test/*.jpg')
  
  for images in files:
    grey_image = io.imread(images, as_grey = True)
    resize_image = transform.resize(grey_image, (256 , 256))
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



def otsu(X):
  otsu_images = []
  for image in X:
    threshold = filters.threshold_otsu(image)
    binary = image <= threshold
    otsu_images += [binary]
  return otsu_images

    
def main():
  x_train, y_train = load_training_set()
  x_train = otsu(x_train)
  x_train = flatten(x_train)
  y_train = label_binarizer(y_train)
  x_test1 = load_test_set1()


 
  x_test1 = otsu(x_test1)
  x_test1 = flatten(x_test1)
   
  
 

  mlp = MLPClassifier(activation = 'relu' , solver = 'adam', verbose = True)
#  mlp.fit(x_train,y_train)
#  y1 = mlp.predict_proba(x_test1)
  
  
  param_grid = {'hidden_layer_sizes': [(16,), (32,), (64,), (128,)] ,'alpha':[0.01,0.1,0.5,0.75,1] }
  grid_search = model_selection.GridSearchCV(mlp, param_grid = param_grid, cv = 3, scoring= 'log_loss')
  grid_search.fit(x_train, y_train)
  print grid_search.grid_scores_
  print grid_search.best_score_
  print grid_search.best_params_
      
  
  df = pd.DataFrame(y1 ,columns = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
  return df



model = main()
#m.to_csv('OTSUMLPSubmission1.csv', index = False)



#  param_grid = {'hidden_layer_sizes': [(16,), (32,), (64,), (128,)] ,'alpha':[0.01,0.1,0.5,0.75,1] }
#  grid_search = model_selection.GridSearchCV(mlp, param_grid = param_grid, cv = 3, scoring= 'log_loss')
#  grid_search.fit(x_train, y_train)
#  print grid_search.grid_scores_
#  print grid_search.best_score_
# print grid_search.best_params_

