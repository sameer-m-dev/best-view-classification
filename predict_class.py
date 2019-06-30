from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pandas as pd
import argparse
import imutils
import pickle
import cv2
import os
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import urllib.request
import argparse


# Remove if argparse gives error
parser = argparse.ArgumentParser(description='Get CSV File.')
parser.add_argument('csvfile', type=argparse.FileType('r'), help='Input CSV File')
args = parser.parse_args()
data_list = pd.read_csv(args.csvfile)
# Remove if argparse gives error


"""
Uncomment the following line in case argparse gives error or does not work

data_list = pd.read_csv('Data/Internship_data.csv')
"""


labels = ['backstrap_BV',
 'backstrap_NBV',
 'buckle_BV',
 'buckle_NBV',
 'hook&look_BV',
 'hook&look_NBV',
 'lace_up_BV',
 'lace_up_NBV',
 'slip_on_BV',
 'slip_on_NBV',
 'zipper_BV',
 'zipper_NBV']

vgg_model = VGG19(weights='imagenet', include_top=False)

x = vgg_model.output
x = GlobalAveragePooling2D()(x)

# add fully-connected layer
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)

# add output layer
predictions = Dense(12, activation='softmax')(x)

model = Model(inputs=vgg_model.input, outputs=predictions)
model.load_weights('Data/fine_tune_shoes_multiclass.best.hdf5')

def identify_type_submission(images , labels):
    class_link = pd.DataFrame(columns = ['predicted_class', 'best_view_image'])
    for i in range(0,1):
      best_image_index = 0
      type_score = dict()
      image_list = []
      best_image_link = []
      j = 0
      for view in ['view_1','view_2','view_3','view_4','view_5']:
        try:
          url_response = urllib.request.urlopen(images[view][i])
          img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
          image = cv2.imdecode(img_array, -1)
          image_list.append(image)
          best_image_link.append(images[view][i])
          image = cv2.resize(image, (128,128))
          image = img_to_array(image)
          image = np.expand_dims(image, axis=0)
          proba = model.predict(image)[0]
          idxs = np.argsort(proba)[::-1][:1]
          type_score[j] = {proba[idxs[0]] : labels[idxs[0]]}
          j = j+1
        except:
          continue
          
      best_image = dict()
      count = 0
      for key,i in type_score.items() :
          type_image  = list(i.values())[0]
          score = list(i.keys())[0]
          if type_image.split('_')[1] == 'BV':
              best_image[count] = {score : type_image.split('_')[0]}
          count = count + 1
          
      type_of_footwear = ''
      if len(best_image) == 1:
        for key,value in best_image.items():
              type_of_footwear = list(value.values())[0]
        best_image_index = list(best_image.keys())[0]
        class_link = class_link.append(pd.Series([type_of_footwear, best_image_link[best_image_index]], index=class_link.columns), ignore_index=True)

      if len(best_image) > 1 :
        max_score = 0
        best_image_index = 0
        for key,value in best_image.items():
          if list(value.keys())[0] > max_score:
            max_score = list(value.keys())[0]
            best_image_index = key
            type_of_footwear = list(value.values())[0]
        class_link = class_link.append(pd.Series([type_of_footwear, best_image_link[best_image_index]], index=class_link.columns), ignore_index=True)

      if len(best_image) == 0:
        max_score = 0
        best_image_index = 0
        for key,value in type_score.items():
          if list(value.keys())[0] > max_score:
            max_score = list(value.keys())[0]
            best_image_index = key
            type_of_footwear = list(value.values())[0].split('_')[0]
        class_link = class_link.append(pd.Series([type_of_footwear, best_image_link[best_image_index]], index=class_link.columns), ignore_index=True)
    return class_link


predicted_data_list = identify_type_submission(data_list, labels)

predicted_data_list.loc[predicted_data_list['predicted_class'] == 'slip', 'predicted_class'] = 'slip_on'
predicted_data_list.loc[predicted_data_list['predicted_class'] == 'lace', 'predicted_class'] = 'lace_up'

new_data = pd.concat([data_list, predicted_data_list], axis=1)
new_data.to_excel('Internship_Data-Predicted(Final).xlsx', index=False)

