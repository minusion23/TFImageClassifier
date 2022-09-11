#Import all necessary modules
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import logging
import json
#imported to get rid of CUDA warnings being an artificat of the model training on GPU
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from PIL import Image

#Build a Parser to get mandatory and optional arguments

parser = argparse.ArgumentParser(
    description='This is an Image Classifier Programme',
)
parser.add_argument('im_path', action="store",default=False)
parser.add_argument('model_ref', action="store")
parser.add_argument('--top_k', action="store", dest="top_k", type=int)
parser.add_argument('--label_map', action="store",dest="label_map")

#Unpack arguments
args = parser.parse_args()
model_ref = args.model_ref
im_path = args.im_path
top_k = args.top_k
label_map = args.label_map

#Load the model for inference
reloaded_model = tf.keras.models.load_model(model_ref, custom_objects={'KerasLayer':hub.KerasLayer},compile = False)

#Account for optional arguments

if label_map != None:
    with open(label_map, 'r') as f:
        class_names = json.load(f)
if args.top_k == None:
    top_k = 1

def process_image(image):
    # Function used to prepare the Image for inference in accordance with the input layer
    
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image /= 255
    
    return image.numpy()

def predict(image_path, model, top_k = 1):
    # Prepare the image
    
    image = Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)
    
    #Expand the image shape to match the model requirements
    
    image = np.expand_dims(image, axis = 0)
    
    # Predict the labels
    ps = model.predict(image)
    labels = np.argpartition(ps,-top_k).squeeze()[-top_k:]
    predictions = ps.squeeze()[labels]
    
    #Create a list with predictions and related probabilities
    
    lab_list = list()
    for idx, item in enumerate(labels):
        item += 1
        if label_map != None:      
            item = class_names[str(item)]
        pred = round(predictions[idx],4)

        pred_str="{:.4f}".format(pred)
        final_item = str(item) + " - " + pred_str
        lab_list.append(final_item)
        lab_list.reverse()
       
    print("Most likely image classes are {}".format(lab_list))

predict(im_path, reloaded_model,top_k)

