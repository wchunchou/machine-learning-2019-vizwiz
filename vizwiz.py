#!/usr/bin/env/python3.6
# Lab assignment 4 
# - Working with the VizWiz dataset
import os
import json
from pprint import pprint
import requests

base_url = 'https://ivc.ischool.utexas.edu/VizWiz/data'
img_dir = '%s/Images/' %base_url

# Retrieve file from ULR and store it locally
split = 'train'
train_file = '%s/Annotations/%s.json' %(base_url,split)
train_data = requests.get(train_file, allow_redirects=True)
print(train_file)

# Read the local file
num_train_VQs = 4
training_data = train_data.json()
for vq in training_data[0:num_train_VQs]:
    image_name = vq['image']
    question = vq['question']
    label = vq['answerable']
    print(image_name)
    print(question)
    print(label)
    
split = 'val'
val_file = '%s/Annotations/%s.json' %(base_url, split)
val_data = requests.get(val_file, allow_redirects=True)
print(val_file)

numValVQs = 3
validation_data = val_data.json()
for vq in validation_data[0:numValVQs]:
    image_name = vq['image']
    question = vq['question']
    label = vq['answerable']
    print(image_name)
    print(question)
    print(label)