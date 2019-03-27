#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 00:28:23 2019

@author: floydesk
"""

#!/usr/bin/env/python3.6
# Lab assignment 4 
# - Working with the VizWiz dataset
import os
import json
from pprint import pprint
import numpy as np
import requests
import key
import argparse
import cv2
from skimage.transform import resize
from skimage import io
from skimage import color
from skimage import feature
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
import key
subscription_key = key.key()
vision_base_url = 'https://westus.api.cognitive.microsoft.com/vision/v1.0'
vision_analyze_url = vision_base_url + '/analyze?'

print("import data from VizWiz...")
base_url = 'https://ivc.ischool.utexas.edu/VizWiz/data'
img_dir = '%s/Images/' %base_url

# Retrieve file from ULR and store it locally
split = 'train'
train_file = '%s/Annotations/%s.json' %(base_url,split)
train_data = requests.get(train_file, allow_redirects=True)
#print(train_file)
print("import training data successfully")

# Read the local file
num_train_VQs = 4
training_data = train_data.json()
for vq in training_data[0:num_train_VQs]:
    image_name = vq['image']
    question = vq['question']
    label = vq['answerable']
    #print(image_name)
    #print(question)
    #print(label)
    
split = 'val'
val_file = '%s/Annotations/%s.json' %(base_url, split)
val_data = requests.get(val_file, allow_redirects=True)
validation_data = val_data.json()
#print(val_file)
print("import validation data successfully")

split = 'test'
test_file = '%s/Annotations/%s.json' %(base_url, split)
test_data = requests.get(test_file, allow_redirects=True)
testing_data = test_data.json()
print("import testing data successfully")


# Transform image
print("extracting image and sentence feature...")

def extract_image_features(image_url):
    image = io.imread(image_url)
    
    # Pre-process image
    width = 255
    height = 255
    image = resize(image, (width, height), mode='reflect') # Ensuring all images have the same dimension
    greyscale_image = color.rgb2gray(image) # Restricting the dimension of our data from 3D to 2D

    # Extract features
    featureVector = feature.hog(greyscale_image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(1,1), block_norm='L2-Hys')
    
    
    return featureVector

def extract_image_features_2(image_url):
    image = io.imread(image_url)
    # Pre-process image
    width = 255
    height = 255
    image = resize(image, (width, height), mode='reflect') # Ensuring all images have the same dimension
    greyscale_image = color.rgb2gray(image) # Restricting the dimension of our data from 3D to 2D

    # Extract features
    featureVector = feature.canny(greyscale_image, sigma=1.0, low_threshold=None, high_threshold=None, mask=None, use_quantiles=False)
    
    
    return featureVector

def extract_image_features_3(image_url):
    image = io.imread(image_url)
    # Pre-process image
    width = 255
    height = 255
    image = resize(image, (width, height), mode='reflect') # Ensuring all images have the same dimension
    greyscale_image = color.rgb2gray(image) # Restricting the dimension of our data from 3D to 2D

    # Extract features
    featureVector = feature.local_binary_pattern(greyscale_image, P=3, R=3.14, method='default')
    
    
    return featureVector

def extract_image_features_azure(image_url):
  
    # Microsoft API headers, params, etc
    headers = {'Ocp-Apim-Subscription-key': subscription_key}
    params = {'visualfeatures': 'Categories,Color,description,Faces,Tags'}
    data = {'url': image_url}
    
    # send request, get API response
    response = requests.post(vision_analyze_url, headers=headers, params=params, json=data)
    response.raise_for_status()
    data = response.json()
    extracted_data = extract_features(data)
    azfeatures=[]
    azfeatures.append(float(extracted_data['tags'][0]['confidence']))
    #for tags in extracted_data['tags']:
        #azfeatures.append(float(extracted_data['tags'][0]['confidence']))
    azfeatures = np.array(azfeatures)
    return azfeatures 

def extract_features(data):
    return {
        "description": data["description"],
        "tags": data["tags"],
        "black_and_white": data["color"]["isBwImg"],
        "categories":  data["categories"],
        "faces": data["faces"],
        "dominant_color_background": data["color"]["dominantColorBackground"],
        "dominant_foreground_color": data["color"]["dominantColorForeground"],
        "accent_color": data["color"]["accentColor"]
    }

# Extract features to describe the questions
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_language_features(question):
    #print(question)
    question = question.lower()
    #print(question)
    
    words = nltk.word_tokenize(question) # for slight variant call question.split() 
    num_words = len(words)
    num_unique_words = len(set(words))
    partsOfSpeechTags = nltk.pos_tag(words)
    partsOfSpeech = [wordResult[1] for wordResult in partsOfSpeechTags]
    tag_fd = nltk.FreqDist(partsOfSpeech)
    whDeterminerCount = tag_fd['WDT']
    whPronounCount = tag_fd['WP']
    comparativeAdjectiveCount = tag_fd['JJR']
    determinerCount = tag_fd['DT']

    featureVector = [num_words, num_unique_words, whDeterminerCount, whPronounCount, comparativeAdjectiveCount,determinerCount]
    
    # Options for additional features to use:
    

    # https://pythonprogramming.net/natural-language-toolkit-nltk-part-speech-tagging/
    #partsOfSpeechTags = nltk.pos_tag(nltk.word_tokenize(question))
    #partsOfSpeech = [wordResult[1] for wordResult in partsOfSpeechTags]
    #tag_fd = nltk.FreqDist(partsOfSpeech)
    #determinerCount = tag_fd['DT']
    #singularNounCount = tag_fd['NN']
    #pluralNounCount = tag_fd['NNS']
    #prepositionCount = tag_fd['IN']
    #existentialCount = tag_fd['EX']
    #adjectiveCount = tag_fd['JJ']
    #comparativeAdjectiveCount = tag_fd['JJR']
    #superlativeAdjectiveCount = tag_fd['JJS']
    #modalCount = tag_fd['MD']
    #verbCount = tag_fd['VB']
    #verbPastTenseCount = tag_fd['VBD']
    #verbPresentTenseCount = tag_fd['VBG']
    #verbThirdPersonPresentCount = tag_fd['VBZ']
    #whDeterminerCount = tag_fd['WDT']
    #whPronounCount = tag_fd['WP']
    
    return featureVector






X_train, y_train, X_test, y_test, X_challenge =[],[],[],[],[]

num_VQs = 20000
for vq in training_data[0:num_VQs]:
    # Question features
    question = vq['question']
    #print(question)
    question_feature = extract_language_features(question)
    #print(question_feature)
    #print(np.array(question_feature).shape)
    
    # PLACEHOLDER
    image_name = vq['image']
    image_url = img_dir + image_name
    #print(image_url)
    image_feature = extract_image_features(image_url)
    print(image_feature)
    print(image_feature.shape)
    
    edge_features = extract_image_features_2(image_url)
    print(edge_features)
    corner_features = extract_image_features_3(image_url)
    print(corner_features)
    
    # PLACEHOLDER: Concatenate the question and image features
    multimodal_features = np.concatenate((question_feature, image_feature, edge_features, corner_features), axis=None)
    #print(multimodal_features[:7])
    #print(multimodal_features.shape)
    X_train.append(multimodal_features)
    y_train.append(vq['answerable'])

num_VQs_testing = 100
for vq in validation_data[0:num_VQs_testing]:
    # Question features
    question = vq['question']
    #print(question)
    question_feature = extract_language_features(question)
    #print(question_feature)
    #print(np.array(question_feature).shape)
    
    # PLACEHOLDER
    image_name = vq['image']
    image_url = img_dir + image_name
    #print(image_url)
    image_feature = extract_image_features(image_url)
    #print(image_feature[:5])
    #print(image_feature.shape)
    
    edge_features = extract_image_features_2(image_url)
    print(edge_features)
    corner_features = extract_image_features_3(image_url)
    print(corner_features)
    
    # PLACEHOLDER: Concatenate the question and image features
    multimodal_features = np.concatenate((question_feature, image_feature, edge_features, corner_features), axis=None)
    #print(multimodal_features[:7])
    #print(multimodal_features.shape)
    X_test.append(multimodal_features)
    y_test.append(vq['answerable'])

print("extra features sucessfully!")

print("training the model...")



mlp = MLPClassifier(max_iter=20, random_state=21, verbose=True, hidden_layer_sizes=(100,))
mlp.fit(X_train, y_train)
print("evalute model")
print("Accuracy on the test set: {:.2f}".format(mlp.score(X_test, y_test)))
print("Activation function used at the output layer: %s" % mlp.out_activation_)
print("Number of outputs at the output layer: %f" % mlp.n_outputs_)
print("List predicted classes at the output layer: %s" % mlp.classes_)
print("Training set loss: %s" % mlp.loss_)



predict_testing = 199
challenge_result=[]
for vq in testing_data[100:predict_testing]:
    # Question features
    question = vq['question']
    #print(question)
    question_feature = extract_language_features(question)
    #print(question_feature)
    #print(np.array(question_feature).shape)
    
    # PLACEHOLDER
    image_name = vq['image']
    image_url = img_dir + image_name
    #print(image_url)
    image_feature = extract_image_features(image_url)
    
    edge_features = extract_image_features_2(image_url)

    corner_features = extract_image_features_3(image_url)
    
    # PLACEHOLDER: Concatenate the question and image features
    multimodal_features = np.concatenate((question_feature, image_feature, edge_features, corner_features), axis=None)
    #print(multimodal_features[:7])
    #print(multimodal_features.shape)
    X_challenge.append(multimodal_features)

y_challenge = mlp.predict(X_challenge)
print(y_challenge)

for y in y_challenge:
    f = open("challenge.txt", "a")
    f.write("%s," %y)