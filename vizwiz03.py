#!/usr/bin/env/python3.6
# Lab assignment 4 
# - Working with the VizWiz dataset
import os
import json
from pprint import pprint
import requests
import numpy as np
import scipy as sp
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from skimage.transform import resize
from skimage import io
from skimage import color
from skimage import feature
from skimage import measure
import scipy.ndimage as ndi

from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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


# Transform image
print("extracting image and sentence feature...")

def extract_image_features(image_url):
    # Read image
    image = io.imread(image_url)

    # Pre-process image
    width = 255
    height = 255
    image = resize(image, (width, height), mode='reflect') # Ensuring all images have the same dimension
    greyscale_image = color.rgb2gray(image) # Restricting the dimension of our data from 3D to 2D
    
    # Extract featuresdes
    #featureVector = feature.hog(greyscale_image, orientations=9, pixels_per_cell=(8,8), cells_per_block=(1,1), block_norm='L2-Hys')
    featureVector = feature.daisy(greyscale_image, step=4, radius=15, rings=3, histograms=8, orientations=8, normalization='l2', sigmas=None, ring_radii=None, visualize=False)
    return featureVector

# Extract features to describe the questions
import nltk
nltk.download('punkt')

def extract_language_features(question):
    #print(question)
    question = question.lower()
    #print(question)
    
    words = nltk.word_tokenize(question) # for slight variant call question.split() 
    num_words = len(words)
    num_unique_words = len(set(words))

    featureVector = [num_words, num_unique_words]
    
    # Options for additional features to use:
    #partsOfSpeechTag = nltk.pos_tag(words)

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
X_train, y_train, X_test, y_test =[],[],[],[]

num_VQs = 200
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
    #print(image_feature[:5])
    #print(image_feature.shape)
    
    # PLACEHOLDER: Concatenate the question and image features
    multimodal_features = np.concatenate((question_feature, image_feature), axis=None)
    #print(multimodal_features[:7])
    #print(multimodal_features.shape)
    X_train.append(multimodal_features)
    y_train.append(vq['answerable'])

num_VQs_testing = 30
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
    
    # PLACEHOLDER: Concatenate the question and image features
    multimodal_features = np.concatenate((question_feature, image_feature), axis=None)
    #print(multimodal_features[:7])
    #print(multimodal_features.shape)
    X_test.append(multimodal_features)
    y_test.append(vq['answerable'])

print("extra features sucessfully!")

print("training the model...")

# Transform scale of data
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)    


#gaussian_model = GaussianNB()
#gaussian_model.fit(X_train, y_train)
#y_predictedNB = gaussian_model.predict(X_test)
#test_accuracy = gaussian_model.score(X_test, y_test)
#print("test_accuracy",test_accuracy)

mlp = MLPClassifier(max_iter=20, random_state=42, verbose=True, hidden_layer_sizes=(200,200,200,200,100))
mlp.fit(X_train, y_train)
print("evalute model")
print("Accuracy on the test set: {:.2f}".format(mlp.score(X_test, y_test)))
print("Activation function used at the output layer: %s" % mlp.out_activation_)
print("Number of outputs at the output layer: %f" % mlp.n_outputs_)
print("List predicted classes at the output layer: %s" % mlp.classes_)
print("Training set loss: %s" % mlp.loss_)


#svm = SVC(kernel="poly", degree=6, gamma=0.1)
#svm.fit(X_train_scaled, y_train)
#test_score = svm.score(X_test_scaled, y_test)
#print("Test set score: {:.2f}".format(test_score))
#y_predicted=svm.predict(X_test_scaled)
#from sklearn import metrics
#print(metrics.classification_report(y_predicted, y_test))



f = open("demofile.txt", "a")
f.write("Now the file has one more line!")