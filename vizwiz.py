# Lab assignment 4 foundation
# - Working with the VizWiz dataset
import os
import json
from pprint import pprint
import requests

base_url = 'https://ivc.ischool.utexas.edu/VizWiz/data'
img_dir = '%s/Images/' %base_url
print(img_dir)