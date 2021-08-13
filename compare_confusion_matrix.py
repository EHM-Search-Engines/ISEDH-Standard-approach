##
## This script generates the confusion matrix using the bruteforce matcher
## Input is a file containing pre-extracted image features. Output is a confusion matrix 
##

from PIL import Image
from io import BytesIO
from IPython.display import display
# from bs4 import BeautifulSoup
from urllib.parse import urlparse
#from google.colab import drive
import requests
import os
import math
import glob
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import csv


BASE = "."

TEST_BASE = BASE + '/grande'

#drive.mount('/content/drive', force_remount=True)



# Serialization functies

import pickle
import cv2 as cv
import codecs


def serialize_descriptors(descr):
  return codecs.encode(pickle.dumps(descr), "base64").decode()
  # return pickle.dumps(descr, protocol=0) # Protocol=0 is printable ascii
def deserialize_descriptors(ser):
  return pickle.loads(codecs.decode(ser.encode(), "base64"))


import csv
import sys

y_pred_grande = []
y_true_grand = []

db_fileName = TEST_BASE + '/data-keypoints.csv'
db_file = open(db_fileName, 'r')
reader = csv.DictReader(db_file)
db_fileName2 = TEST_BASE + '/adjust.csv'
db_file2 = open(db_fileName2, 'r')
reader2 = csv.DictReader(db_file2)
sift = cv.SIFT_create()
csv.field_size_limit(sys.maxsize)

for images in reader2:
    #keyp_1 = []
    #keyp_2 = []
    desc_1 = []
    desc_2 = []
    
    print('Start search data:', images['img1'], ' &', images['img2'])
    db_file.seek(0)
    for data in reader:
        if images['img1'] == data['url']:
            desc_1 = data
        elif images['img2'] == data['url']:
            desc_2 = data
    
    # if desc_1 is None or desc_2 is None:
    #     print('Not in current dataset')
    #     continue
    
   
    bf = cv.BFMatcher()
    try:
        matches = bf.knnMatch(deserialize_descriptors(desc_1['descriptors']), deserialize_descriptors(desc_2['descriptors']), k=2) 
    except:
        print('Deze was dus te groot, of niet in de set')
        continue
    good = []
    # match_i = []

    for m,n in matches:
        # dist1 = m.distance
        # dist2 = n.distance
        # rela = dist1/dist2
        # match_i.append(rela)
        if m.distance < 0.6132918588356167*n.distance:
            good.append([m])

    # sorteer = np.sort(match_i)
    # perfPrint('filter matches')
    # print(type(match_i))
    # print(match_i)
    # vaag = np.array(match_i,dtype=float)
    # writer = csv.DictWriter(db_file, fieldnames=fieldnames)
    # writer.writerow({
    #     'url_needle' : images['img1'],  
    #     'url_haystack': images['img1'], 
    #     'keypoints' : len(matches), 
    #     'relatie': serialize_descriptors(sorteer[:300])
    # })

    perc = len(good)/len(matches)

    if perc >= 0.008: 
        print('Found a match!')
        # print(row['url'], ' and', row2['url'])
        print('Number of keypoints:', len(matches))
        print('Number of qualitive matches', len(good))
        print('Percentage of good matches: ', perc*100, '%')
        y_pred_grande.append(1) 
    else:
        y_pred_grande.append(0) 

    if images['match'] == '1':
        y_true_grand.append(1)
    else:
        y_true_grand.append(0)
    print('lengths:', len(y_pred_grande), ' &', len(y_true_grand))



from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# db_fileName = TEST_DATA + '/results_orb_notated.csv'
# db_file = open(db_fileName, 'r')
# reader = csv.DictReader(db_file)

def calculate_score(y_true, y_pred):
  
  matrix = metrics.confusion_matrix(y_true, y_pred)
  acc_score = metrics.accuracy_score(y_true, y_pred)

  #Results

  plt.figure(figsize=(4,4))
  sns.heatmap(matrix, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'Blues');
  plt.ylabel('Actual label');
  plt.xlabel('Predicted label');
  all_sample_title = 'Accuracy Score: {:.3f}'.format(acc_score)
  plt.title(all_sample_title, size = 10);


calculate_score(y_true_grand, y_pred_grande)