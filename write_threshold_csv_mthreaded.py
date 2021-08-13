##
## Generates match data of which in other scripts the ideal threshold can be calculated
## For every image combination, match the images, keep all matches (good and bad), and save them to threshold.csv
##

from IPython.display import display
# from bs4 import BeautifulSoup
from urllib.parse import urlparse
#from google.colab import drive
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import csv



BASE = "."

TEST_BASE = BASE + '/ehm_dataset'

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

def serialize_keypoints(keyps):
  simplified = []

  for keyp in keyps:
    simplified.append((
      keyp.pt, 
      keyp.size, 
      keyp.angle, 
      keyp.response, 
      keyp.octave, 
      keyp.class_id
    ))

  return pickle.dumps(simplified, protocol=0)

def deserialize_keypoints(simplified):
  keypoints = []

  unpickled = pickle.loads(simplified)

  for simp in unpickled:
    keypoint = cv.KeyPoint(x=simp[0][0],y=simp[0][1],_size=simp[1], _angle=simp[2], _response=simp[3], _octave=simp[4], _class_id=simp[5])
    
    keypoints.append(keypoint)

  return keypoints



import csv
import pandas
import sys
import threading
import concurrent.futures
import logging
logging.root.setLevel(logging.DEBUG)
import time


db_fileName = TEST_BASE + '/data-keypoints.csv'
keypoints_df = pandas.read_csv(db_fileName)

# Adjust contains a lookup table for matching images
db_fileName2 = TEST_BASE + '/adjust.csv'
db_file2 = open(db_fileName2, 'r', encoding="latin")
reader2 = csv.DictReader(db_file2)

# The output file
db_fileName3 = TEST_BASE + '/threshold.csv'
db_file3 = open(db_fileName3, 'a')
fieldnames = ['url_needle', 'matchTime', 'url_haystack', 'keypoints', 'relatie', 'match']
writer = csv.DictWriter(db_file3, fieldnames=fieldnames)
writer.writeheader()

# Write a single line to the threshold file
write_lock = threading.Lock()
def write_row(needle, haystack, matchTime, keypoints, relatie, match):
  print('Writing csv: ', needle, haystack)
  with write_lock:
    writer.writerow({
        'url_needle' : needle,  
        'url_haystack': haystack,
        'matchTime': matchTime,
        'keypoints' : keypoints, 
        'relatie': relatie,
        'match' : match
    })
  print('Done writing csv')




sift = cv.SIFT_create()
csv.field_size_limit(sys.maxsize)


# Creates match data for a given image match.
def thread_function(images):
    startTime = time.time()
    #keyp_1 = []
    #keyp_2 = []
    desc_1 = None
    desc_2 = None
    
    logging.debug('Start search data:' + images['img1']+ ' & ' + images['img2'])

    try:
      desc_1 = keypoints_df.loc[keypoints_df['url'] == images['img1']].iloc[0].at['descriptors']
      desc_2 = keypoints_df.loc[keypoints_df['url'] == images['img2']].iloc[0].at['descriptors']
    except:
      logging.error('NOT FOUND' + images['img1'] + ' & ' + images['img2'])
      return

    logging.debug('Done searching through index')

    # db_file.close()

    if desc_1 is None or desc_2 is None:
        logging.debug('Not in current dataset')
        return
    
    logging.debug('Matching')
    bf = cv.BFMatcher()
    try:
        matches = bf.knnMatch(deserialize_descriptors(desc_1), deserialize_descriptors(desc_2), k=2) 
    except Exception as e:
        logging.debug('MATCH ERROR: ')
        logging.debug(e)
        return
    good = []
    match_i = []

    logging.debug('Matches: ' + str(len(matches)))

    for m,n in matches:
        dist1 = m.distance
        dist2 = n.distance
        rela = dist1/dist2
        match_i.append(rela)
        if m.distance < 0.6132918588356167*n.distance:
            good.append([m])

    sorteer = np.sort(match_i)
    # print(type(match_i))
    # print(match_i)
    # vaag = np.array(match_i,dtype=float)
    # writer = csv.DictWriter(db_file, fieldnames=fieldnames)

    write_row(images['img1'], images['img2'], time.time()-startTime, len(matches), serialize_descriptors(sorteer[:500]), images['match'])
    logging.debug('Thresholding data written for '+ images['img1'] +' '+ images['img2'])


logging.debug('Starting threads...')
logging.basicConfig(format="%(thread)d - %(asctime)s:\t%(message)s", datefmt='%d,%H:%M:%S.%f')
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(thread_function, reader2)

print('End of script')