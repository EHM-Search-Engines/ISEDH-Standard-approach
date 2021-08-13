##
## Calculates the precision recall curves of an image dataset by varying each of the three parameters seperately, keeping the other 2 at their optimum
##


import math
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






import csv
import sys
import cv2 as cv
import numpy as np

#resetPerf()


import pandas

threshFilePath = TEST_BASE + '/threshold.csv'
#db_fileName = BASE + '/Datasets/test_data/data_dist_labels.csv'
threshFile = open(threshFilePath, 'r')
reader = csv.DictReader(threshFile)

def threshold_performance_v2(keypoint_th, min_abs_matches, match_ratio_th):
    threshFile.seek(0)
    next(reader)


    rela_good = np.array([])
    rela_worse = np.array([])

    for row in reader:
        # try:
        matches_need = math.ceil(keypoint_th*int(row['keypoints']))
        # except:
        #     continue
        
        # We only save the best 500 matches in threshold.csv
        if matches_need >= 500:
            matches_need = 499
            # continue
        
        # Minstens min_abs_matches nodig om een match te zijn, voorkomt enkele matches die meer dan 1% worden
        if matches_need < min_abs_matches:
            matches_need = min_abs_matches

        
        relation = deserialize_descriptors(row['relatie'])
        #print(matches_need)
        if row['match'] == '1':
            rela_good = np.append(rela_good, relation[matches_need])
        else:
            try:
                rela_worse = np.append(rela_worse, relation[matches_need])
            except:
                print('rela_worse param: ', rela_worse, matches_need, relation)
                exit()



    rela_good = np.sort(rela_good)
    rela_worse = np.sort(rela_worse)
    
    # max_th = max(rela_good)
    # min_th = min(rela_worse)

    FN = 0
    FP = 0
    TN = 0
    TP = 0

    ## Replaced with the code below, optmised using numpy native functions
    # for i in rela_good:
    #     if i > dec_th:
    #         FN = FN + 1
    # for j in rela_worse:
    #     if j < dec_th:
    #         FP = FP + 1
    
    FN = np.count_nonzero(np.where(rela_good > match_ratio_th))
    FP = np.count_nonzero(np.where(rela_worse < match_ratio_th))

    TP = len(rela_good) - FN
    TN = len(rela_worse) - FP
    # print(TN, FN)
    # print(FP, TP)
    balanced_acc = 0.5*((TP/(TP+FN))+(TN/(TN+FP)))
    try:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
    except:
        precision = 0
        recall = 0

    return precision, recall, balanced_acc, FN, FP, TN, TP



fieldnames = ['keypoint_th', 'min_abs_match', 'match_ratio_th', 'precision', 'recall', 'balanced', 'FN', 'FP', 'TN', 'TP']

optimal_keypoint_th = 0.037
optimal_min_abs_match = 21
optimal_match_ratio_th = 0.5519978894151195

precision, recall, balanced, FN, FP, TN, TP = threshold_performance_v2(optimal_keypoint_th, optimal_min_abs_match, optimal_match_ratio_th)
print('Performance best: p:', precision, 'r:', recall, 'b:', balanced, 'FN:', FN, 'FP', FP, 'TN', TN, 'TP', TP)


# Vary keypoint thresh
db_fileName = TEST_BASE + '/PRC_keypoint_th.csv'
db_file = open(db_fileName, 'a', newline='')
writer = csv.DictWriter(db_file, fieldnames=fieldnames)
writer.writeheader()

for keypoint_th in np.linspace(0, 0.15, 200):
    precision, recall, balanced, FN, FP, TN, TP = threshold_performance_v2(keypoint_th, optimal_min_abs_match, optimal_match_ratio_th)
    
    print('Param for: ', keypoint_th, optimal_min_abs_match, optimal_match_ratio_th, '\n\tperf: ', precision, recall, balanced)

    writer.writerow({
        'keypoint_th': keypoint_th, 
        'min_abs_match': optimal_min_abs_match, 
        'match_ratio_th': optimal_match_ratio_th,
        'precision': precision,
        'recall': recall,
        'balanced': balanced,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'TP': TP
    })

db_file.close()
print('Done!')
 


# Vary min_abs_match
db_fileName = TEST_BASE + '/PRC_min_abs_match.csv'
db_file = open(db_fileName, 'a', newline='')
writer = csv.DictWriter(db_file, fieldnames=fieldnames)
writer.writeheader()

for min_abs_match in range(0, 80):
    precision, recall, balanced, FN, FP, TN, TP = threshold_performance_v2(optimal_keypoint_th, min_abs_match, optimal_match_ratio_th)
    
    print('Param for: ', optimal_keypoint_th, min_abs_match, optimal_match_ratio_th, '\n\tperf: ', precision, recall, balanced)

    writer.writerow({
        'keypoint_th': optimal_keypoint_th, 
        'min_abs_match': min_abs_match, 
        'match_ratio_th': optimal_match_ratio_th,
        'precision': precision,
        'recall': recall,
        'balanced': balanced,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'TP': TP
    })

db_file.close()
print('Done!')



# Vary optimal_match_ratio_th
db_fileName = TEST_BASE + '/PRC_match_ratio_th.csv'
db_file = open(db_fileName, 'a', newline='')
writer = csv.DictWriter(db_file, fieldnames=fieldnames)
writer.writeheader()

for match_ratio_th in np.linspace(0, 1, 300):
    precision, recall, balanced, FN, FP, TN, TP = threshold_performance_v2(optimal_keypoint_th, optimal_min_abs_match, match_ratio_th)
    
    print('Param for: ', optimal_keypoint_th, optimal_min_abs_match, match_ratio_th, '\n\tperf: ', precision, recall, balanced)

    writer.writerow({
        'keypoint_th': optimal_keypoint_th, 
        'min_abs_match': optimal_min_abs_match, 
        'match_ratio_th': match_ratio_th,
        'precision': precision,
        'recall': recall,
        'balanced': balanced,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'TP': TP
    })

db_file.close()
print('Done!')


#print_avergae_runtimes()
    