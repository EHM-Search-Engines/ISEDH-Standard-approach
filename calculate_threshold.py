##
## Given a extracted feature set of the image database, find the ideal threshold optimizing for precision, recall and balanced accuracy
## Input is threshold.csv, output is the parameters for an optimum precision, recall and balanced accuracy
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

def threshold_performance(th):
    
    
    balance_opt = [0, 0, 0, 0, 0]
    precision_opt = [0, 0, 0, 0, 0]
    recall_opt = [0, 0, 0, 0, 0]
    best_balance = 0
    opt_precision = 0
    db_fileName = TEST_BASE + '/threshold.csv'
    #db_fileName = BASE + '/Datasets/test_data/data_dist_labels.csv'
    db_file = open(db_fileName, 'r')
    reader = csv.DictReader(db_file)

    for min_match in range (1,30):
        db_file.seek(0)
        rela_good = np.array([])
        rela_worse = np.array([])

        #perfPrint(None)
        for row in reader:
            try:
                matches_need = math.ceil(th*int(row['keypoints']))
            except:
                continue
            
            
            if matches_need >= 500:
                matches_need = 499
                # continue
            
            if matches_need < min_match:
                matches_need = min_match

            
            relation = deserialize_descriptors(row['relatie'])
            #print(matches_need)
            if row['match'] == '1':
                rela_good = np.append(rela_good, relation[matches_need])
            else:
                rela_worse = np.append(rela_worse, relation[matches_need])

        #perfPrint('Done reading threshold_file')
                
        rela_good = np.sort(rela_good)
        rela_worse = np.sort(rela_worse)
        
        max_th = max(rela_good)
        min_th = min(rela_worse)
        #print(rela_worse)
        #print(min_match, th_point)
            # (((TP/(TP+FN)+(TN/(TN+FP))) / 2
            # TP/TP+FP
    
        if max_th < min_th:
            print(th, ' is a good threshold')
            opt_th = min_th
        else:
            for dec_th in rela_worse:
                #perfPrint(None)
                FN = 0
                FP = 0
                TN = 0
                TP = 0

                ## Replaced with facter numpy code below this commented block. Left here for reference
                # for i in rela_good:
                #     if i > dec_th:
                #         FN = FN + 1
                # for j in rela_worse:
                #     if j < dec_th:
                #         FP = FP + 1
                
                FN = np.count_nonzero(np.where(rela_good > dec_th))
                FP = np.count_nonzero(np.where(rela_worse < dec_th))

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
                # print(balanced_acc)
                # print(best_balance)
                # print('The optimal threshold ', precision, th)
                # print('The optimal threshold ', recall, min_match)
                # print('The optimal threshold ', balanced_acc)
                if precision > precision_opt[4]:
                    precision_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]
                elif precision == precision_opt[4] and recall >= precision_opt[3] and balanced_acc >= precision_opt[0]:
                    precision_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]

                if balanced_acc > balance_opt[0]:
                    balance_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]
                elif balanced_acc == balance_opt[0] and recall >= balance_opt[3] and precision >= balance_opt[4]:
                    balance_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]

                if recall > recall_opt[3]:
                    recall_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]
                elif recall == recall_opt[3] and precision >= recall_opt[4] and balanced_acc >= recall_opt[0]:
                    recall_opt = [balanced_acc, dec_th, min_match, recall, precision, th, TN, TP, FN, FP]
                #perfPrint('for dec_th in rela_worse')

    

    return precision_opt, balance_opt, recall_opt

balance_optimal = [0, 0, 0, 0, 0]
precision_optimal = [0, 0, 0, 0, 0]
recall_optimal = [0, 0, 0, 0, 0]

db_fileName = TEST_BASE + '/PRC.csv'
print("Precision recall data at: ", db_fileName)
db_file = open(db_fileName, 'a', newline='')
fieldnames = ['th', 'precision', 'balance', 'recall']
writer = csv.DictWriter(db_file, fieldnames=fieldnames)
writer.writeheader()


for th in np.linspace(0, 0.15, 200):
    # th = i * 0.001
    precision, balance, recall = threshold_performance(th)
    
    # print('precision', precision)
    # print('balance', balance)
    # print('recall', recall)
    writer.writerow({
            'th': th, 
            'precision': serialize_descriptors(precision), 
            'balance': serialize_descriptors(balance),
            'recall' : serialize_descriptors(recall)
        })

    if precision[4] > precision_optimal[4]:
        precision_optimal = precision 
        #print('current new best precision', precision)
    elif precision[4] == precision_optimal[4] and precision[3] >= precision_optimal[3] and precision[0] >= precision_optimal[0]:
        precision_optimal = precision
        #print('current new best precision', precision)

    if balance[0] > balance_optimal[0]:
        balance_optimal = balance
        #print('current new best balance', balance)
    elif balance[0] == balance_optimal[0] and balance[3] >= balance_optimal[3] and balance[4] >= balance_optimal[4]:
        balance_optimal = balance
        #print('current new best balance', balance)
    
    if recall[3] > recall_optimal[3]:
        recall_optimal = recall
        #print('current new best recall', recall)
    elif recall[3] == recall_optimal[3] and recall[4] >= recall_optimal[4] and recall[0] >= recall_optimal[0]:
        recall_optimal = recall
        #print('current new best recall', recall)
    
writer.writerow({
            'th': 'BESTE', 
            'precision': serialize_descriptors(precision_optimal), 
            'balance': serialize_descriptors(balance_optimal),
            'recall' : serialize_descriptors(recall_optimal)
        })    
print('The optimal threshold for best precision', precision_optimal)

print('The optimal threshold for best recall', recall_optimal)

print('The optimal threshold for best balanced accuracy', balance_optimal)
db_file.close()
print('Done!')
 
#print_avergae_runtimes()
    