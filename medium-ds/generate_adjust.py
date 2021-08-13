#!/usr/bin/python3

import csv
import os

HAYSTACKPATH = '_haystack'
NEEDLEPATH = '_needle'
ADJUSTPATH = './adjust.csv'

db_file3 = open(ADJUSTPATH, 'a')
fieldnames = ['img1', 'img2', 'match']
writer = csv.DictWriter(db_file3, fieldnames=fieldnames)
writer.writeheader()

def write_row(data):
    writer.writerow(data)

allowedExt = [
    'jpg',
    'jpeg',
    'png'
]

for needleRoot, needleDirs, needleFiles in os.walk(NEEDLEPATH):
    needleDirName = needleRoot.split('/')[-1]

    for needleFile in needleFiles:
        if needleFile.split('.')[-1].lower() not in allowedExt:
            print('needle ext', needleFile.split('.')[-1])
            print('skipping needle, wrong ext: ', needleFile)
            continue

        for haystackRoot, dirs, haystackFiles in os.walk(HAYSTACKPATH):
            haystackDirName = haystackRoot.split('/')[-1]

            for haystackFile in haystackFiles:
                if haystackFile.split('.')[-1].lower() not in allowedExt:
                    print('haystack ext', haystackFile.split('.')[-1])
                    print('skipping haystack, wrong ext: ', haystackFile)
                    continue

                isMatch = needleDirName == haystackDirName

                data = {
                    'img1': os.path.join(needleRoot, needleFile),
                    'img2': os.path.join(haystackRoot, haystackFile),
                    'match': '1' if isMatch else '0'
                }
                write_row(data)

                print(data)