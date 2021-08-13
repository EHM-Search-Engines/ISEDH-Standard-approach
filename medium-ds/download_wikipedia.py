

import numpy as np
import requests
import cv2 as cv
import io
from PIL import Image

def getWikiImageUrls(nameFrom='', batchSize=10):
  wikiApi = "https://en.wikipedia.org/w/api.php"
  response = requests.get(wikiApi,params={"action":"query","format":"json","list":"allimages","aifrom":nameFrom,"ailimit":str(batchSize)}).json()

  nextName = response['continue']['aicontinue'] if 'continue' in response else None
  results = {item['name']:item['url'] for item in response['query']['allimages']}
  return results, nextName
  
def getImageFromUrl(url):
    isAbsolute = True if '//' in url else False
    try:
    # image = Image.open(BytesIO(requests.get(url).content)) if isAbsolute else Image.open(url)
        if isAbsolute:
            res = requests.get(url, stream=True)
            if res.status_code != 200:
                raise 'Non 200 status code'

            # image = Image.open(io.BytesIO(res.content))

            nparr = np.fromstring(res.content, np.uint8)
            if nparr is None:
                raise 'nparr is None'

            image = cv.imdecode(nparr, cv.IMREAD_COLOR)

            if image is None:
                raise 'None!'
            #image = cv.imread(BytesIO(requests.get(url).content))
        else: 
            image = cv.imread(url)
    except Exception as e:
        print('Image retrieval error: ', e)
        image = None
    return image



BATCH_SIZE = 40
 
nextName = ''
i = 0
n = 0
while i < 1000: # 5000 batches of 10
# while nextName is not None:
  i += 1
  sourceImageUrlDict, nextName = getWikiImageUrls(nextName, BATCH_SIZE)

  for sourceId in sourceImageUrlDict:
    sourceImageUrl = sourceImageUrlDict[sourceId]
    print('Downloading image: ', sourceImageUrl)
    sourceImage = getImageFromUrl(sourceImageUrl)
    if sourceImage is None:
      continue

    targetPath = './_haystack/wikipedia/'+str(sourceId)
    print('\ttoPath: ', targetPath)
    cv.imwrite(targetPath, sourceImage)
