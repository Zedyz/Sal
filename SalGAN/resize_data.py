import os
import cv2
from tqdm import tqdm
from utils import *
from paths import *

#Create paths if doesn't exist
createPath(resizedImagesTrain)
createPath(resizedImagesVal)
createPath(resizedImagesTest)
createPath(resizedMapsTrain)
createPath(resizedMapsVal)

#Getting file names without the file format extension
train_names = [k[:-4] for k in os.listdir(imagesTrain)]
val_names = [k[:-4] for k in os.listdir(imagesVal)]
test_names = [k[:-4] for k in os.listdir(imagesTest)]


#Resizing Training Data
for filename in tqdm(train_names,desc='Resizing Training Data  '):
    fullImagePath = imagesTrain + filename + '.png'
    fullResizeImagePath = resizedImagesTrain + filename + '.png'
    fullMapPath = mapsTrain + filename + '.png'
    fullResizeMapPath = resizedMapsTrain + filename + '.png'
    try:
        resizedImage = resizeImage(fullImagePath)
        resizedMap = resizeImage(fullMapPath)
        cv2.imwrite(fullResizeImagePath,resizedImage)
        cv2.imwrite(fullResizeMapPath,resizedMap)
    except Exception as e:
        print(e)
        break

print("Training Data - Finished")

#Resizing Validation Data
for filename in tqdm(val_names,desc='Resizing Validation Data'):
    fullImagePath = imagesVal + filename + '.png'
    fullResizeImagePath = resizedImagesVal + filename + '.png'
    fullMapPath = mapsVal + filename + '.png'
    fullResizeMapPath = resizedMapsVal + filename + '.png'
    try:
        resizedImage = resizeImage(fullImagePath)
        resizedMap = resizeImage(fullMapPath)
        cv2.imwrite(fullResizeImagePath,resizedImage)
        cv2.imwrite(fullResizeMapPath,resizedMap)
    except Exception as e:
        print(e)
        break

print("Validation Data - Finished")

#Resizing Test Data
for filename in tqdm(test_names,desc='Resizing Test Data      '):
    fullImagePath = imagesTest + filename + '.png'
    fullResizeImagePath = resizedImagesTest + filename + '.png'
    try:
        resizedImage = resizeImage(fullImagePath)
        cv2.imwrite(fullResizeImagePath,resizedImage)
    except Exception as e:
        print(e)
        break
        
print("Test Data - Finished")
    
