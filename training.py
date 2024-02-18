import cv2
from cvzone.HandTrackingModule import HandDetector #detector used to detect the hands
import numpy as np
import math
import time
cap = cv2.VideoCapture(0)   #to capture the video 
detector = HandDetector(maxHands=1) #given to detect only one hand
offset = 20
imgSize = 300
folder = "Data/ILoveYou"    #should change the folder name while training with images based on the required sign
counter = 0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']  #parameters for the bondary box
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255 #to get a white image window to dsiplay our cropped image
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset] #cropping the image with starting and ending parameters
        imgCropShape = imgCrop.shape
        aspectRatio = h / w # setting up the height and width of the image to provide it to a classifier
        #the images are centered by using the following if else loops which makes the classifier easier to read the images 
        if aspectRatio > 1: 
            k = imgSize / h
            wCal = math.ceil(k * w) #ceil makes sure the answer alwyas rounded off to higher value
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2) #finding the gap in the white image to push the image 
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:   #condition when width is greater than height
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"): #press s and the images will be saved in the given folder
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)





    
