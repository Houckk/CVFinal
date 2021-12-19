import cv2
import numpy as np
import os
import HandTrackingFeatures as handFeatures

#For drawing on screen - Kenyan
brushThickness = 25
eraserThickness = 100

#for our project we have 4 different images and we toggle between them based on what the user "selects" - Flora
folderPath="Header"
headerImages=os.listdir(folderPath)
customGraphicOverlays=[]
for imagePath in headerImages:
    image=cv2.imread(f'{folderPath}/{imagePath}')
    customGraphicOverlays.append(image)
header=customGraphicOverlays[0]
penColor=(139, 0, 0)

#need to set the height and width of video because our header images above a specific height and width - Flora
cap=cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

handDetector=handFeatures.handDetector(detectionPercentageThreshold=0.65, maxHands=1)
xCoord, yCoord=0, 0
imgCanvas=np.zeros((720, 1280, 3), np.uint8)

#Process we followed to get video working
# 1. Need to import image & flip it so that writing looks normal - Flora
# 2. Need to find hands using hand landmarks - Eric
# 3. Determine which fingers are up so we know whether to draw, select, or change colors - Eric
# 4. Select mode allows users to change colors or select eraser (2 fingers up) - Kenyan
# 5. Draw mode allows users to draw or erase content on the screen (1 finger up) - Kenyan
# 6. New feature not shown in demo in class --> Holding up all five fingers in a tight bunch and then quickly making a fist changes drawing color to a new random color - Flora
#    This is basic hand gesture recognition. In the future making another hand gesture would clear the screen etc but it doesnt work 100% yet
while True:

    # 1 Flora.
    success, img=cap.read()
    img=cv2.flip(img, 1)

    # 2 Eric.
    img = handDetector.findUsersHands(img)
    keypointList = handDetector.findHandPosition(img,draw=False)

    if len(keypointList) != 0:

        # getting the position of the tips of index and middle fingers (keyPoint 8 is index finger and keyPoint 12 is middle finger)
        indexXCoord = keypointList[0][8][1]
        indexYCoord = keypointList[0][8][2]
        middleXCoord = keypointList[0][12][1]
        middleYCoord = keypointList[0][12][2]

        # 3 Eric.
        fingers = handDetector.getFingersThatAreUp()

        # 4 Kenyan.
        if fingers[1] and fingers[2]:
            # To toggle the three main colors & eraser we are checking the coordinate location of the users pointer finger
            if indexYCoord < 125:
                if indexYCoord < 125:
                    if 1000 < indexXCoord < 1250:
                        header = customGraphicOverlays[3]
                        penColor = (0, 0, 0)
                    elif 550 < indexXCoord < 750:
                        header = customGraphicOverlays[2]
                        penColor = (0, 128, 55)
                    elif 300 < indexXCoord < 500:
                        header = customGraphicOverlays[1]
                        penColor = (147, 20, 255)
                    elif 50 < indexXCoord < 250:
                        header = customGraphicOverlays[0]
                        penColor = (139,0,0)
            cv2.rectangle(img, (indexXCoord, indexYCoord - 25), (middleXCoord, middleYCoord + 25), penColor, cv2.FILLED)

        # 5 Kenyan.
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (indexXCoord, indexYCoord), 15, penColor, cv2.FILLED)
            if xCoord == 0 and yCoord == 0:
                xCoord, yCoord = indexXCoord, indexYCoord
            cv2.line(img, (xCoord, yCoord), (indexXCoord, indexYCoord), penColor, brushThickness)
            #pen
            if penColor != (0, 0, 0):
                cv2.line(img, (xCoord, yCoord), (indexXCoord, indexYCoord), penColor, brushThickness)
                cv2.line(imgCanvas, (xCoord, yCoord), (indexXCoord, indexYCoord), penColor, brushThickness)
            #eraser
            else:
                cv2.line(img, (xCoord, yCoord), (indexXCoord, indexYCoord), penColor, eraserThickness)
                cv2.line(imgCanvas, (xCoord, yCoord), (indexXCoord, indexYCoord), penColor, eraserThickness)
            xCoord, yCoord = indexXCoord, indexYCoord

        # 6 Flora.
        penColor=handDetector.detectFist(fingers, penColor)

    # 7 Flora.
    imgGray=cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv=cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img, imgInv)
    img=cv2.bitwise_or(img, imgCanvas)
    img[0:125, 0:1280]=header

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)


