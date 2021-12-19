import cv2
import mediapipe
import math
import random
import numpy as np

class handDetector():
    #Flora - creating the constructor for handDetector
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionPercentageThreshold=0.5, trackingPercentageThreshold=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.modelComplex=modelComplexity
        self.detectionPercentageThreshold=detectionPercentageThreshold
        self.trackingPercentageThreshold=trackingPercentageThreshold
        self.mpHands=mediapipe.solutions.hands
        self.hands=self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionPercentageThreshold, self.trackingPercentageThreshold)
        self.mpDraw=mediapipe.solutions.drawing_utils
        self.fingerTipKeypoints=[4, 8, 12, 16, 20]

    # checks which fingers are up by looking at keypoints - Eric
    def getFingersThatAreUp(self):
        fingersCurrentlyUp = []
        # need to check thumb individually
        if self.keypoints[self.fingerTipKeypoints[0]][1] < self.keypoints[self.fingerTipKeypoints[0]-1][1]:
            fingersCurrentlyUp.append(0)
        else:
            fingersCurrentlyUp.append(1)

        # now checking fingers
        for id in range(1,5):
            if self.keypoints[self.fingerTipKeypoints[id]][2] < self.keypoints[self.fingerTipKeypoints[id]-2][2]:
                fingersCurrentlyUp.append(1)
            else:
                fingersCurrentlyUp.append(0)
        return fingersCurrentlyUp

    # locates the hand position and provides a box around the users fingers (can be seen when holding up two fingers) - Eric
    def findHandPosition(self, img, handNo=0, draw=True):
        perimeterBox = []
        self.keypoints = []
        xCoords = []
        yCoords = []

        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id, kp in enumerate(hand.landmark):
                height, width, c = img.shape
                cXPos, cYPos = int(kp.x * width), int(kp.y * height)
                xCoords.append(cXPos)
                yCoords.append(cYPos)
                self.keypoints.append([id,cXPos,cYPos])
                if draw:
                    cv2.circle(img,(cXPos,cYPos),5,(255,0,255),cv2.FILLED)

        xmin, xmax = min(xCoords), max(xCoords)
        ymin, ymax = min(yCoords), max(yCoords)
        perimeterBox = xmin, ymin, xmax, ymax

        if draw:
            cv2.rectangle(img, (xmin-20,ymin-20), (xmax+20,ymax+20),(0,255,0),2)
        return self.keypoints, perimeterBox


    #Flora - if all the users fingers are down then create and return a random color
    # this is still a WIP - sometimes it triggers incorrectly
    def detectFist(self, fingers, currentColor):
        if all(x >= 1 for x in fingers):
            return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        return currentColor


    #Finds both of the users hands and draws the keypoints on them using mediapipe - Kenyan
    def findUsersHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            #need to check keypoints in both hands and thats why you need the for loop
            for handKeypoints in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handKeypoints, self.mpHands.HAND_CONNECTIONS)
        return img

    #Finds the distance between keypoints - Kenyan
    def findDistance(self, p1, p2, img, draw=True, rVal=15, t=3):
        kpX1, kpY1 = self.keypoints[p1][1:]
        kpX2, kpY2 = self.keypoints[p2][1:]
        cXPos, cYPos = (kpX1 + kpX2) // 2, (kpY1 + kpY2) // 2

        if (draw == True):
            cv2.line(img, (kpX1, kpY1), (kpX2, kpY2), (255, 0, 255), t)
            cv2.circle(img, (kpX1, kpY1), rVal, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (kpX2, kpY2), rVal, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cXPos, cYPos), rVal, (0, 0, 255), cv2.FILLED)
            #can find distance using hypotenuse
            length = math.hypot(kpX2 - kpX1, kpY2 - kpY1)

        return length, img, [kpX1, kpY1, kpX2, kpY2, cXPos, cYPos]

