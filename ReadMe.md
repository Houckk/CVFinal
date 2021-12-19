*Note that we only have 1 commit because a group member wasn't comfortable using github so we just sent files back and forth and now I am making the first commit and pushing to github - Kenyan

We used PyCharms as our IDE - Some instructions like installations will be specific to that

To run the code run the code:
1. Make sure cv2 & mediapipe are installed on your computer
   1. To install on PyCharms IDE go to File > Settings > Project: CVHandTracking > Python Interpreter > Click the + symbol at the top right > search opencv-python & mediapipe respectively

2. Run the HandTracker.py file (we used PyCharms and just right clicked on the file)
   
   1.Note that you will need to hold a hand up to the camera while the program is loading or you will get an error 



Contributions: All contributions are labeled with comments in the code

Flora:
1. Setting up the video
2. Creating constructor for HandTrackingFeatures
3. Made a function to generate a random color based on user hand keypoints

Eric:
1. Wrote a function to determine which fingers of a user are up by checking keypoints
2. Wrote a function to track a users hand and give a bounding box around their fingers
3. Created graphics that are used as the header

Kenyan:
1. Handled finger recognition to determine whether the user was in select or draw mode
   1. Draw mode is 1 finger up
   2. Select mode is 2 fingers up
2. Implemented a distance function for keypoints
