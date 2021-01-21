import cv2 as cv
import numpy as np
import datetime

import dlib
from math import hypot

# Loading Camera and Nose image and Creating mask
cap = cv.VideoCapture(0)
nose_image = cv.imread("nose.png")
_, frame = cap.read()
rows, cols, _ = frame.shape
nose_mask = np.zeros((rows, cols), np.uint8)

# Loading Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


while True:
    _, frame = cap.read()
    nose_mask.fill(0)
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = detector(frame)
    for face in faces:
        landmarks = predictor(gray_frame, face)

        # Nose coordinates
        top_nose = (landmarks.part(29).x, landmarks.part(29).y)
       
        # cv.circle(frame,top_nose,3,(255,0,0),-1)
        
        center_nose = (landmarks.part(30).x, landmarks.part(30).y)
        # cv.circle(frame,center_nose,3,(255,0,0),-1)
       
        left_nose = (landmarks.part(31).x, landmarks.part(31).y)
        # cv.circle(frame,left_nose,3,(255,0,0),-1)

        right_nose = (landmarks.part(35).x, landmarks.part(35).y)
        # cv.circle(frame,right_nose,3,(255,0,0),-1)

        nose_width = int(hypot(left_nose[0] - right_nose[0],
                           left_nose[1] - right_nose[1]) * 1.7)
        nose_height = int(nose_width * 0.77)

        # New nose position
        top_left = (int(center_nose[0] - nose_width / 2),
                              int(center_nose[1] - nose_height / 2))
        bottom_right = (int(center_nose[0] + nose_width / 2),
                       int(center_nose[1] + nose_height / 2))


        # Adding the new nose
        nose = cv.resize(nose_image, (nose_width, nose_height))
        nose_gray = cv.cvtColor(nose, cv.COLOR_BGR2GRAY)
        _, nose_mask = cv.threshold(nose_gray, 25, 255, cv.THRESH_BINARY_INV)

        nose_area = frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width]
        nose_area_no_nose = cv.bitwise_and(nose_area, nose_area, mask=nose_mask)
        final_nose = cv.add(nose_area_no_nose, nose)

        frame[top_left[1]: top_left[1] + nose_height,
                    top_left[0]: top_left[0] + nose_width] = final_nose

        # cv.imshow("Nose area", nose_area)
        # cv.imshow("Nose ", nose)
        # cv.imshow("final nose", final_nose)



    cv.imshow("Frame", frame)


    dt = str(datetime.datetime.now())
    
    key = cv.waitKey(1)


    if key == 27:
        break

    elif key ==ord("c"):
        photo="photo_{}.png".format(dt)
        cv.imwrite(photo,frame)


        

cap.release()
cv.destroyAllWindows()