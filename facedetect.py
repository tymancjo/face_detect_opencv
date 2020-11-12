import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
W, H = (1280 // 2, 720 // 2)
cap.set(3, W)
cap.set(4, H)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cv2.namedWindow("Frame")
cv2.createTrackbar("Neighbours", "Frame", 6, 20, nothing)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    neighbours = cv2.getTrackbarPos("Neighbours", "Frame")

    faces = face_cascade.detectMultiScale(gray, 1.3, neighbours)
    for rect in faces:
        (x, y, w, h) = rect
        mid_x = int(x + w/2)
        mid_y = int(y + h/2)

        h_line = cv2.line(frame, (0,mid_y), (W, mid_y), (0,0,255), 2) 
        v_line = cv2.line(frame, (mid_x,0), (mid_x, H), (0,0,255), 2) 
        # frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()