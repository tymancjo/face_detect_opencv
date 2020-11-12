import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)
W, H = (1280 // 2, 720 // 2)
# W, H = (1280 , 720 )
cap.set(3, W)
cap.set(4, H)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

last = []
for i in range(10):
    last.append((0,0))

target = False

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    

    faces = detector(gray)

    for i,face in enumerate(faces):
        x, y = face.left(), face.top()
        x1, y1 = face.right(), face.bottom()
        last_x, last_y = last[i]

        # (x, y, w, h) = rect
        _x = int((x + x1) / 2)
        _y = int((y + y1)/2)

        A = 0.5

        mid_x = int(A * _x + (1-A) * last_x)
        mid_y = int(A * _y + (1-A) * last_y)

        last[i] = ( mid_x, mid_y )

        cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)

        if target:
            h_line = cv2.line(frame, (0,mid_y), (W, mid_y), (0,0,255), 1) 
            v_line = cv2.line(frame, (mid_x,0), (mid_x, H), (0,0,255), 1) 


        # landmarks = predictor(gray, face)
        # left_point = (landmarks.part(36).x, landmarks.part(36).y)
        # right_point = (landmarks.part(39).x, landmarks.part(39).y)
        # center_top = midpoint(landmarks.part(37), landmarks.part(38))
        # center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

        # hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
        # ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    upsized = cv2.resize(frame, (2*W, 2*H), interpolation = cv2.INTER_AREA)
    cv2.imshow("Frame", upsized)

    key = cv2.waitKey(1)
    if key == 27 or key==ord('q'):
        break
    if key==ord('t'):
        target = not target

cap.release()
cv2.destroyAllWindows()