import cv2
import numpy as np
cap=cv2.VideoCapture('rtsp://192.168.1.116:554/11')
print cap.isOpened()
frameNum=1
while(cap.isOpened()):
    ret,frame=cap.read()
    print frameNum
    if type(frame)!=np.ndarray:  # is frame readed is null,   go to the next loop
        print('********************')
        continue
    frameNum=frameNum+1
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
