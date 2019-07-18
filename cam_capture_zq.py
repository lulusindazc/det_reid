import numpy as np
import cv2
import argparse

### for online video read

#ip = '192.168.0.12:554/11'
ip3 = 'rtsp://admin:a1234567@192.168.0.7:554/11'
parser = argparse.ArgumentParser(description='Input IP and video name.')
#parser.add_argument('-i', '--ip', type=str)
# parser.add_argument('-n', '--name', type=str)
# args = parser.parse_args()

cap3 = cv2.VideoCapture(ip3)
# cap7 = cv2.VideoCapture(ip7)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'MP42')
# out3 = cv2.VideoWriter(args.name + '.mp4',fourcc, 20.0, (1920,1080))
# out7 = cv2.VideoWriter(args.name + '7.mp4',fourcc, 20.0, (1920,1080))
#and ret7==True:
#print(cap.isOpened())
count = 0
while(cap3.isOpened()):
    ret3, frame3 = cap3.read()
    if ret3==True:
        #frame = cv2.flip(frame,0)
        cv2.imwrite("cam_7.png",frame3)
        break
        if cv2.waitKey(1) & 0xFF == ord('s'):
            count += 1
            print (count)
            cv2.imwrite("cam_6.png",frame3)
            #count += 1
        frame3 = cv2.resize(frame3,(640,480))
        cv2.imshow('frame3',frame3)
        # frame7 = cv2.resize(frame7,(640,480))
        # cv2.imshow('frame7',frame7)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # else:
    #     cap3 = cv2.VideoCapture(ip3)
    #     cap7 = cv2.VideoCapture(ip7)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

# Release everything if job is finished
#cap3.release()
#out3.release()
#cv2.destroyAllWindows()

### for offline video read

'''
#ip = '192.168.0.2:554/11'
ip3 = 'rtsp://admin:a1234567@192.168.0.3:554/11'
ip7 = 'rtsp://admin:a1234567@192.168.0.7:554/11'
parser = argparse.ArgumentParser(description='Input IP and video name.')
#parser.add_argument('-i', '--ip', type=str)
parser.add_argument('-n', '--name', type=str)
args = parser.parse_args()

# cap3 = cv2.VideoCapture(ip3)
# cap7 = cv2.VideoCapture(ip7)
cap3 = cv2.VideoCapture("first.mp4")
cap7 = cv2.VideoCapture("fourth.mp4")

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'MP42')
# out3 = cv2.VideoWriter(args.name + '.mp4',fourcc, 20.0, (1920,1080))
# out7 = cv2.VideoWriter(args.name + '7.mp4',fourcc, 20.0, (1920,1080))

#print(cap.isOpened())
count = 0
while(cap3.isOpened()):
    ret3, frame3 = cap3.read()
    ret7, frame7 = cap7.read()
    if ret3==True and ret7==True:
        #frame = cv2.flip(frame,0)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            count += 1
            print ("111")
            cv2.imwrite("C://Users//zhangqian//Desktop//collect_data//my_data_capture//out3_"+str(count)+".png",frame3)
            cv2.imwrite("C://Users//zhangqian//Desktop//collect_data//my_data_capture//out7_"+str(count)+".png",frame7)
        #count += 1
        frame3 = cv2.resize(frame3,(640,480))
        cv2.imshow('frame3',frame3)
        frame7 = cv2.resize(frame7,(640,480))
        cv2.imshow('frame7',frame7)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # else:
    #     cap3 = cv2.VideoCapture(ip3)
    #     cap7 = cv2.VideoCapture(ip7)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

# Release everything if job is finished
cap3.release()
cap7.release()
out3.release()
out7.release()
cv2.destroyAllWindows()
'''
