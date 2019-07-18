#encoding: utf-8

from multiprocessing import Pool, Process
from multiprocessing.managers import BaseManager
import os, time, random

import HKIPcamera
import cv2
import copy
import math
from loadconfig import *

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

from utils import *
from darknet import Darknet
import Common
from Common import *

import os.path as osp
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature

import publish_msg.publish_msg as pubmsg
import pickle
import torchvision.transforms as T


class MyManager(BaseManager):
    pass


MyManager.register('Common', Common)



class HKCamera(object):
    def __init__(self, ip, name, pw):
        self._ip = ip
        self._name = name
        self._pw = pw
        HKIPcamera.init(self._ip, self._name, self._pw)

    def getFrame(self):
        frame = HKIPcamera.getframe()
        return frame


def call_camera(common, camera_id):
    ip = '192.168.0.' + str(camera_id)
    name = str('admin')
    pw = str('a1234567')
    camera = HKCamera(ip, name, pw)

    model = common.loadDataset()

    while not rospy.is_shutdown():
        frame_origin = camera.getFrame()

        frame_origin = np.array(frame_origin)
        cv2.imshow('Cam2', frame_origin)
        frame_origin = cv2.resize(frame_origin, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        frame_trans = copy.deepcopy(frame_origin)

        print("*******")

        wh_ratio = frame_origin.shape[1] / frame_origin.shape[0]

        #cv2.imshow('Cam2', cv2.resize(frame_origin, (int(512 * wh_ratio), 512)))



        # cfg = Darknet('cfg/yolov3.cfg')
        # cfg.load_weights('yolov3.weights')
        # cfg.cuda()
        # # global frame_number
        # frame_number2 = [0]
        # flag = [0]
        # bridge = CvBridge()
        #
        # res, camera_id = common.callback(
        #     (None, cfg, model, frame_number2, bridge, camera_id, flag, frame_origin))

    HKIPcamera.release()




if __name__ == '__main__':
    rospy.init_node('MultiProcessingNode', anonymous=True)
    manager = MyManager()
    manager.start()

    common = manager.Common()

    # 开启多进程, 每个进程处理每个摄像头
    camera_ids = [2, 6, 7, 12]
    #camera_ids = [2]
    proces = []
    for camera_id in camera_ids:
        if camera_id == 2:
            # 2号摄像头处理1,2货架
            proces.append(Process(target=call_camera, args=(common, camera_id)))
        elif camera_id == 6:
            proces.append(Process(target=call_camera, args=(common, camera_id)))
        elif camera_id == 7:
            proces.append(Process(target=call_camera, args=(common, camera_id)))
        else:
            proces.append(Process(target=call_camera, args=(common, camera_id)))


    for p in proces:
        p.start()
    for p in proces:
        p.join()

    print("rospy.spin()")
    rospy.spin()

    manager.shutdown()
    print('master exit.')