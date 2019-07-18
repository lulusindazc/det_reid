import rospy
from rospy.exceptions import *
import cv2, os, time, socket, threading
import numpy as np

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

import os.path as osp
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature

import publish_msg.publish_msg as pubmsg
import pickle
import torchvision.transforms as T



class Common(object):
    def __init__(self):
        pass

    def loadDataset(self):
        torch.cuda.set_device(0)
        logs_dir = 'market-1501-Exper33/RPP/'
        num_features = 256
        num_classes = 751
        T = 1
        dim = 256
        dropout = 0.5

        model = models.create('resnet50_rpp', num_features=num_features, dropout=dropout, num_classes=num_classes,
                              cut_at_pooling=False, FCN=True, T=T, dim=dim)
        model = model.cuda()
        checkpoint = load_checkpoint(osp.join(logs_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])

        res = []
        frame_number = 0

        # --datasets
        shujuku = {}
        rentidir = '/data/reid/renti/queries'

        return model

    def callback(self, param_tuple):  # param_tuple
        # threadPubMsg_shelfID_3 = param_tuple[0]
        cfg = param_tuple[1]
        model = param_tuple[2]

        frame_number_list = param_tuple[3]
        bridge = param_tuple[4]
        camera_id = param_tuple[5]
        flag = param_tuple[6]

        frame = param_tuple[7]

        shape = frame.shape

        global size
        size = (shape[1], shape[0])

        # global frame_number
        frame_number_list[0] = frame_number_list[0] + 1
        frame_number = frame_number_list[0]
        wh_ratio = frame.shape[1] / frame.shape[0]

        if type(frame) != np.ndarray:
            return True

        # detect per 8 frame
        # if frame_number % 8 == 1 or frame_number % 8 == 2 or frame_number % 8 == 3 or frame_number % 8 == 4 or frame_number % 8 == 5 or frame_number % 8 == 6 or frame_number % 8 == 7:
        #    return True
        cfg.cuda()

        use_cuda = 1
        sized = cv2.resize(frame, (cfg.width, cfg.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        r = do_detect(cfg, sized, 0.5, 0.4, use_cuda)

        num_classes = 80
        if num_classes == 20:
            namesfile = 'data/voc.names'
        elif num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/names'

        class_names = load_class_names(namesfile)

        res = []
        for item in r:
            if class_names[item[6]] == 'person':
                res.append(item)

        # # get the max rectangle
        # result = []
        # for item in res:
        #     result = item
        #     if (len(result) > 0):
        #         reid_draw(frame, result, model, cfg)

        cv2.imshow('Cam2', cv2.resize(frame, (int(512 * wh_ratio), 512)))

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

        return res, camera_id
