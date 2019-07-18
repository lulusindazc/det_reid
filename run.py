import sys
import cv2
import time
import os
import threading
import pdb

import rospy  # Bingoren
from std_msgs.msg import Header #Bingo
from sensor_msgs.msg import Image # Bingo
from sensor_msgs.msg import CompressedImage #Bingo
from cv_bridge import CvBridge, CvBridgeError #Bingo


import numpy as np
from PIL import Image, ImageDraw
from models.tiny_yolo import TinyYoloNet
from utils import *
from darknet import Darknet

import os.path as osp
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature
from numpy import average, linalg, dot
from torchvision import transforms
import torch.nn.functional as F

fall_time=time.time()
# frame_number = 0
class_names = None
# m=None


def playsound(SOUND_IP):
    cmd='sshpass -p "hri" ssh hri@{} "python play.py"'.format(SOUND_IP)
    os.system(cmd)

def pairwise_distance(fea1, fea2):
    fea1 = torch.squeeze(fea1,0)
    fea1 = torch.squeeze(fea1,-1)
    fea2 = torch.squeeze(fea2,0)
    fea2 = torch.squeeze(fea2,-1)
    x = fea1
    y = fea2
    #m, n = x.size(0), y.size(0)
    m1,n=1,1
    x = x.view(m1, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m1, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m1).t()
    dist.addmm_(1, -2, x, y.t())
    return torch.sum(dist)
	
def jieduan(img,left,top,right,bottom):
    imgg=np.zeros((bottom-top,right-left,3))
    #for i in range(top,bottom):
    #  for j in range(left,right):
    #    for k in range(3):
    #      imgg[i-top][j-left][k] = img[i][j][k]
    imgg = img[top:bottom, left:right, :]
    return imgg

def reid_draw(frame, b_b, model, cfg):
    global size

    # print("size = ", size)
    # print("b_b = ", b_b)
    # print model


    cfg.cuda()

    left=int((b_b[0] - b_b[2]/2.0) * size[0])
    top=int((b_b[1]- b_b[3]/2.0) * size[1])
    right=int((b_b[0] + b_b[2]/2.0) * size[0])
    bottom=int((b_b[1] + b_b[3]/2.0) * size[1])
    img1 = jieduan(frame,left,top,right,bottom)
    img = np.transpose(img1, (2,0,1)).astype(np.float32)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)

    feature = extract_cnn_feature(model, img.cuda())

    minsim = -1
    id_name = 'new'
    rentidir = '/home/tujh/renti/'
    #for feature2,filename in shujuku:

    for query in shujuku:
        for fea in shujuku[query]:
            distan = pairwise_distance(feature,fea)
            if minsim > distan or minsim == -1:
                minsim = distan
                id_name = query

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(frame,id_name,(left,top),cv2.FONT_HERSHEY_COMPLEX,6,(255,0,0),2)

    print("set customer message")
    threadPubMsg.set_customer(first_name=id_name, x=left, y=top, w=right, h=bottom)
    threadPubMsg.set_commodity_recognition_trigger(camera_id=camera_id, flag1=1, flag2=2)
    return frame



def loadDataset():
    torch.cuda.set_device(0)
    logs_dir = 'market-1501-Exper33/RPP/'
    num_features = 256
    num_classes = 751
    T = 1
    dim = 256
    dropout = 0.5

    ###
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
    for query in os.listdir(rentidir):
        query_dir = osp.join(rentidir, query)
        shujuku[query] = []
        for img in os.listdir(query_dir):
            _img = cv2.imread(osp.join(query_dir, img))
            _img = np.transpose(_img, (2, 0, 1)).astype(np.float32)
            _img = torch.from_numpy(_img)
            _img = torch.unsqueeze(_img, 0)
            _feature = extract_cnn_feature(model, _img.cuda())
            shujuku[query].append(_feature)

            # --

    return model, shujuku


def callback(msg):    # param_tuple
    header_cur = msg.header
    frame = bridge.compressed_imgmsg_to_cv2(msg)  # for compressed
    # frame = bridge.imgmsg_to_cv2(msg)

    # cv2.imshow('Fall detection', cv2.resize(frame, (int(512), 512)))
    # cv2.waitKey(3)
    # return True

    shape = frame.shape     # (1080, 1920, 3)

    global size
    size = (shape[1], shape[0])


    # global frame_number
    frame_number = frame_number + 1
    wh_ratio = frame.shape[1] / frame.shape[0]
    # Quit when the input video file ends
    if type(frame) != np.ndarray:  # is frame readed is null,   go to the next loop
        print('********************')
        # continue
        return True
    # detect per 8 frame
    if frame_number % 8 == 1 or frame_number % 8 == 2 or frame_number % 8 == 3 or frame_number % 8 == 4 or frame_number % 8 == 5 or frame_number % 8 == 6 or frame_number % 8 == 7:
        return True

    cfg.cuda()
    print cfg.width, cfg.height
    use_cuda = 1
    sized = cv2.resize(frame, (cfg.width, cfg.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    # start=time.time()
    r = do_detect(cfg, sized, 0.5, 0.4, use_cuda)


    ###########
    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    class_names = load_class_names(namesfile)
    ################


    # print('the whole running time is: '+str(time.time()-start))
    res = []
    for item in r:
        if class_names[item[6]] == 'person' or class_names[item[6]] == 'dog' or class_names[item[6]] == 'cat' or \
                class_names[item[6]] == 'horse':
            res.append(item)

    # get the max rectangle
    result = []
    maxArea = 0
    if len(res) == 1:
        result = res[0]
        if (len(result) > 0):
            frame = reid_draw(frame, result, model, cfg)

    elif len(res) > 1:
        for item in res:
            result = item
            if (len(result) > 0):
                frame = reid_draw(frame, result, model, cfg)

    cv2.imshow('Fall detection', cv2.resize(frame, (int(512 * wh_ratio), 512)))

    if len(res) > 1:
        for item in res:
            if item[2] * item[3] > maxArea:
                maxArea = item[2] * item[3]
                result = item
    elif len(res) == 1:
        result = res[0]
        # draw the result
    if (len(result) > 0):
        # label the result
        reid_draw(frame, result, model, cfg)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node('det_reid', anonymous=True)

    # import beginner_tutorials.scripts.publish_msg as pubmsg
    import scripts.publish_msg as pubmsg

    threadPubMsg = pubmsg.MsgPublishClass()
    threadPubMsg.setDaemon(True)
    threadPubMsg.start()


    # logs_dir = '/home/tujh/PCB_RPP_1028/market-1501-Exper33/RPP/'
    model, shujuku = loadDataset()

    cfg = Darknet('cfg/yolov3.cfg')
    cfg.load_weights('yolov3.weights')
    cfg.cuda()
    print "main cfg: ", cfg.width, cfg.height

    # global frame_number
    frame_number = 0
    bridge = CvBridge()

    camera_id = 3
    frame_topic = '/camera_' + str(camera_id) + '/rgb/compressed/image_color'
    # frame_topic = '/camera_' + str(camera_id) + '/rgb/image_gray'
    # frame_topic = '/camera_' + str(camera_id) + '/rgb/image_color'

    # type. "Image" for image_gray or image_color. "CompressedImage" for compressed
    rospy.Subscriber(frame_topic, CompressedImage, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

    while not rospy.is_shutdown():
        pass
