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

import os.path as osp
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature

import time

import publish_msg.publish_msg as pubmsg
import pickle
import torchvision.transforms as T



class HKCamera(object):
    def __init__(self, ip, name, pw):
        self._ip = ip
        self._name = name
        self._pw = pw
        HKIPcamera.init(self._ip, self._name, self._pw)

    def getFrame(self):
        frame = HKIPcamera.getframe()
        return frame


def compare_dic(dic1, dic2):
    for i in (dic1):
        for j in (dic2):
            if i == j and dic1[i] != dic2[j]:
                return True
    return False


def exist_people(dic1):
    for i in (dic1):
        if dic1[i] == 1:
            return True
    return False


def diff_dic(dic2, dic1):
    diff = []
    for i in (dic1):
        for j in (dic2):
            if i == j and dic1[i] != dic2[j]:
                diff.append(i)
        if not dic2.has_key(i):
            diff.append(i)
    return diff


def pairwise_distance(fea1, fea2):
    fea1 = torch.squeeze(fea1, 0)
    fea1 = torch.squeeze(fea1, -1)
    fea2 = torch.squeeze(fea2, 0)
    fea2 = torch.squeeze(fea2, -1)
    x = fea1
    y = fea2
    m1, n = 1, 1
    x = x.view(m1, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m1, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m1).t()
    dist.addmm_(1, -2, x, y.t())
    return torch.sum(dist)


def jieduan(img, left, top, right, bottom):
    imgg = np.zeros((bottom - top, right - left, 3))
    imgg = img[top:bottom, left:right, :]
    return imgg


def calcIOU(p_x, p_y, p_bx, p_by, c_x, c_y, c_bx, c_by):
    zh = c_x
    c_x = c_bx  # 960 - c_x
    c_bx = zh  # 960 - c_bx
    condition1 = p_x >= c_x and p_x <= c_bx
    condition2 = p_bx >= c_x and p_bx <= c_bx
    condition3 = p_y >= c_y and p_y <= c_by
    condition4 = p_by >= c_y and p_by <= c_by
    # print p_x, p_y, p_bx, p_by, c_x, c_y, c_bx, c_by

    if (condition1 and condition3) or (condition1 and condition4) or \
            (condition2 and condition3) or (condition2 and condition4):
        calcIOU = 1
    else:
        calcIOU = -1

    return calcIOU


def newcalcIOU(two_x, two_y, two_w, two_h, one_x, one_y, one_w, one_h):
    zh = one_x
    one_x = one_w  # 960-one_x
    one_w = zh  # 960-one_w
    S_rec1 = (one_w - one_x) * (one_h - one_y)
    S_rec2 = (two_w - two_x) * (two_h - two_y)
    sum_area = S_rec1 + S_rec2
    left_line = max(one_x, two_x)
    right_line = min(one_w, two_w)
    top_line = max(one_y, two_y)
    bottom_line = min(one_h, two_h)

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return -1
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        # print intersect, S_rec2
        iou = float(intersect) / S_rec2
        return iou


def coordinate_IOU(two_x, two_y, two_w, two_h, one_x, one_y, one_w, one_h):  # compute the coordinate of the IOU area
    zh = one_x
    one_x = one_w  # 960-one_x
    one_w = zh  # 960-one_w
    left_line = max(one_x, two_x)
    right_line = min(one_w, two_w)
    top_line = max(one_y, two_y)
    bottom_line = min(one_h, two_h)
    return left_line, top_line, right_line, bottom_line


def distanceCal(p_y, s_y):
    # return math.sqrt(pow(abs(p_x - s_x), 2) + pow(abs(p_y - s_y), 2))
    return abs(p_y - s_y)

# person detection and reid
def preprocess(img):
    img = cv2.resize(img, (128, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img[:,:,::-1]
    img = test_transformer(img)
    img = torch.unsqueeze(img, 0)
    return img

global save_box_no
save_box_no = 0
def reid_draw(frame, b_b, model, cfg, huojia1_id, pre_res,change_idnum):
    global size
    global save_box_no
    id_name = 0
    cfg.cuda()
    left = int((b_b[0] - b_b[2] / 2.0) * size[0])
    top = int((b_b[1] - b_b[3] / 2.0) * size[1])
    right = int((b_b[0] + b_b[2] / 2.0) * size[0])
    bottom = int((b_b[1] + b_b[3] / 2.0) * size[1])

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        return left, top, right, bottom, 999

    # if bottom > 530:
    #     ratio = float(bottom - top) / (right - left)
    #     #print("ratio is: {}".format(ratio))
    #     if ratio < 1.5:
    #         #print("ratio is: {}".format(ratio))
    #         print('filtered out')
    #         return left, top, right, bottom, 999

    frame_reid = copy.deepcopy(frame)
    # draw shangpin area
    left_x, top_y, right_m, bottom_n = shangpin_area(huojia1_id)
    cv2.rectangle(frame, (left_x, top_y), (right_m, bottom_n), (0, 255, 0), 2)

    ratio = float(bottom - top) / (right - left)
    # # print(ratio)
    # if ratio < 2.0:
    #     # print('filtered out')
    #     return left, top, right, bottom, 999

    img1 = jieduan(frame_reid, left, top, right, bottom)

    img = preprocess(img1)

    feature = extract_cnn_feature(model, img.cuda())

    minsim = -1

    try:
        pkl_file = open('/data/reid/renti/data.pkl', 'rb')
        shujuku = pickle.load(pkl_file)
        pkl_file.close()
    except:
        pkl_file = open('/data/reid/renti/data_bu.pkl', 'rb')
        shujuku = pickle.load(pkl_file)
        pkl_file.close()

    rentidir = '/home/tujh/renti/'
    # pkl_file = open('/data/reid/renti/data.pkl', 'rb')
    # shujuku = pickle.load(pkl_file)
    # pre_item_huoid={}
    # for id_name, pre_item in pre_res.items():
    #     if huojia1_id == pre_item[-1]:
    #         pre_item_huoid[id_name]=pre_item ##person id in front of huojia_id

    if change_idnum:#len(pre_res) == len(shujuku) and pre_item_huoid:
        id_name = reid_draw_multi(pre_res, b_b)
        pre_fix = 'B:'
    else:
        # for feature2,filename in shujuku:
        for query in shujuku:
            for fea in shujuku[query]:
                distan = pairwise_distance(feature, fea)
                if minsim > distan or minsim == -1:
                    minsim = distan
                    id_name = int(query)
        pre_fix='R:'
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(frame, pre_fix+str(id_name), (left, top), cv2.FONT_HERSHEY_COMPLEX, 6, (255, 0, 0), 2)
    cv2.imwrite('/home/zhaocy/yhr/tmp_imgs/' + str(save_box_no) + '_' + str(id_name) + '.jpg', img1)
    save_box_no += 1

    return left, top, right, bottom, id_name


def reid_draw_multi(pre_res, result):
    dic_res = {}
    if len(pre_res) != 0:
        pre_item_center = [abs(result[0] - pre_item[0]) for id_name, pre_item in pre_res.items()]
        dist_res = min(pre_item_center)
        index_min_dist = pre_item_center.index(dist_res)
        id_name_res = list(pre_res.keys())

        id_name = id_name_res[index_min_dist]
        left, top, right, bottom = pre_res[id_name]
        # dic_res[id_name] = [left, top, right, bottom]
        return id_name



normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transformer = T.Compose([T.ToTensor(),
                              normalizer,
                              ])


# the area of shangpin
# shelf1,2,3,5,6,7
def shangpin_area(shelfid):
    if shelfid == 1:
        left_x, top_y, right_m, bottom_n = 741, 18, 596, 253
    elif shelfid == 10:
        left_x, top_y, right_m, bottom_n = 568, 9, 389, 252
    elif shelfid == 4:
        left_x, top_y, right_m, bottom_n = 870, 10, 190, 370
    elif shelfid == 2:
        left_x, top_y, right_m, bottom_n = 680, 27, 169, 332
    elif shelfid == 6:
        left_x, top_y, right_m, bottom_n = 353, 39, 200, 273
    elif shelfid == 7:
        left_x, top_y, right_m, bottom_n = 712, 3, 310, 344
    else:
        left_x, top_y, right_m, bottom_n = 0, 0, 0, 0

    return left_x, top_y, right_m, bottom_n



# initial the flag of all the people we detected
def initial_flag(left, top, right, bottom, shelfid):
    left_x, top_y, right_m, bottom_n = shangpin_area(shelfid)
    calcIOU1 = newcalcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
    #print("Shelf {}: Filter in IOU = {:4f}".format(shelfid, calcIOU1))
    distance = distanceCal(bottom_n, bottom)
    #print("shelf {}: distance = {}".format(shelfid, distance))
    if calcIOU1 > 0.5 and distance < 100:
        flag = 1
    else:
        flag = 0
    return flag


def initial_flag_out(left, top, right, bottom, shelfid):
    left_x, top_y, right_m, bottom_n = shangpin_area(shelfid)
    calcIOU1 = newcalcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
    #print("Shelf {}: Filter out IOU = {}".format(shelfid, calcIOU1))
    distance = distanceCal(bottom_n, bottom)
    #print("shelf {}: distance = {}".format(shelfid, distance))
    #if calcIOU1 > 0.2 or distance < 130:
    if calcIOU1 > 0.2 and distance < 130:
        flag = 1
    else:
        flag = 0
    return flag


def xuanze_original(res, frame, model, cfg, camera_id, dic_change, huojia1_id,pre_res):
    dic = {}
    change_idnum = len(pre_res.keys()) == len(res)
    if len(res) == 1:
        result = res[0]
        left, top, right, bottom, id_name = reid_draw(frame, result, model, cfg, huojia1_id,pre_res,change_idnum)
        if id_name == 999:
            None
        else:
            if dic_change.has_key(id_name):
                if dic_change[id_name] == 1:
                    flag = initial_flag_out(left, top, right, bottom, huojia1_id)
                else:
                    flag = initial_flag(left, top, right, bottom, huojia1_id)
            else:
                flag = initial_flag(left, top, right, bottom, huojia1_id)
            dic[id_name] = flag


    elif len(res) > 1:
        for item in res:
            result = item
            if (len(result) > 0):
                left, top, right, bottom, id_name = reid_draw(frame, result, model, cfg, huojia1_id,pre_res,change_idnum)
                if id_name == 999:
                    None
                else:
                    if dic_change.has_key(id_name):
                        if dic_change[id_name] == 1:
                            flag = initial_flag_out(left, top, right, bottom, huojia1_id)
                        else:
                            flag = initial_flag(left, top, right, bottom, huojia1_id)
                    else:
                        flag = initial_flag(left, top, right, bottom, huojia1_id)
                    dic[id_name] = flag

    return dic


def people_list(res, shelfid):
    peolist = []
    left_x, top_y, right_m, bottom_n = shangpin_area(shelfid)
    for b_b in res:
        global size
        left = int((b_b[0] - b_b[2] / 2.0) * size[0])
        top = int((b_b[1] - b_b[3] / 2.0) * size[1])
        right = int((b_b[0] + b_b[2] / 2.0) * size[0])
        bottom = int((b_b[1] + b_b[3] / 2.0) * size[1])
        if calcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n) > 0:
            x1, x2, x3, x4 = coordinate_IOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
            peolist.append(x1)
            peolist.append(x2)
            peolist.append(x3)
            peolist.append(x4)
    return peolist

global in_out_moments_count
in_out_moments_count = 0

# choose the one which is next to the area of shangpin
def xuanze(res, frame, model, cfg, threadPubMsg_dict, camera_id, dic, change_dic,
           huojia1_id, frame_trans,pre_res):
    global in_out_moments_count
    change_idnum = len(pre_res.keys()) == len(res)
    for item in res:
        result = item
        # add new person
        if (len(result) > 0):
            left, top, right, bottom, id_name = reid_draw(frame, result, model, cfg, huojia1_id,pre_res,change_idnum)
            if id_name == 999:
                None
            else:
                in_out_people = diff_dic(dic, change_dic)
                if id_name in in_out_people:
                    left_x, top_y, right_m, bottom_n = shangpin_area(huojia1_id)
                    customer_name = "name" + str(id_name)
                    assert type(id_name) is int  # must be number
                    # print("set customer message")

                    threadPubMsg = threadPubMsg_dict['shelfID_' + str(huojia1_id)]

                    if change_dic[id_name] == 1:
                        time.sleep(1)  # time delay x seconds
                        flag = 1
                        flag1 = 0
                        flag2 = 1
                        print(
                            "t = {}, flag = {}, flag1 = {}, flag2 = {}, id_name = {}, shelfID = {}".format(time.time(),
                                                                                                           flag, flag1,
                                                                                                           flag2,
                                                                                                           id_name,
                                                                                                           huojia1_id))

                        threadPubMsg.set_commodity_recognition_trigger_with_image(camera_id=camera_id,
                                                                                  person_id=id_name,
                                                                                  shelf_id=huojia1_id, flag=flag,
                                                                                  flag1=flag1,
                                                                                  flag2=flag2,
                                                                                  flag_list=people_list(res, huojia1_id),
                                                                                  frame=frame_trans)
                        cv2.imwrite('/home/shiw/yhr/in_out_moments/' + str(in_out_moments_count) + '.jpg', frame)
                        in_out_moments_count += 1
                        print("huojia1: {}".format(huojia1_id))

                    else:

                        flag = 0
                        flag1 = 1
                        flag2 = 0
                        print(
                            "t = {}, flag = {}, flag1 = {}, flag2 = {}, id_name = {}, shelfID = {}".format(time.time(),
                                                                                                           flag, flag1,
                                                                                                           flag2,
                                                                                                           id_name,
                                                                                                           huojia1_id))
                        threadPubMsg.set_commodity_recognition_trigger_with_image(camera_id=camera_id,
                                                                                  person_id=id_name,
                                                                                  shelf_id=huojia1_id, flag=flag,
                                                                                  flag1=flag1,
                                                                                  flag2=flag2,
                                                                                  flag_list=people_list(res, huojia1_id),
                                                                                  frame=None)
                        cv2.imwrite('/home/shiw/yhr/in_out_moments/' + str(in_out_moments_count) + '.jpg', frame)
                        in_out_moments_count += 1

                        print("huojia1: {}".format(huojia1_id))


                    threadPubMsg.set_customer(name=customer_name, person_id=id_name, camera_id=camera_id, x=left, y=top,
                                              w=right, h=bottom)



    return dic


def loadDataset():
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
    checkpoint = load_checkpoint(osp.join(logs_dir, 'cvs_checkpoint_0107.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    res = []
    frame_number = 0

    # --datasets
    shujuku = {}
    rentidir = '/data/reid/renti/queries'

    return model

def callback(param_tuple):  # param_tuple
    cfg = param_tuple[1]
    model = param_tuple[2]
    dict_res = {}
    frame_number_list = param_tuple[3]
    bridge = param_tuple[4]
    camera_id = param_tuple[5]
    flag = param_tuple[6]

    frame = param_tuple[7]
    pre_res = param_tuple[9]
    huojia1_id = param_tuple[8]

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

    # get the max rectangle
    result = []
    change_idnum = len(pre_res.keys()) == len(res)
    for item in res:
        result = item
        if (len(result) > 0):
            left, top, right, bottom, id_name = reid_draw(frame, result, model, cfg, huojia1_id, pre_res,change_idnum)
            dict_res[id_name] = [left, top, right, bottom, huojia1_id]

    cv2.imshow('Cam2', cv2.resize(frame, (int(512 * wh_ratio), 512)))

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    return res, camera_id, dict_res


def main(camera_id, shelf_id):
    rospy.init_node('MultiProcessingNode', anonymous=True)
    ip = '192.168.0.' + str(camera_id)
    name = str('admin')
    pw = str('a1234567')
    camera = HKCamera(ip, name, pw)

    threadPubMsg_shelfID_1 = pubmsg.MsgPublishClass(cameraID=camera_id, shelfID=shelf_id[0])
    threadPubMsg_shelfID_1.setDaemon(True)
    threadPubMsg_shelfID_1.start()


    shelf1 = 'shelfID_' + str(shelf_id[0])
    threadPubMsg_dict = {shelf1: threadPubMsg_shelfID_1}


    model = loadDataset()

    cfg = Darknet('cfg/yolov3.cfg')
    cfg.load_weights('yolov3.weights')
    cfg.cuda()
    # global frame_number
    frame_number2 = [0]
    flag = [0]
    bridge = CvBridge()

    dic_change = {}
    pre_res = {}
    huojia1_id = shelf_id[0]
    print("huojia1_id: {}".format(huojia1_id))
    tmp = 0
    while not rospy.is_shutdown():
        frame_origin = camera.getFrame()

        frame_origin = np.array(frame_origin)
        frame_origin = cv2.resize(frame_origin, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        frame_trans = copy.deepcopy(frame_origin)

        # draw the shangping area
        # left_x, top_y, right_m, bottom_n = shangpin_area(huojia1_id)
        # cv2.rectangle(frame_origin, (left_x, top_y), (right_m, bottom_n), (0, 255, 0), 2)


        res, camera_id, dict_res = callback((None, cfg, model, frame_number2, bridge, camera_id, flag, frame_origin, huojia1_id, pre_res))


        if res == []:
            if tmp > 30:
                threadPubMsg = threadPubMsg_dict['shelfID_' + str(huojia1_id)]
                threadPubMsg.set_commodity_recognition_trigger_with_image(camera_id=camera_id, person_id=-1,
                                                                          shelf_id=huojia1_id, flag=0, flag1=0, flag2=0,
                                                                          flag_list=[], frame=None)

                tmp = 0

            else:
                tmp += 1
            continue
        else:
            tmp = 0



        dic = xuanze_original(res, frame_origin, model, cfg, camera_id, dic_change, huojia1_id,pre_res)

        if compare_dic(dic, dic_change) == False:
            pass
            pre_res = dict_res
        else:
            dic = xuanze(res, frame_origin, model, cfg, threadPubMsg_dict, camera_id, dic, dic_change,
                                      huojia1_id, frame_trans,pre_res)
            pre_res = {}
        #print("**********************")
        #print("dic_change_shelf_{}: {}".format(shelf_id[0], dic))
        #print("")
        dic_change = dic


    HKIPcamera.release()



if __name__ == '__main__':
    #rospy.init_node('MultiProcessingNode', anonymous=True)
    # manager = MyManager()
    # manager.start()


    #main(2, [1, 2])

    # 开启多进程, 每个进程处理每个摄像�?
    camera_ids = [5]
    shelf_ids = {5: [2]}

    # camera_ids = [6]
    # shelf_ids = {6: [7]}



    proces = []

    for camera_id in camera_ids:
        p = Process(target=main, args=(camera_id, shelf_ids[camera_id]))
        proces.append(p)
        p.start()


    for p in proces:
        p.join()

    print("rospy.spin()")
    rospy.spin()
