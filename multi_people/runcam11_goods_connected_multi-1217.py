#me             : 2018-11-27
# @Author           : Lu zhisheng,Zhang linlin
# @Description      : Give an interface to one cam
# @last_modification: 2018-12-10
#---------------------------------

import HKIPcamera
import cv2
from loadconfig import *


import rospy  # Bingoren
from sensor_msgs.msg import CompressedImage #Bingo
from cv_bridge import CvBridge, CvBridgeError #Bingo


from utils import *
from darknet import Darknet

import os.path as osp
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature

import publish_msg.publish_msg as pubmsg
import pickle
import torchvision.transforms as T


class_names = None

def compare_dic(dic1,dic2):
    for i in (dic1):
      for j in (dic2):
        if i==j and dic1[i] != dic2[j]:
            return True
    return False

def exist_people(dic1):
    for i in (dic1):
      if dic1[i] == 1:
          return True
    return False

def diff_dic(dic2,dic1):
    diff = []
    for i in (dic1):
      for j in (dic2):
        if i==j and dic1[i]!= dic2[j]:
            diff.append(i)
      if not dic2.has_key(i):
        diff.append(i)
    return diff

def pairwise_distance(fea1, fea2):
    fea1 = torch.squeeze(fea1,0)
    fea1 = torch.squeeze(fea1,-1)
    fea2 = torch.squeeze(fea2,0)
    fea2 = torch.squeeze(fea2,-1)
    x = fea1
    y = fea2
    m1,n=1,1
    x = x.view(m1, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m1, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m1).t()
    dist.addmm_(1, -2, x, y.t())
    return torch.sum(dist)
	
def jieduan(img,left,top,right,bottom):
    imgg=np.zeros((bottom-top,right-left,3))
    imgg = img[top:bottom, left:right, :]
    return imgg


def calcIOU(p_x, p_y, p_bx, p_by, c_x, c_y, c_bx, c_by):
    zh = c_x
    c_x = c_bx#960 - c_x
    c_bx = zh#960 - c_bx
    condition1 = p_x >= c_x and p_x <= c_bx
    condition2 = p_bx >= c_x and p_bx <= c_bx
    condition3 = p_y >= c_y and p_y <= c_by
    condition4 = p_by >= c_y and p_by <= c_by
    #print p_x, p_y, p_bx, p_by, c_x, c_y, c_bx, c_by
    
    if (condition1 and condition3) or (condition1 and condition4) or \
            (condition2 and condition3) or (condition2 and condition4):
        calcIOU = 1
    else:
        calcIOU = -1

    return calcIOU

def calcIOU_old(one_x, one_y, one_w, one_h, two_x, two_y, two_w, two_h):
    #one_x = 960-one_x
    #one_w = 960-one_w
    #zh=two_x
    #two_x = two_m
    #two_m = zh
    #print one_x,one_m,two_x,two_m
    #print one_y,one_n,two_y,two_n
    #one_w = abs(one_x - one_m)
    #one_h = abs(one_y - one_n)
    #two_w = abs(two_x - two_m)
    #two_h = abs(two_y - two_n)
    #print one_w,one_h,two_w,two_h
    if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)

        inter_square = inter_w * inter_h
        union_square = (one_w * one_h) + (two_w * two_h) - inter_square

        #calcIOU = inter_square / union_square * 1.0
        print two_w,two_h
        calcIOU = inter_square / (abs(two_y-two_h)*abs(two_x-two_w))
        print("calcIOU:", calcIOU)
    else:
        # print("No interse   ction!")
        calcIOU = -1

    return calcIOU

def newcalcIOU(two_x, two_y, two_w, two_h,one_x, one_y, one_w,one_h ):
    zh = one_x
    one_x = one_w#960-one_x
    one_w = zh#960-one_w
    S_rec1 = (one_w- one_x) * (one_h - one_y)
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
        #print intersect, S_rec2
        iou = float(intersect) / S_rec2
        #print("IOU:",iou)
        return iou

def coordinate_IOU(two_x, two_y, two_w, two_h,one_x, one_y, one_w,one_h): # compute the coordinate of the IOU area
    zh =one_x
    one_x = one_w#960-one_x
    one_w = zh#960-one_w
    left_line = max(one_x, two_x)
    right_line = min(one_w, two_w)
    top_line = max(one_y, two_y)
    bottom_line = min(one_h, two_h)
    return left_line, top_line, right_line, bottom_line
    
# person detection and reid
def preprocess(img):
    img = cv2.resize(img, (128,256))
    img = test_transformer(img)
    img = torch.unsqueeze(img, 0)
    #print(img.shape)
    return img
    
def reid_draw(frame, b_b, model, cfg):
    global size
    id_name = 'new'
    cfg.cuda()
    left=int((b_b[0] - b_b[2]/2.0) * size[0])
    top=int((b_b[1]- b_b[3]/2.0) * size[1])
    right=int((b_b[0] + b_b[2]/2.0) * size[0])
    bottom=int((b_b[1] + b_b[3]/2.0) * size[1])
    #print left,top,right,bottom
    if left<0 or right< 0 or top <0 or bottom <0:
        return left, top, right, bottom, id_name

    img1 = jieduan(frame,left,top,right,bottom)
    img = preprocess(img1)
    feature = extract_cnn_feature(model, img.cuda())

    minsim = -1

    rentidir = '/home/tujh/renti/'
    pkl_file = open('/data/reid/renti/data.pkl', 'rb')
    shujuku = pickle.load(pkl_file)

    #for feature2,filename in shujuku:
    for query in shujuku:
        for fea in shujuku[query]:
            distan = pairwise_distance(feature,fea)
            if minsim > distan or minsim == -1:
                minsim = distan
                id_name = int(query)
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(frame, str(id_name), (left,top), cv2.FONT_HERSHEY_COMPLEX, 6, (255,0,0),2)
    
    # draw shangpin area
    left_x, top_y, right_m, bottom_n = shangpin_area()
    cv2.rectangle(frame, (left_x, top_y), (right_m, bottom_n), (0, 255, 0), 2)

    left_x_2, top_y_2, right_m_2, bottom_n_2 = shangpin_area_huojia2()
    cv2.rectangle(frame, (left_x_2, top_y_2), (right_m_2, bottom_n_2), (255, 0, 0), 2)

    #print(left, top, right, bottom)
    
    return left,top,right,bottom,id_name



normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transformer = T.Compose([T.ToTensor(),
                              normalizer,
                              ])

#the area of shangpin
def shangpin_area():
    #detect_region = config.config['CameraConfig']['camera_7']['detect_region'] #11/17
    #left_x = detect_region[0]
    #top_y = detect_region[1] - 80
    #right_m = detect_region[2]
    #bottom_n = detect_region[3] + 60
    #left_x = 610
    #top_y = 31
    #right_m = 350
    #bottom_n = 369
    #left_x = 678
    #top_y = 4
    #right_m = 300
    #bottom_n = 330
    left_x = 570
    top_y = 3
    right_m = 451
    bottom_n = 231
    return left_x,top_y,right_m,bottom_n

def shangpin_area_huojia2():
    #detect_region = config.config['CameraConfig']['camera_7']['detect_region'] #11/17
    #left_x = detect_region[0]
    #top_y = detect_region[1] - 80
    #right_m = detect_region[2]
    #bottom_n = detect_region[3] + 60
    #left_x = 910
    #top_y = 31
    #right_m = 350
    #bottom_n = 369
    #left_x = 1100
    #top_y = 30
    #right_m = 673
    #bottom_n = 400
    left_x = 410
    top_y = 49
    right_m = 182
    bottom_n = 235
    return left_x,top_y,right_m,bottom_n 

#initial the flag of all the people we detected    
def initial_flag(left,top,right,bottom):
    left_x,top_y,right_m,bottom_n = shangpin_area()
    calcIOU1 = newcalcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
    #calcIOU1 = calcIOU_old(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
    #print(calcIOU1)
    if calcIOU1 > 0.5:
      flag = 1
    else:
      flag = 0
    return flag
    
def initial_flag_out(left,top,right,bottom):
    left_x,top_y,right_m,bottom_n = shangpin_area()
    calcIOU1 = newcalcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
    if calcIOU1 > 0:
      flag = 1
    else:
      flag = 0
    return flag

def initial_flag_huojia2(left,top,right,bottom):
    left_x,top_y,right_m,bottom_n = shangpin_area_huojia2()
    #print('huojia2_initial')
    calcIOU1 = newcalcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
    if calcIOU1 > 0.5:
      flag = 1
    else:
      flag = 0
    return flag
    
def initial_flag_out_huojia2(left,top,right,bottom):
    left_x,top_y,right_m,bottom_n = shangpin_area_huojia2()
    calcIOU1 = newcalcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
    if calcIOU1 > 0:
      flag = 1
    else:
      flag = 0
    return flag

def xuanze_original(res, frame, model, cfg, threadPubMsg, camera_id, dic_change, dic_change_huojia2, huojia1_id, huojia2_id):
    dic = {}
    #left_x, top_y, right_m, bottom_n = shangpin_area()
    #cv2.rectangle(frame, (left_x, top_y), (right_m, bottom_n), (0, 255, 0), 2)
    dic_huojia2 = {}
    #left_x_2, top_y_2, right_m_2, bottom_n_2 = shangpin_area_huojia2()
    #cv2.rectangle(frame, (left_x_2, top_y_2), (right_m_2, bottom_n_2), (0, 255, 0), 2)
    if len(res) == 1:
        result = res[0]
        left,top,right,bottom,id_name = reid_draw(frame, result, model, cfg)
        if dic_change.has_key(id_name):
            if dic_change[id_name] == 1:
                flag = initial_flag_out(left,top,right,bottom)
            else:
                flag = initial_flag(left,top,right,bottom)
        else:
            flag = initial_flag(left, top,right,bottom)
        dic[id_name] = flag

        if dic_change_huojia2.has_key(id_name):
            if dic_change_huojia2[id_name] == 1:
                flag = initial_flag_out_huojia2(left,top,right,bottom)
            else:
                flag = initial_flag_huojia2(left,top,right,bottom)
        else:
            flag = initial_flag_huojia2(left, top,right,bottom)
        dic_huojia2[id_name] = flag

    elif len(res) > 1:
        for item in res:
            result = item
            if (len(result) > 0):
                left,top,right,bottom,id_name = reid_draw(frame, result, model, cfg)
                if dic_change.has_key(id_name):
                    if dic_change[id_name] == 1:
                        flag = initial_flag_out(left,top,right,bottom)
                    else:
                        flag = initial_flag(left,top,right,bottom)
                else:
                    flag = initial_flag(left, top,right,bottom)
                dic[id_name] = flag
                if dic_change_huojia2.has_key(id_name):
                    if dic_change_huojia2[id_name] == 1:
                        flag = initial_flag_out_huojia2(left,top,right,bottom)
                    else:
                        flag = initial_flag_huojia2(left,top,right,bottom)
                else:
                    flag = initial_flag_huojia2(left, top,right,bottom)
                dic_huojia2[id_name] = flag
    return dic,dic_huojia2
    

def people_list(res):
    peolist = []
    left_x,top_y,right_m,bottom_n = shangpin_area()
    for b_b in res:
      global size
      left=int((b_b[0] - b_b[2]/2.0) * size[0])
      top=int((b_b[1]- b_b[3]/2.0) * size[1])
      right=int((b_b[0] + b_b[2]/2.0) * size[0])
      bottom=int((b_b[1] + b_b[3]/2.0) * size[1])
      if calcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)>0:
        x1,x2,x3,x4 = coordinate_IOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
        peolist.append(x1)
        peolist.append(x2)
        peolist.append(x3)
        peolist.append(x4)
    return peolist

#choose the one which is next to the area of shangpin 
def xuanze(res, frame, model, cfg, threadPubMsg, camera_id, dic, change_dic, dic_huojia2, change_dic_huojia2, huojia1_id, huojia2_id):
    for item in res:
        result = item
        # add new person
        if (len(result) > 0):
            left, top, right, bottom, id_name = reid_draw(frame, result, model, cfg)
            in_out_people = diff_dic(dic,change_dic)
            in_out_people_huojia2 = diff_dic(dic_huojia2,change_dic_huojia2)
            if id_name in in_out_people:
                left_x, top_y, right_m, bottom_n = shangpin_area()
                print('huojia1',newcalcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n))
                print(people_list(res))
                customer_name = "name" + str(id_name)
                assert type(id_name) is int  # must be number
                print("set customer message")
                if change_dic[id_name] == 1:
                    flag=1
                    flag1=0
                    flag2=1
                    print("t = {}, flag = {}, flag1 = {}, flag2 = {}, id_name = {}".format(time.time(), flag, flag1, flag2, id_name))
                    threadPubMsg.set_commodity_recognition_trigger_with_image(camera_id=camera_id, person_id=id_name,
                                                                              shelf_id=huojia1_id, flag=flag,
                                                                              flag1=flag1,
                                                                              flag2=flag2,
                                                                              flag_list=people_list(res), frame=frame)
                else:
                    flag=0
                    flag1=1
                    flag2=0
                    print("t = {}, flag = {}, flag1 = {}, flag2 = {}, id_name = {}".format(time.time(), flag, flag1, flag2, id_name))
                    threadPubMsg.set_commodity_recognition_trigger_with_image(camera_id=camera_id, person_id=id_name,
                                                                              shelf_id=huojia1_id, flag=flag,
                                                                              flag1=flag1,
                                                                              flag2=flag2,
                                                                              flag_list=people_list(res), frame=None)

                threadPubMsg.set_customer(name=customer_name, person_id=id_name, camera_id=camera_id, x=left, y=top,
                                          w=right, h=bottom)

            if id_name in in_out_people_huojia2:
                left_x, top_y, right_m, bottom_n = shangpin_area_huojia2()
                print('huojia2', newcalcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n))
                print(people_list(res))
                customer_name = "name" + str(id_name)
                assert type(id_name) is int  # must be number
                print("set customer message")

                if change_dic_huojia2[id_name] == 1:
                    flag=1
                    flag1=0
                    flag2=1
                    print("t = {}, flag = {}, flag1 = {}, flag2 = {},id_name = {}".format(time.time(), flag, flag1, flag2, id_name))
                    threadPubMsg.set_commodity_recognition_trigger_with_image(camera_id=camera_id, person_id=id_name,
                                                                              shelf_id=huojia2_id, flag=flag,
                                                                              flag1=flag1, flag2=flag2,
                                                                              flag_list=people_list(res), frame=frame)
                else:
                    flag=0
                    flag1=1
                    flag2=0
                    print("t = {}, flag = {}, flag1 = {}, flag2 = {},id_name = {}".format(time.time(), flag, flag1, flag2, id_name))
                    threadPubMsg.set_commodity_recognition_trigger_with_image(camera_id=camera_id, person_id=id_name,
                                                                              shelf_id=huojia2_id, flag=flag,
                                                                              flag1=flag1, flag2=flag2,
                                                                              flag_list=people_list(res), frame=None)

                threadPubMsg.set_customer(name=customer_name, person_id=id_name, camera_id=camera_id, x=left, y=top,
                                          w=right, h=bottom)


    return dic,dic_huojia2


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
    checkpoint = load_checkpoint(osp.join(logs_dir, 'checkpoint.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    res = []
    frame_number = 0

    #--datasets
    shujuku = {}
    rentidir = '/data/reid/renti/queries'


    return model


# def callback(msg, param_tuple): # param_tuple
def callback(param_tuple):  # param_tuple
    threadPubMsg = param_tuple[0]
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
    #if frame_number % 8 == 1 or frame_number % 8 == 2 or frame_number % 8 == 3 or frame_number % 8 == 4 or frame_number % 8 == 5 or frame_number % 8 == 6 or frame_number % 8 == 7:
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
    for item in res:
        result = item
        if (len(result) > 0):
            reid_draw(frame, result, model, cfg)


    cv2.imshow('Cam5', cv2.resize(frame, (int(512 * wh_ratio), 512)))

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        
    return res, threadPubMsg, camera_id

class HKCamera(object):
    def __init__(self, ip, name, pw):
        self._ip = ip
        self._name = name
        self._pw = pw
        HKIPcamera.init(self._ip, self._name, self._pw)

    def getFrame(self):
        # HKIPcamera.init(self._ip, self._name, self._pw)
        frame = HKIPcamera.getframe()
        return frame


def main(camera_id):
    ip = str('192.168.0.12')
    name = str('admin')
    pw = str('a1234567')
    camera = HKCamera(ip, name, pw)

    threadPubMsg = pubmsg.MsgPublishClass(cameraID=camera_id)
    threadPubMsg.setDaemon(True)
    threadPubMsg.start()

    model = loadDataset()


    cfg = Darknet('cfg/yolov3.cfg')
    cfg.load_weights('yolov3.weights')
    cfg.cuda()
    # global frame_number
    frame_number2 = [0]
    flag = [0]
    bridge = CvBridge()

    threadPubMsg.set_commodity_recognition_trigger(camera_id=12, person_id=0, flag=1, flag1=0, flag2=0)

    dic_change = {}
    dic_change_huojia2 = {}
    huojia1_id = 1
    huojia2_id = 2
    while not rospy.is_shutdown():
        frame = camera.getFrame()
        frame = np.array(frame)
        frame = cv2.resize(frame, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)

        res, threadPubMsg, camera_id=callback((threadPubMsg, cfg, model, frame_number2, bridge, camera_id, flag, frame))
        # if no person in the scene, send(0, 0, 0)
        #print("res: {}".format(res))
        if res == []:
            #print("alallalallal")
            threadPubMsg.set_commodity_recognition_trigger_with_image(camera_id=camera_id, person_id=-1,
                                                                      shelf_id=-1, flag=0, flag1=0, flag2=0,
                                                                      flag_list=[], frame=None)
            continue
        dic, dic_huojia2 = xuanze_original(res, frame, model, cfg, threadPubMsg, camera_id, dic_change, dic_change_huojia2, huojia1_id, huojia2_id)
        #print("11111122222222")
        #print('dic',dic)
        #print('dic_change',dic_change)
        #print(compare_dic(dic, dic_change))
        if compare_dic(dic,dic_change) == False and compare_dic(dic_huojia2,dic_change_huojia2) == False:
            pass
        else:
            #print("track")
            #print("22222233333333")
            dic, dic_huojia2 = xuanze(res, frame, model, cfg, threadPubMsg, camera_id, dic, dic_change, dic_huojia2, dic_change_huojia2, huojia1_id, huojia2_id)
            #print("22222233333333")
            #dic, dic_huojia2 = xuanze(res, frame, model, cfg, threadPubMsg, camera_id, dic, dic_change, huojia1_id, huojia2_id)
        dic_change = dic
        dic_change_huojia2 = dic_huojia2


    HKIPcamera.release()



if __name__ == '__main__':
    rospy.init_node('det_reid', anonymous=True)
    main(camera_id = 11)

    rospy.spin()

