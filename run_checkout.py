# ---------------------------------
# @time             : 2018-11-03
# @Author           : Huang weibo, Wison
# @Description      : Give an interface to one cam
# @last_modification: 2018-11-03
# ---------------------------------

import HKIPcamera
import cv2

import rospy  # Bingoren
from sensor_msgs.msg import CompressedImage  # Bingo
from cv_bridge import CvBridge, CvBridgeError  # Bingo

from torchvision import transforms as T
from utils import *
from darknet import Darknet

import os.path as osp
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature

# import scripts.publish_msg as pubmsg
import publish_msg.publish_msg as pubmsg
import pickle

from touching_ai_develop.msg import PaymentMsgAsk

# import scripts.publish_msg as pubmsg

fall_time = time.time()
# frame_number = 0
class_names = None
# m=None

# global isAskPerson
isAskPerson = False

camrea5_frame = None


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# torch.cuda.set_device(1)

def playsound(SOUND_IP):
    cmd = 'sshpass -p "hri" ssh hri@{} "python play.py"'.format(SOUND_IP)
    os.system(cmd)


def pairwise_distance(fea1, fea2):
    fea1 = torch.squeeze(fea1, 0)
    fea1 = torch.squeeze(fea1, -1)
    fea2 = torch.squeeze(fea2, 0)
    fea2 = torch.squeeze(fea2, -1)
    x = fea1
    y = fea2
    # m, n = x.size(0), y.size(0)
    m1, n = 1, 1
    x = x.view(m1, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m1, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m1).t()
    dist.addmm_(1, -2, x, y.t())
    return torch.sum(dist)


def jieduan(img, left, top, right, bottom):
    imgg = np.zeros((bottom - top, right - left, 3))
    # for i in range(top,bottom):
    #  for j in range(left,right):
    #    for k in range(3):
    #      imgg[i-top][j-left][k] = img[i][j][k]
    imgg = img[top:bottom, left:right, :]
    return imgg


def calcIOU(p_x, p_y, p_bx, p_by, c_x, c_y, c_bx, c_by):
    c_x = c_x
    c_bx = c_bx
    condition1 = p_x >= c_x and p_x <= c_bx
    condition2 = p_bx >= c_x and p_bx <= c_bx
    condition3 = p_y >= c_y and p_y <= c_by
    condition4 = p_by >= c_y and p_by <= c_by
    print(p_x, p_y, p_bx, p_by, c_x, c_y, c_bx, c_by)

    if (condition1 and condition3) or (condition1 and condition4) or \
            (condition2 and condition3) or (condition2 and condition4):
        calcIOU = 1
    else:
        calcIOU = -1

    return calcIOU


####
normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transformer = T.Compose([T.ToTensor(),
                              normalizer,
                              ])


def preprocess(img):
    img = cv2.resize(img, (128, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # img[:,:,::-1]
    img = test_transformer(img)
    img = torch.unsqueeze(img, 0)
    # print(img.shape)
    return img


####


normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transformer = T.Compose([T.ToTensor(),
                              normalizer,
                              ])



def reid_draw(frame, b_b, model, cfg, shujuku, threadPubMsg, camera_id):
    global size

    # print("size = ", size)
    # print("b_b = ", b_b)
    # print model
    id_name = 0

    cfg.cuda()

    left = int((b_b[0] - b_b[2] / 2.0) * size[0])
    top = int((b_b[1] - b_b[3] / 2.0) * size[1])
    right = int((b_b[0] + b_b[2] / 2.0) * size[0])
    bottom = int((b_b[1] + b_b[3] / 2.0) * size[1])
    if top >= bottom or left >= right or top <= 0 or left <= 0:
        return 0, 0, 0, 0, 0

    ratio = float(bottom - top) / (right - left)
    # print(ratio)
    # if ratio < 2.0:
    #     # print('filtered out')
    #     return left, top, right, bottom, 999


    if bottom > 530:
        ratio = float(bottom - top) / (right - left)
        #print("ratio is: {}".format(ratio))
        if ratio < 1.5:
            #print("ratio is: {}".format(ratio))
            #print('filtered out')
            return left, top, right, bottom, 999


    img1 = jieduan(frame, left, top, right, bottom)

    img = preprocess(img1)

    feature = extract_cnn_feature(model, img.cuda())

    minsim = -1
    # id_name = 1
    rentidir = '/home/tujh/renti/'
    # for feature2,filename in shujuku:

    for query in shujuku:
        for fea in shujuku[query]:
            distan = pairwise_distance(feature, fea)
            if minsim > distan or minsim == -1:
                minsim = distan
                id_name = query
            '''
        distan=0
        discount=0
        for fea in shujuku[query]:
            distan += pairwise_distance(feature,fea)
            discount+=1
        if discount!=0:
          distan=distan/discount
        if minsim > distan or minsim == -1 or distan!=0:
            minsim = distan
            id_name = query
            '''

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(frame, str(id_name), (left, top), cv2.FONT_HERSHEY_COMPLEX, 6, (255, 0, 0), 2)
    # print(frame.shape)
    # left_x = 450
    # top_y = 30
    # right_m = 570
    # bottom_n = 130

    left_x = 86
    top_y = 130
    right_m = 413
    bottom_n = 479
    #cv2.rectangle(frame, (left_x, top_y), (right_m, bottom_n), (0, 255, 0), 2)
    calcIOU1 = calcIOU(left, top, right, bottom, left_x, top_y, right_m, bottom_n)
    print("calIOU : {}".format(calcIOU1))

    if calcIOU1 > 0.8:
        flag_check = 1
    else:
        flag_check = 0

    print('flag', flag_check)
    print(calcIOU1)
    print("set customer message")
    customer_name = "name" + str(id_name)
    customer_id = id_name  # number
    global isAskPerson
    print(isAskPerson)
    # if isAskPerson == True and flag_check==1:
    # cv2.imwrite('/home/zhaocy/catkin_touching_AI/checkout_img/checkout_' + str(customer_id) + '.jpg')
    if flag_check == 1:
        threadPubMsg.set_customer(name=customer_name, person_id=customer_id, camera_id=camera_id,
                                  x=left, y=top, w=right, h=bottom)
        threadPubMsg.set_commodity_recognition_trigger(camera_id=camera_id, person_id=customer_id,
                                                       flag=1, flag1=0, flag2=0)
        print('save_images')
        cv2.imwrite('/home/zhaocy/catkin_touching_AI/checkout_img/checkout_' + str(customer_id) + '.jpg', frame)
    else:
        # print(1111111111111)
        threadPubMsg.set_customer(name=customer_name, person_id=customer_id, camera_id=0, x=left, y=top,
                                  w=right, h=bottom)
        threadPubMsg.set_commodity_recognition_trigger(camera_id=camera_id, person_id=customer_id,
                                                       flag=0, flag1=0, flag2=0)
    # return frame
    return left, right, top, bottom, id_name


def loadDataset():
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

    return model


def callback(param_tuple):  # param_tuple
    threadPubMsg = param_tuple[0]
    cfg = param_tuple[1]
    model = param_tuple[2]
    # shujuku = param_tuple[3]
    # frame_number = param_tuple[4]
    frame_number_list = param_tuple[3]
    bridge = param_tuple[4]
    camera_id = param_tuple[5]
    # flag = param_tuple[7]

    # header_cur = msg.header
    # frame = bridge.compressed_imgmsg_to_cv2(msg)  # for compressed
    # frame = bridge.imgmsg_to_cv2(msg)
    frame = param_tuple[6]

    # cv2.imshow('Fall detection', cv2.resize(frame, (int(512), 512)))
    # cv2.waitKey(3)
    # return True

    shape = frame.shape

    # header_cur = msg.header
    # frame = bridge.compressed_imgmsg_to_cv2(msg)  # for compressed
    # frame = bridge.imgmsg_to_cv2(msg)

    # cv2.imshow('Fall detection', cv2.resize(frame, (int(512), 512)))
    # cv2.waitKey(3)
    # return True

    shape = frame.shape  # (1080, 1920, 3)
    # (1080, 1920, 3)

    global size
    size = (shape[1], shape[0])

    # global frame_number
    frame_number_list[0] = frame_number_list[0] + 1
    frame_number = frame_number_list[0]
    wh_ratio = frame.shape[1] / frame.shape[0]
    # Quit when the input video file ends
    if type(frame) != np.ndarray:  # is frame readed is null,   go to the next loop
        print('********************')
        # continue
        return True

    try:
        pkl_file = open('/data/reid/renti/data.pkl', 'rb')
        shujuku = pickle.load(pkl_file)
        pkl_file.close()
    except:
        pkl_file = open('/data/reid/renti/data_bu.pkl', 'rb')
        shujuku = pickle.load(pkl_file)
        pkl_file.close()

    # print(frame_number)
    # detect per 8 frame
    if frame_number % 8 == 1 or frame_number % 8 == 2 or frame_number % 8 == 3 or frame_number % 8 == 4 or frame_number % 8 == 5 or frame_number % 8 == 6 or frame_number % 8 == 7:
        return True
    cfg.cuda()
    # print cfg.width, cfg.height
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
    ################yyyyyyyyyyy
    # print('the whole running time is: '+str(time.time()-start))
    res = []
    for item in r:
        if class_names[item[6]] == 'person' or class_names[item[6]] == 'dog' or class_names[item[6]] == 'cat' or \
                class_names[item[6]] == 'horse':
            res.append(item)

    # get the max rectangle
    result = []
    if res == []:
        print('flag', 0)
        threadPubMsg.set_commodity_recognition_trigger(camera_id=camera_id, person_id=-1,
                                                       flag=0, flag1=0, flag2=0)

    for item in res:
        result = item
        if (len(result) > 0):
            # frame = reid_draw(frame, result, model, cfg, shujuku, threadPubMsg, camera_id)
            lef, righ, to, botto, id_nam = reid_draw(frame, result, model, cfg, shujuku, threadPubMsg, camera_id)
            if lef == 0 and righ == 0 and to == 0 and botto == 0:
                continue
            #cv2.rectangle(frame, (lef, to), (righ, botto), (255, 0, 0), 2)
            # cv2.rectangle(frame, (960-right, top), (960-left, bottom), (255, 0, 0), 2)
            #cv2.putText(frame, str(id_nam), (lef, to), cv2.FONT_HERSHEY_COMPLEX, 6, (255, 0, 0), 2)
    # if type(frame) == None:
    #     return

    cv2.rectangle(frame, (86, 130), (413, 479), (0, 255, 0), 2)
    cv2.imshow('Camcheckout', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


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
    #
    ip = str('192.168.0.12')
    name = str('admin')
    pw = str('a1234567')
    camera = HKCamera(ip, name, pw)

    threadPubMsg = pubmsg.MsgPublishClass(camera_id)
    threadPubMsg.setDaemon(True)
    threadPubMsg.start()

    model = loadDataset()

    cfg = Darknet('cfg/yolov3.cfg')
    cfg.load_weights('yolov3.weights')
    cfg.cuda()
    print "main cfg: ", cfg.width, cfg.height

    # global frame_number
    frame_number2 = [0]
    # flag = [0]
    bridge = CvBridge()

    while not rospy.is_shutdown():
        frame = camera.getFrame()
        frame = np.array(frame)
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
        # cv2.imshow('4', frame)
        # cv2.waitKey(1)
        callback((threadPubMsg, cfg, model, frame_number2, bridge, camera_id, frame))

    HKIPcamera.release()


def callback_payment(msg):
    header = msg.header
    global isAskPerson
    isAskPerson = msg.askPayPerson
    # print(isAskPerson)


if __name__ == '__main__':
    rospy.init_node('PaymentPersonFeedbackNode', anonymous=True)
    # / PaymentAskPerson
    rospy.Subscriber("/PaymentAskPerson", PaymentMsgAsk, callback_payment)

    main(camera_id=10)

    rospy.spin()
