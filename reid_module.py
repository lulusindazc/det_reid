#---------------------------------
# @time             : 2018-11-29
# @Author           : Hanrong Ye
# @Description      : Only reid and draw id
# @last_modification: 2018-11-29
#---------------------------------

import cv2
from utils import *
from darknet import Darknet

import os.path as osp
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature

import pickle
import torchvision.transforms as T

# import scripts.publish_msg as pubmsg


# frame_number = 0
class_names = None
# m=None



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


def calcIOU_old(one_x, one_y, one_m, one_n, two_x, two_y, two_m, two_n):
    one_w = abs(one_x - one_m)
    one_h = abs(one_y - one_n)
    two_w = abs(two_x - two_m)
    two_h = abs(two_y - two_n)
    if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) and (abs(one_y - two_y) < ((one_h + two_h) / 2.0))):
        lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
        lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

        rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
        rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

        inter_w = abs(rd_x_inter - lu_x_inter)
        inter_h = abs(lu_y_inter - rd_y_inter)

        inter_square = inter_w * inter_h
        union_square = (one_w * one_h) + (two_w * two_h) - inter_square

        calcIOU = inter_square / union_square * 1.0
        print("calcIOU:", calcIOU)
    else:
        # print("No interse   ction!")
        calcIOU = -1
        # print("calcIOU:", calcIOU)

    return calcIOU


def calcIOU(p_x, p_y, p_bx, p_by, c_x, c_y, c_bx, c_by):
    c_x = 960 - c_x
    c_bx = 960 - c_bx
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

####
normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
test_transformer = T.Compose([T.ToTensor(),
                              normalizer,
                              ])

def preprocess(img):
    img = cv2.resize(img, (128, 256))
    img = test_transformer(img)
    img = torch.unsqueeze(img, 0)
    # print(img.shape)
    return img

####


def reid_draw(frame, b_b, model, shujuku, frame_size):
    size = frame_size

    left=int((b_b[0] - b_b[2]/2.0) * size[0])
    top=int((b_b[1]- b_b[3]/2.0) * size[1])
    right=int((b_b[0] + b_b[2]/2.0) * size[0])
    bottom=int((b_b[1] + b_b[3]/2.0) * size[1])
    if top>=bottom or left>=right or top<=0 or left<=0:
        return
        
    img1 = jieduan(frame,left,top,right,bottom)
    img = preprocess(img1)
    feature = extract_cnn_feature(model, img.cuda())

    minsim = -1

    #for feature2,filename in shujuku:
    customer_id = -1
    for query in shujuku:
        for fea in shujuku[query]:
            distan = pairwise_distance(feature,fea)
            if minsim > distan or minsim == -1:
                minsim = distan
                customer_id = int(query)
    #add to debug customer_id
    #customer_id = 1
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
    cv2.putText(frame,str(customer_id),(left,top),cv2.FONT_HERSHEY_COMPLEX,6,(255,0,0),2)

    customer_name = "name_"+str(customer_id)
    assert type(customer_id) is int # must be number

    # return frame
    #print("pub msg finish")
    return customer_name, customer_id

