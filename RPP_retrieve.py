import numpy as np
import os.path as osp
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.utils.serialization import load_checkpoint
from reid import models
from reid.feature_extraction import extract_cnn_feature
from numpy import average, linalg, dot
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import cv2

def oushi_distance(fea1,fea2):
    #print(fea1)
    fea1 = torch.squeeze(fea1,0)
    fea1 = torch.squeeze(fea1,-1)
    fea2 = torch.squeeze(fea2,0)
    fea2 = torch.squeeze(fea2,-1)
    #print(fea1.shape)
    #print(fea1)
    fea = [fea1,fea2]
    dist = np.linalg.norm(fea1-fea2)
    return dist


def cosine_distance(fea1,fea2):
    #print(fea1)
    fea1 = torch.squeeze(fea1,0)
    fea1 = torch.squeeze(fea1,-1)
    fea2 = torch.squeeze(fea2,0)
    fea2 = torch.squeeze(fea2,-1)
    #print(fea1.shape)
    #print(fea1)
    fea = [fea1,fea2]
    vectors= []
    norms=[]
    for fe in fea:
        vector=[]
        for i in range(256):
            vector.append(torch.mean(fe[i]))
        vectors.append(vector)
        norms.append(linalg.norm(vector,2))
    a,b,=vectors
    a_norm,b_norm = norms
    res=dot(a/a_norm,b/b_norm)
    return res

def cosine_distance2(fea1,fea2):
    fea1 = torch.squeeze(fea1,0)
    fea1 = torch.squeeze(fea1,-1)
    fea2 = torch.squeeze(fea2,0)
    fea2 = torch.squeeze(fea2,-1)
    f=torch.norm(fea1,2,1,True)
    c=torch.norm(fea2,2,1,True)
    g=f.expand_as(fea1)
    d=c.expand_as(fea2)
    l=fea1.div(g)
    e=fea2.div(d)
    j=torch.t(e)
    m=np.dot(l,e[0])
    print(m)
    n=np.dot(l.numpy(),e[1].numpy())
    print(n.sum(0))
    print(n.cumprod(0))
    return n.size

def pairwise_distance(fea1, fea2):
    fea1 = torch.squeeze(fea1,0)
    fea1 = torch.squeeze(fea1,-1)
    fea2 = torch.squeeze(fea2,0)
    fea2 = torch.squeeze(fea2,-1)
    x = fea1
    y = fea2
    #m, n = x.size(0), y.size(0)
    m,n=1,1
    x = x.view(m, -1)
    y = y.view(n, -1)
    print(x.shape)
    dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
    print(dist.shape)
    dist.addmm_(1, -2, x, y.t())
    print(dist.shape)
    return torch.sum(dist)

def main():
    model = models.create('resnet50_rpp', num_features=num_features, dropout=dropout, num_classes=num_classes, cut_at_pooling=False, FCN=True, T=T, dim=dim)
    #model = model.cuda()
    checkpoint = load_checkpoint(osp.join(logs_dir, 'checkpoint.pth.tar'))

# #======================added by syf, to remove undeployed layers=============
#     model_dict = model.state_dict()
#     checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
#     model_dict.update(checkpoint_load)
#     model.load_state_dict(model_dict)       
#     #        model.load_state_dict(checkpoint['state_dict'])


    model.load_state_dict(checkpoint['state_dict'])  #in checkpoint, I change gpu into cpu
    #model.load_state_dict(torch.load('model_best.pth.tar', map_location=lambda storage, loc:storage))
    img = cv2.imread(osp.join(img_dir, img_name))
    img = np.transpose(img, (2,0,1)).astype(np.float32)

   
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)
    print(img.shape)
    #img = img.cuda()
    

    feature = extract_cnn_feature(model, img)
    print(feature.shape)

    img2 = cv2.imread(osp.join(img_dir2, img_name2))
    img2 = np.transpose(img2, (2,0,1)).astype(np.float32)

    img2 = torch.from_numpy(img2)
    img2 = torch.unsqueeze(img2, 0)
    feature2 = extract_cnn_feature(model,img2)
    
    print(oushi_distance(feature,feature2))
    print(cosine_distance2(feature,feature2))
    print(pairwise_distance(feature,feature2))

if __name__ == '__main__':
    logs_dir = 'market-1501-Exper33/RPP/'
    #logs_dir = 'logs/market-1501/RPP/'
    img_dir = '/home/luis/datasets/renti'
    num_features = 256
    num_classes = 751
    T = 1
    dim = 256
    dropout = 0.5
    img_name = '1.jpg'

    img_dir2 = '/home/luis/datasets/renti'
    img_name2 = '2.jpg'
    main()
