import pdb
import time
import argparse
import os
import datasets
from PIL import Image, ImageDraw
import numpy as np
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from model import *
from loss import FocalLoss
from utils import freeze_bn
from logger import Logger
from encoder import DataEncoder

def load_label(labelpath, img):
    bbx = np.loadtxt(labelpath) # load the label 
    if len(bbx.shape) == 1:
        bbx = np.reshape(bbx,[1,5])  # if the label is only one, we have to resize the shape of the bbx
    x1 = (bbx[:,1] - bbx[:,3]/2)*img.width  # calculate the original label x1_min
    x2 = (bbx[:,1] + bbx[:,3]/2)*img.width  # calculate the original label x2_max
    y1 = (bbx[:,2] - bbx[:,4]/2)*img.height # calculate the original label y1_min
    y2 = (bbx[:,2] + bbx[:,4]/2)*img.height # calculate the original label y2_max
    bbx[:,1] = x1   # xmin
    bbx[:,2] = y1   # ymin
    bbx[:,3] = x2   # xmax
    bbx[:,4] = y2   # ymax
    return bbx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-data', type=str, default='VOC')
    parser.add_argument('--loss_fn', '-loss', type=str, default='sigmoid')
    parser.add_argument('--epoch', '-e', type=str, default='None')
    parser.add_argument('--debug', '-d', type=str, default='False')
    parser.add_argument('--weight_path', '-w', type=str, default='None')
    args = parser.parse_args()

    scale = 600
    use_cuda = torch.cuda.is_available() 
    num_workers = os.cpu_count()
    batch_size = 1
    gpus = [0,1]
    save_path = args.weight_path
    if not os.path.exists(save_path+'/test_img/'):
        os.mkdir(save_path+'/test_img/')

    if args.debug == 'True':
        num_workers = 0
    
    transform = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    if args.data == "VOC":
        test_root = '/media/NAS/dataset/PASCALVOC/VOCdevkit/07+12/test.txt'
        if args.loss_fn == 'sigmoid':
            voc_label = ['aeroplane','bicycle','bird','boat','bottle','bus','car',
                 'cat','chair','cow','diningtable','dog','horse','motorbike'
                 ,'person','pottedplant','sheep','sofa','train','tvmonitor',]
            num_classes = 20
        elif args.loss_fn == 'softmax':
            voc_label = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car',
                 'cat','chair','cow','diningtable','dog','horse','motorbike'
                 ,'person','pottedplant','sheep','sofa','train','tvmonitor',]
            num_classes = 21
        color_label = [(  0,   0,   0),
                       (  0,   0,   0),
                       (111,  74,   0),
                       ( 81,   0,  81),
                       (128,  64, 128),
                       (244,  35, 232),
                       (230, 150, 140),
                       ( 70,  70, 700),
                       (102, 102, 156),
                       (190, 153, 153),
                       (150, 120,  90),
                       (153, 153, 153),
                       (250, 170,  30),
                       (220, 220,   0),
                       (107, 142,  35),
                       ( 52, 151,  52),
                       ( 70, 130, 180),
                       (220,  20,  60),
                       (  0,   0, 142),
                       (  0,   0, 230),
                       (119,  11,  32)]

    elif args.data == "COCO":
        test_root = '/media/NAS/dataset/COCO/minival2014/test.txt'
        if args.loss_fn == 'sigmoid':
            num_classes = 80
        elif args.loss_fn == 'softmax':
            num_classes = 81

    global device
    device = torch.device("cuda" if use_cuda else "cpu")

    print('Loading model..')
    if args.data == 'VOC':
        weights = './{}/retina_{}.pth'.format(args.weight_path,args.epoch)
    elif args.data == 'COCO':
        weights = './{}/retina_{}.pth'.format(args.weight_path,args.epoch)

    model = RetinaNet(num_classes)

    checkpoint = torch.load(weights)
    if use_cuda:
        if len(gpus) >= 1:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)
        model.cuda()
    model.load_state_dict(checkpoint['state_dict'])
    print('\nTest')

    with open(test_root, 'r') as file:
        lines = file.readlines()

    encoder = DataEncoder(args.loss_fn)
    model.eval()
    result = ''
    for img_idx in lines[:100]:
        img_path = img_idx.rstrip()
        labelpath = img_path.replace('images','labels').replace('JPEGImages'
                    ,'labels').replace('.jpg','.txt').replace('.png','.txt')
        img = Image.open(img_path).convert('RGB')
        label = load_label(labelpath, img)

        input_img = img.resize((scale,scale))

        input_img = transform(input_img)
        data = torch.zeros(1,3,input_img.shape[1],input_img.shape[2])
        data[0] = input_img
        inputs = data.to(device)
        loc_preds_split, cls_preds_split = model(inputs.cuda())
        loc_preds_nms, cls_preds_nms, score = encoder.decode(loc_preds_split,
                                                             cls_preds_split,
                                                             data.shape,
                                                             data[0].shape,
                                                             0)
        image_id = img_path[-10:]

        if not os.path.exists(save_path+'/test_img/val_epoch_{}'\
                        .format(args.epoch)):
            os.mkdir(save_path+'/test_img/val_epoch_{}'.format(args.epoch))

        if score.shape[0] != 0:
            box_preds = loc_preds_nms.cpu().detach().numpy().astype(int)
            box_preds = np.ndarray.tolist(box_preds)
            category_preds = cls_preds_nms.cpu().detach().numpy().astype(str)
            c = np.ndarray.tolist(category_preds)
            score_preds = score.cpu().detach().numpy().astype(str)
            score_preds = np.ndarray.tolist(score_preds)

        else:
            box_preds = []
            c = []
            score_preds = []

        new_img = cv2.imread(img_path)
        for i in range(int(label.shape[0])):
            coor_min = (int(label[i][1]), int(label[i][2]))
            coor_max = (int(label[i][3]), int(label[i][4]))
            cls = int(label[i][0])
            # cv2.rectangle(new_img, coor_min, coor_max, color_label[cls], 2)
            cv2.rectangle(new_img, coor_min, coor_max, (250,0,0), 2)
            cv2.putText(new_img, voc_label[cls] + ' | ' + 'GT', (coor_min[0]+5, coor_min[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        if len(box_preds) > 0:
            for idx, box_pred in enumerate(box_preds):
                box_pred_xmin = int(float(box_pred[0]))
                if box_pred_xmin < 0: box_pred_xmin = 0
                box_pred_ymin = int(float(box_pred[1]))
                if box_pred_ymin < 0: box_pred_ymin = 0
                box_pred_xmax = int(float(box_pred[2]))
                if box_pred_xmax < 0: box_pred_xmax = 0
                box_pred_ymax = int(float(box_pred[3]))
                if box_pred_ymax < 0: box_pred_ymax = 0
                cls_idx = int(category_preds[idx])
                box_pred_min = (int(box_pred_xmin), int(box_pred_ymin))
                box_pred_max = (int(box_pred_xmax), int(box_pred_ymax))
                box_pred_min = (int(box_pred_xmin*new_img.shape[1]/scale), int(box_pred_ymin*new_img.shape[0]/scale))
                box_pred_max = (int(box_pred_xmax*new_img.shape[1]/scale), int(box_pred_ymax*new_img.shape[0]/scale))
                cls_name = voc_label[cls_idx]
                cls_color = color_label[cls_idx]
                box_coor = (box_pred_min, box_pred_max)
                conf = score_preds[idx][:4]
                # cv2.rectangle(new_img, box_pred_min, box_pred_max, cls_color, 2)
                cv2.rectangle(new_img, box_pred_min, box_pred_max, (0,250,0), 2)
                cv2.putText(new_img, cls_name + ' | ' + conf, (box_pred_min[0]+5, box_pred_min[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        new_path = save_path+'/test_img/val_epoch_{}/'.format(args.epoch) + image_id
        cv2.imwrite(new_path, new_img)
        print(image_id)


if __name__ == '__main__':
    main()
