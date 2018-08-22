import pdb
import time
import argparse
import os
import datasets
from PIL import Image
import numpy as np
import json

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

    if args.debug == 'True':
        num_workers = 0
    
    transform = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    if args.data == "VOC":
        test_root = '/media/NAS/dataset/PASCALVOC/VOCdevkit/07+12/test.txt'
        if args.loss_fn == 'sigmoid':
            label = ['aeroplane','bicycle','bird','boat','bottle','bus','car',
                 'cat','chair','cow','diningtable','dog','horse','motorbike'
                 ,'person','pottedplant','sheep','sofa','train','tvmonitor',]
            num_classes = 20
        elif args.loss_fn == 'softmax':
            label = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car',
                 'cat','chair','cow','diningtable','dog','horse','motorbike'
                 ,'person','pottedplant','sheep','sofa','train','tvmonitor',]
            num_classes = 21
        if not len(label) == num_classes:
            print("The label number is wrong")

    elif args.data == "COCO":
        test_root = '/media/NAS/dataset/COCO/minival2014/test.txt'
        label_prototxt = '/media/NAS/dataset/COCO/coco_api_caffe_2014/coco/labelmap_coco.txt'
        labels = {}
        if args.loss_fn == 'sigmoid':
            num_classes = 80
        elif args.loss_fn == 'softmax':
            num_classes = 81
        with open(label_prototxt) as file:
            while True:
                label = file.readline()
                if not label: break
                label = label.rstrip()
                label = label.split(",")
                labels[int(label[1])] = [int(label[0]), label[2]]

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
    if args.data == 'VOC':
        result = ''
    elif args.data == 'COCO':
        result = ['[']

    for img_idx in lines:
        img_path = img_idx.rstrip()
        labelpath = img_path.replace('images','labels').replace('JPEGImages'
                    ,'labels').replace('.jpg','.txt').replace('.png','.txt')
        img = Image.open(img_path).convert('RGB')
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
        if args.data == 'VOC':
            image_id = img_path[-10:-4]

            if not os.path.exists(save_path+'/val_epoch_{}'.format(args.epoch)):
                os.mkdir(save_path+'/val_epoch_{}'.format(args.epoch))

            if score.shape[0] != 0:
                box_preds = loc_preds_nms.cpu().detach().numpy()
                xmin = box_preds[:,0]*img.size[0]/scale
                ymin = box_preds[:,1]*img.size[1]/scale
                xmax = box_preds[:,2]*img.size[0]/scale
                ymax = box_preds[:,3]*img.size[1]/scale
                xmin[xmin < 0] = 0
                ymin[ymin < 0] = 0
                xmax[xmax > img.width] = img.width
                ymax[ymax > img.height] = img.height
                box_preds[:,0] = xmin
                box_preds[:,1] = ymin
                box_preds[:,2] = xmax
                box_preds[:,3] = ymax
                box_preds = box_preds.astype(str)
                box_preds = np.ndarray.tolist(box_preds)
                category_preds = cls_preds_nms.cpu().detach().numpy().astype(str)
                category_preds = np.ndarray.tolist(category_preds)
                score_preds = score.cpu().detach().numpy().astype(str)
                score_preds = np.ndarray.tolist(score_preds)
                for i in range(len(score_preds)):
                    ## [Image ID, Class, Box()]
                    result = ' '.join([image_id,score_preds[i],
                                       box_preds[i][0], box_preds[i][1],
                                       box_preds[i][2],box_preds[i][3]]) + '\n'
                    f = open(save_path+'/val_epoch_{}/comp3_det_test_{}.txt'\
                            .format(args.epoch,label[int(category_preds[i])]), 'a')
                    f.write(result)

            print(img_path)

        elif args.data == 'COCO':
            image_id = int(img_path[-16:-4])
            if not os.path.exists(save_path+'/val_epoch_{}'.format(args.epoch)):
                os.mkdir(save_path+'/val_epoch_{}'.format(args.epoch))

            if score.shape[0] != 0:
                box_preds = loc_preds_nms.cpu().detach().numpy()
                xmin   = box_preds[:,0]*img.size[0]/scale
                ymin   = box_preds[:,1]*img.size[1]/scale
                xmax   = box_preds[:,2]*img.size[0]/scale
                ymax   = box_preds[:,3]*img.size[1]/scale
                width  = xmax-xmin
                height = ymax-ymin

                box_preds[:,0] = xmin
                box_preds[:,1] = ymin
                box_preds[:,2] = width
                box_preds[:,3] = height
                box_preds = np.ndarray.tolist(box_preds)
                category_preds = cls_preds_nms.cpu().detach().numpy()
                category_preds = np.ndarray.tolist(category_preds)
                score_preds = score.cpu().detach().numpy()
                score_preds = np.ndarray.tolist(score_preds)

                for idx in range(len(box_preds)):
                    j = str({"image_id": image_id, 
                             "category_id": labels[category_preds[idx]+1][0],
                             "bbox": box_preds[idx],
                             "score": score_preds[idx]})
                    result.append(j)
            else: continue
            print(img_path)

    if args.data == 'COCO':
        f = open(save_path+'/val_epoch_{}/coco_results.txt'.format(args.epoch), 'w')
        json.dump(result,f)

if __name__ == '__main__':
    main()
