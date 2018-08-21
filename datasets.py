
import os
import pdb
import random

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw

from encoder import DataEncoder

#coco_label = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
#
class LoadDataset(Dataset):
    def __init__(self, root, scale=None, shuffle=True, transform=None, train=False, \
            batch_size=16, num_workers=2):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples = len(self.lines)
        self.transform = transform
        self.train = train
        self.scale = scale
        self.batch_size = batch_size

        self.encoder = DataEncoder()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        img, label = load_data_detection(imgpath, self.scale, self.train)
        label = torch.from_numpy(label)

        if self.transform is not None:
            img = self.transform(img)
        boxes = label[:,1:]    # split the bbx label and cls label
        labels = label[:,0]
        return (img, boxes, labels)

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]
        
        inputs = []
        ori_img_shape = []
        for i in range(len(imgs)):
            w = imgs[i].shape[1]
            h = imgs[i].shape[2]
            input = torch.zeros(1,3,h,w)
            inputs.append(input)
            ori_img_shape.append(imgs[i].shape)
        torch.stack(inputs)

        w = self.scale
        h = self.scale
        num_img = len(imgs)
        inputs = torch.zeros(num_img,3,h,w)
        loc_targets = []
        cls_targets = []
        for i in range(num_img):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i],labels[i],input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
            
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets), ori_img_shape


def load_data_detection(imgpath, scale, train):
    labelpath = imgpath.replace('images','labels').replace('JPEGImages','labels').replace('.jpg','.txt').replace('.png','.txt')
    img = Image.open(imgpath).convert('RGB')
    resized_img,flip = data_augmentation(img, scale, train)  # augment the img 
    label = load_label(labelpath, flip, img, resized_img) # load the label
    return resized_img, label

def data_augmentation(img, scale, train):
    flip = random.randint(1,10000)%2 # apply the flip
    img = img.resize((scale,scale))
    if train:
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img, flip

def load_label(labelpath, flip, img, resized_img):
    bbx = np.loadtxt(labelpath) # load the label 
    if len(bbx.shape) == 1:
        bbx = np.reshape(bbx,[1,5])  # if the label is only one, we have to resize the shape of the bbx
    x1 = (bbx[:,1] - bbx[:,3]/2)*img.width  # calculate the original label x1_min
    x2 = (bbx[:,1] + bbx[:,3]/2)*img.width  # calculate the original label x2_max
    y1 = (bbx[:,2] - bbx[:,4]/2)*img.height # calculate the original label y1_min
    y2 = (bbx[:,2] + bbx[:,4]/2)*img.height # calculate the original label y2_max
    r_x1 = x1 * resized_img.width / img.width     # calculate the resized label x1_min
    r_x2 = x2 * resized_img.width / img.width     # calculate the resized label x2_max
    r_y1 = y1 * resized_img.height / img.height   # calculate the resized label y1_min
    r_y2 = y2 * resized_img.height / img.height   # calculate the resized label y2_max
    bbx[:,1] = ((r_x1 + r_x2)/2)   # center_x
    bbx[:,2] = ((r_y1 + r_y2)/2)   # center_y
    bbx[:,3] = ((r_x2 - r_x1))     # width
    bbx[:,4] = ((r_y2 - r_y1))     # height
    if flip:
        bbx[:,1] = resized_img.width - bbx[:,1]
    return bbx 


def debug_img(img, labels):
    draw = ImageDraw.Draw(img)
    COLOR = (255, 0, 0)
    for label in labels:
        xyxy = [label[1]-label[3]/2, label[2]-label[4]/2,label[1]+label[3]/2, label[2]+label[4]/2]
        draw.rectangle(xyxy, outline=COLOR)
        draw.rectangle([xyxy[0], xyxy[1], xyxy[0]+len(coco_label[int(label[0])])*7, \
                xyxy[1]+15], fill=COLOR)
        draw.text([xyxy[0]+2, xyxy[1]+2], coco_label[int(label[0])])
    img.save('bbox_test.png')
   

def test():
    import torchvision

    transform = transforms.Compose([transforms.ToTensor(), \
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    trainlist = '/media/NAS/dataset/PASCALVOC/train.txt'
    dataset = VOCDataset(trainlist, shape=(600,600), shuffle=True, transform=transform, \
            train=True,batch_size=16,num_workers=0)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, \
            num_workers=0, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in trainloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')

#test()

