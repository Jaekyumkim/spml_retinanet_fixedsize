import os
import pdb
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F

import argparse

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, \
                bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, \
                stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, \
                kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, \
                            kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        # input channel = 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,   64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block,  128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block,  256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block,  512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(256,  256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        # p5, C=256
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down Layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)        
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)        
        self.toplayer3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        # Top-down
        m5 = self.latlayer1(c5)
        p5 = self.toplayer1(m5)
        m4 = self._upsample_add(m5, self.latlayer2(c4))
        p4 = self.toplayer2(m4)
        m3 = self._upsample_add(m4, self.latlayer3(c3))
        p3 = self.toplayer3(m3)
        # paper page 4
        return p3, p4, p5, p6, p7

class PIXOR(nn.Module):
    def __init__(self, block, num_blocks):
        super(PIXOR, self).__init__()
        self.in_planes = 36

        # input channel 36
        self.conv1 = nn.Conv2d(36, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 24, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(block, 48, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 96, num_blocks[3], stride=2)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(384, 196, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.ConvTranspose2d(196, 128, kernel_size=3, \
                stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.toplayer2 = nn.ConvTranspose2d(128, 96, kernel_size=3, \
                stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(96)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Squential(*layers)

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = F.relu(self.bn2(self.conv2(c1)))
        c3 = self.layer1(c2)
        c4 = self.layer2(c3)
        c5 = self.layer3(c4)
        c6 = self.layer4(c5)

        # Top-down
        p1 = self.latlayer1(c6)
        p1 = self.latlayer2(c5) + self.bn3(self.toplayer1(p1))
        p2 = self.latlayer3(c4) + self.bn4(self.toplayer2(p1))

        return p2

class RetinaNet(nn.Module):
    # p5 A=9
    num_anchors = 9
    def __init__(self, num_classes=21, net_type='FPN50'):
        global net_func
        super(RetinaNet, self).__init__()
        self.fpn = net_func[net_type]()
        self.num_classes = num_classes
        self.loc_head = self._make_head(4*self.num_anchors)
        self.cls_head = self._make_head(num_classes*self.num_anchors)
        #self.freeze_bn()

    def _make_head(self, out_channel):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_channel, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        subnets = self.fpn(x)
        loc_preds, cls_preds = [], []
        for sub in subnets:
            loc_pred = self.loc_head(sub)
            cls_pred = self.cls_head(sub)
            # [N, KA, H, W] -> [N, H, W, KA] -> [N, H*W, KA]
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(batch_size,-1,4)
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(batch_size,-1,self.num_classes)
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        return loc_preds, cls_preds

def FPN50():
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    return FPN(Bottleneck, [2,4,23,3])

def PIXOR_net():
    return PIXOR(Bottleneck, [3,4,6,3])

# INITIALIZE
net_func = {
        'FPN50': FPN50,
        'FPN101': FPN101,
        'PIXOR' : PIXOR_net,
        }

def build_net(net_type, loss_fn, data_name):
    # Build RetinaNet module
    save_path = './init_weight/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print('Loading resnet into FPN..')
    # pre-trained on ImageNet
    res = models.resnet50(True)
    fpn = net_func[net_type]()
    fpn_wgt = fpn.state_dict()
    res_wgt = res.state_dict()
    if data_name == 'VOC' and loss_fn == 'sigmoid':
        num_classes = 20
    elif data_name == 'VOC' and loss_fn == 'softmax':
        num_classes = 21
    elif data_name == 'COCO' and loss_fn == 'sigmoid':
        num_classes = 80
    if data_name == 'COCO' and loss_fn == 'softmax':
        num_classes = 81

    # load resnet weight
    for k in res_wgt.keys():
        if not k.startswith('fc'):
            fpn_wgt[k] = res_wgt[k]

    if net_type is 'PIXOR':
        pass
    else: 
        print('Construct RetinaNet..')
        net = RetinaNet(num_classes=num_classes, net_type=net_type)
        for mod in net.modules():
            # Initialization p5
            if isinstance(mod, nn.Conv2d):
                init.normal_(mod.weight, mean=0, std=0.01)
                if mod.bias is not None:
                    init.constant_(mod.bias, 0)

            # Because of torch initialize BN w~U[0,1], b=0 value initialize w=1 again
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
        
        # Initizliation p5
        pi = 0.01
        init.constant_(net.cls_head[-1].bias, -math.log((1-pi)/pi))
        if loss_fn == 'softmax':
            net.cls_head[-1].bias.data[[0,21,42,63,84,105,126,147,168]] = np.log(20*(1-pi)/pi)
        net.fpn.load_state_dict(fpn_wgt)
        torch.save(net.state_dict(), save_path+'net_'+args.data+'_'+args.loss_fn+'.pt')
        print('Success')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RetinaNet network build parser')
    parser.add_argument('--network', '-n', default='FPN50', choices=['FPN50', 'FPN101', 'PIXOR'], \
            type=str, help='FPN50 || FPN101 || PIXOR')
    parser.add_argument('--loss_fn', '-loss', default='sigmoid', choices=['sigmoid', 'softmax'], \
            type=str)
    parser.add_argument('--data', '-data', default='VOC', choices=['VOC', 'COCO'], type=str)
    args = parser.parse_args()

    build_net(args.network, args.loss_fn, args.data)
