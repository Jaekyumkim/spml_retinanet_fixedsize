# retinanet

1. fixed code (softmax, sigmoid, ...) - done
2. free size code (new) - done
3. write_txt, draw_box code (new) - done
4. caffe2 train - yecheol(Fri.)
5. Pascal VOC - done. COCO mAP code - done.

| Data | Batch |  Lr   | Ignore |    Loss    |      Size       | Epoch | Basenet | mAP   |
| :--: | :---: | :---: | :----: | :--------: | :-------------: | :---: | :-----: | ----- |
| VOC  |   8   | 0.001 |   on   |  softmax   |    Fixed 600    |  86   |  FPN50  | yckim |
| VOC  |   8   | 0.001 |  off   |  softmax   |    Fixed 600    |  86   |  FPN50  |       |
| VOC  |   8   | 0.001 |   on   |  sigmoid   |    Fixed 600    |  86   |  FPN50  | 75.92 |
| VOC  |   8   | 0.001 |  off   |  sigmoid   |    Fixed 600    |  86   |  FPN50  |       |
| VOC  |   8   | 0.001 |   on   | sigmoid(c) |    Fixed 600    |  86   |  FPN50  | 76.04 |
| VOC  |   8   | 0.001 |  off   | sigmoid(c) |    Fixed 600    |  86   |  FPN50  | jhkoh |
| VOC  |   4   | 0.001 |   on   |  sigmoid   | Free (600,1000) |  86   |  FPN50  | jkkim |
| VOC  |   8   | 0.001 |   on   |  sigmoid   |    Fixed 600    |  86   | FPN101  |       |
| COCO |   4   | 0.001 |   on   |  sigmoid   |    Fixed 600    |  86   |  FPN50  |       |
| COCO |   4   | 0.001 |   on   |  softmax   |    Fixed 600    |  86   |  FPN50  |       |
| COCO |   4   | 0.001 |   on   |  sigmoid   | Free (600,1000) |  86   |  FPN50  |       |

