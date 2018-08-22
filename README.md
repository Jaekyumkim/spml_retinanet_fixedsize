# retinanet

1. fixed code (softmax, sigmoid, ...) - done
2. free size code (new) - jkkim (on debugging)
3. write_txt, draw_box code (new) - done
4. caffe2 train - yecheol(Fri.)
5. Pascal VOC - done. COCO mAP code - jhkoh

| Data | Batch |  Lr   | Ignore |    Loss    |      Size       | Epoch | Basenet | mAP   |
| :--: | :---: | :---: | :----: | :--------: | :-------------: | :---: | :-----: | ----- |
| VOC  |   8   | 0.001 |   on   |  softmax   |    Fixed 600    |  85   |  FPN50  | yckim |
| VOC  |   8   | 0.001 |  off   |  softmax   |    Fixed 600    |  85   |  FPN50  |       |
| VOC  |   8   | 0.001 |   on   |  sigmoid   |    Fixed 600    |  85   |  FPN50  | jkkim |
| VOC  |   8   | 0.001 |  off   |  sigmoid   |    Fixed 600    |  85   |  FPN50  |       |
| VOC  |   8   | 0.001 |   on   | sigmoid(c) |    Fixed 600    |  85   |  FPN50  | jhkoh |
| VOC  |   8   | 0.001 |  off   | sigmoid(c) |    Fixed 600    |  85   |  FPN50  |       |
| VOC  |   8   | 0.001 |   on   |  sigmoid   | Free (600,1000) |  85   |  FPN50  |       |
| VOC  |   8   | 0.001 |   on   |  sigmoid   |    Fixed 600    |  85   | FPN101  |       |
|      |       |       |        |            |                 |       |         |       |

