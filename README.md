# retinanet

1. fixed code (softmax, sigmoid, ...) - jkkim
2. free size code (new)
3. write_txt, draw_box code (new) - jkkim
4. caffe2 train - yecheol(Fri.)
5. Pascal VOC. COCO mAP code - jhkoh
6. 

| Data | Batch |  Lr   | Ignore |    Loss    |   Size    | Epoch | mAP  |
| :--: | :---: | :---: | :----: | :--------: | :-------: | :---: | :--: |
| VOC  |   8   | 0.001 |   on   |  softmax   | Fixed 600 |  85   |      |
| VOC  |   8   | 0.001 |  off   |  softmax   | Fixed 600 |  85   |      |
| VOC  |   8   | 0.001 |   on   |  sigmoid   | Fixed 600 |  85   |      |
| VOC  |   8   | 0.001 |  off   |  sigmoid   | Fixed 600 |  85   |      |
| VOC  |   8   | 0.001 |   on   | sigmoid(c) | Fixed 600 |  85   |      |
| VOC  |   8   | 0.001 |  off   | sigmoid(c) | Fixed 600 |  85   |      |
| COCO |       |       |        |            |           |       |      |
|      |       |       |        |            |           |       |      |
|      |       |       |        |            |           |       |      |

