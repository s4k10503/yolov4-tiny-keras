## YOLOV4-Tiny: Implementation of You Only Look Once-Tiny target detection model in Keras
---

## Table of contents
1. [Warehouse Update Top News](#warehouse update)
2. [Related warehouse Related code](#Related warehouse)
3. [Performance situation Performance](#performance situation)
4. [Required environment Environment](#Required environment)
5. [File Download Download](#File Download)
6. [Training steps How2train](#training steps)
7. [Prediction step How2predict](#prediction step)
8. [Evaluation step How2eval](#Evaluation step)
9. [Reference Material Reference](#Reference)

## Top News
**`2022-04`**:** Support multi-GPU training, add the calculation of the number of targets of each type, and add heatmap. **

**`2022-03`**:** Substantial updates have been made, and the composition of loss has been modified to make the ratio of classification, target, and regression loss appropriate, support step, cos learning rate drop method, support adam, sgd optimizer Select, support adaptive adjustment of learning rate according to batch_size, and add image cropping. **
The original warehouse address in the BiliBili video is: https://github.com/bubbliiiiing/yolov4-tiny-keras/tree/bilibili

**`2021-10`**:** Substantial updates have been made, adding a large number of comments, adding a large number of adjustable parameters, modifying the components of the code, adding functions such as fps, video prediction, and batch prediction. **

## Related warehouses
| model | path |
| :----- | :----- |
YoloV3 | https://github.com/bubbliiiiing/yolo3-keras
Efficientnet-Yolo3 | https://github.com/bubbliiiiing/efficientnet-yolo3-keras
YoloV4 | https://github.com/bubbliiiiing/yolov4-keras
YoloV4-tiny | https://github.com/bubbliiiiing/yolov4-tiny-keras
Mobilenet-Yolov4 | https://github.com/bubbliiiiing/mobilenet-yolov4-keras
YoloV5-V5.0 | https://github.com/bubbliiiiing/yolov5-keras
YoloV5-V6.1 | https://github.com/bubbliiiiing/yolov5-v6.1-keras
YoloX | https://github.com/bubbliiiiing/yolox-keras
YoloV7 | https://github.com/bubbliiiiing/yolov7-keras
Yolov7-tiny | https://github.com/bubbliiiiing/yolov7-tiny-keras

## Performance
| training dataset | weight file name | test dataset | input image size | mAP 0.5:0.95 | mAP 0.5 |
| :-----: | :-----: | :------: | :------: | :------: | :-----: |
| VOC07+12+COCO | [yolov4_tiny_weights_voc.h5](https://github.com/bubbliiiing/yolov4-tiny-keras/releases/download/v1.1/yolov4_tiny_weights_voc.h5) | VOC-Test07 | 416x416 | - | 77.5
| VOC07+12+COCO | [yolov4_tiny_weights_voc_SE.h5](https://github.com/bubbliiiing/yolov4-tiny-keras/releases/download/v1.1/yolov4_tiny_weights_voc_SE.h5) | VOC-Test07 | 416x416 | - | 78.6
| VOC07+12+COCO | [yolov4_tiny_weights_voc_CBAM.h5](https://github.com/bubbliiiing/yolov4-tiny-keras/releases/download/v1.1/yolov4_tiny_weights_voc_CBAM.h5) | VOC-Test07 | 416x416 | - | 78.9
| VOC07+12+COCO | [yolov4_tiny_weights_voc_ECA.h5](https://github.com/bubbliiiing/yolov4-tiny-keras/releases/download/v1.1/yolov4_tiny_weights_voc_ECA.h5) | VOC-Test07 | 416x416 | - | 78.2
| COCO-Train2017 | [yolov4_tiny_weights_coco.h5](https://github.com/bubbliiiing/yolov4-tiny-keras/releases/download/v1.1/yolov4_tiny_weights_coco.h5) | COCO-Val2017 | 416x416 | 21.8 | 41.3

## required environment
tensorflow-gpu==1.13.1
keras==2.1.5

## Download Document
All kinds of weights required for training can be downloaded from Baidu Netdisk.
Link: https://pan.baidu.com/s/1f9VXWsi4fcYEkEO2YPQKIw
Extraction code: i2ut

The download address of the VOC dataset is as follows, which already includes the training set, test set, and verification set (same as the test set), and there is no need to divide it again:
Link: https://pan.baidu.com/s/19Mw2u_df_nBzsC2lg20fQA
Extraction code: j5ge

## Training steps
### a. Training VOC07+12 data set
1. Dataset preparation
**This article uses the VOC format for training. You need to download the VOC07+12 dataset before training, and put it in the root directory after decompression**

2. Dataset Processing
Modify annotation_mode=2 in voc_annotation.py, run voc_annotation.py to generate 2007_train.txt and 2007_val.txt in the root directory.

3. Start network training
The default parameters of train.py are used to train the VOC dataset, and the training can be started by running train.py directly.

4. Training result prediction
Two files are required for training result prediction, namely yolo.py and predict.py. We first need to modify model_path and classes_path in yolo.py, these two parameters must be modified.
**model_path points to the trained weight file in the logs folder.
classes_path points to the txt corresponding to the detection category. **
After completing the modification, you can run predict.py for detection. After running, enter the image path to detect.

### b. Train your own data set
1. Dataset preparation
**This article uses the VOC format for training, you need to make your own data set before training,**
Before training, put the label file in the Annotation under the VOC2007 folder under the VOCdevkit folder.
Before training, put the picture file in JPEGImages under the VOC2007 folder under the VOCdevkit folder.

2. Dataset Processing
After completing the placement of the data set, we need to use voc_annotation.py to obtain 2007_train.txt and 2007_val.txt for training.
Modify the parameters in voc_annotation.py. The first training can only modify the classes_path, which is used to point to the txt corresponding to the detected category.
When training your own data set, you can create a cls_classes.txt by yourself, and write the categories you need to distinguish in it.
The content of the model_data/cls_classes.txt file is:
```python
cat
the dog
...
```
Modify the classes_path in voc_annotation.py to correspond to cls_classes.txt, and run voc_annotation.py.

3. Start network training
** There are many training parameters, all of which are in train.py. You can read the comments carefully after downloading the library. The most important part is still the classes_path in train.py. **
**classes_path is used to point to the txt corresponding to the detection category, which is the same as the txt in voc_annotation.py! Training your own data set must be modified! **
After modifying the classes_path, you can run train.py to start training. After training for multiple epochs, the weights will be generated in the logs folder.

4. Training result prediction
Two files are required for training result prediction, namely yolo.py and predict.py. Modify model_path and classes_path in yolo.py.
**model_path points to the trained weight file in the logs folder.
classes_path points to the txt corresponding to the detection category. **
After completing the modification, you can run predict.py for detection. After running, enter the image path to detect.

## Prediction step
### a. Use pre-trained weights
1. After downloading the library, unzip it, download yolo_weights.pth from Baidu Netdisk, put it into model_data, run predict.py, and enter
```python
img/street.jpg
```
2. Setting in predict.py can perform fps test and video video detection.
### b. Use your own training weights
1. Follow the training steps to train.
2. In the yolo.py file, modify model_path and classes_path in the following parts to correspond to the trained files; **model_path corresponds to the weight file under the logs folder, and classes_path is the class that model_path corresponds to**. 
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/yolov4_tiny_weights_coco.h5',
    "classes_path"      : 'model_data/coco_classes.txt',
    #---------------------------------------------------------------------#
    #   anchors_path代表先验框对应的txt文件，一般不修改。
    #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
    #---------------------------------------------------------------------#
    "anchors_path"      : 'model_data/yolo_anchors.txt',
    "anchors_mask"      : [[3,4,5], [1,2,3]],
    #-------------------------------#
    #   所使用的注意力机制的类型
    #   phi = 0为不使用注意力机制
    #   phi = 1为SE
    #   phi = 2为CBAM
    #   phi = 3为ECA
    #-------------------------------#
    "phi"               : 0,  
    #---------------------------------------------------------------------#
    #   输入图片的大小，必须为32的倍数。
    #---------------------------------------------------------------------#
    "input_shape"       : [416, 416],
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.3,
    "max_boxes"         : 100,
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
}
```
3. Run predict.py, enter  
```python
img/street.jpg
```
4. Setting in predict.py can perform fps test and video video detection. 

## Evaluation Step
### a. Evaluate the test set of VOC07+12
1. This article uses the VOC format for evaluation. VOC07+12 has already divided the test set, so there is no need to use voc_annotation.py to generate txt under the ImageSets folder.
2. Modify model_path and classes_path in yolo.py.**model_path points to the trained weight file in the logs folder. classes_path points to the txt corresponding to the detection category.**  
3. Run get_map.py to get the evaluation result, and the evaluation result will be saved in the map_out folder.

### b. Evaluate your own dataset
1. This article uses the VOC format for evaluation.
2. If the voc_annotation.py file has been run before training, the code will automatically divide the data set into training set, verification set and test set. If you want to modify the proportion of the test set, you can modify the trainval_percent under the voc_annotation.py file. trainval_percent is used to specify the ratio of (training set + validation set) to test set, by default (training set + validation set): test set = 9:1. train_percent is used to specify the ratio of training set to validation set in (training set + validation set), by default training set: validation set = 9:1.
3. After using voc_annotation.py to divide the test set, go to the get_map.py file to modify the classes_path. The classes_path is used to point to the txt corresponding to the detection category. This txt is the same as the txt during training. Evaluating your own dataset has to be modified.
4. Modify model_path and classes_path in yolo.py. **model_path points to the trained weight file in the logs folder. classes_path points to the txt corresponding to the detection category.**  
5. Run get_map.py to get the evaluation result, and the evaluation result will be saved in the map_out folder.

## Reference
https://github.com/qqwweee/keras-yolo3  
https://github.com/eriklindernoren/PyTorch-YOLOv3   
https://github.com/BobLiu20/YOLOv3_PyTorch
