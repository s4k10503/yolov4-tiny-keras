import os
import random
import xml.etree.ElementTree as ET
import numpy as np
from utils.utils import get_classes

class VOCPreprocessor:
    def __init__(self, annotation_mode=2, classes_path='model_data/voc_classes.txt', trainval_percent=0.9, train_percent=0.9, VOCdevkit_path='VOCdevkit'):
        # annotation_mode determines what processing is done at runtime
        # 0: all label processing is done
        # 1: Generate txt file in VOCdevkit/VOC2007/ImageSets
        # 2: 2007_train.txt, 2007_val.txt are generated
        self.annotation_mode = annotation_mode

        # classes_path is the path to the text file containing the class names
        self.classes_path = classes_path

        # trainval_percent determines the ratio of (train+validation) set to test set
        # train_percent determines the ratio of training and validation sets in the (train+validation) set
        self.trainval_percent = trainval_percent
        self.train_percent = train_percent

        # VOCdevkit_path is the path to the folder where the VOC dataset is stored
        self.VOCdevkit_path = VOCdevkit_path
        self.VOCdevkit_sets = [('2007', 'train'), ('2007', 'val')]
        self.classes, _ = get_classes(classes_path)

        self.photo_nums = np.zeros(len(self.VOCdevkit_sets))
        self.nums = np.zeros(len(self.classes))

    def convert_annotation(self, year, image_id, list_file):
        # convert_annotation converts the annotation corresponding to the given image and writes it to the list file
        # Also statistics the goals for each class
        in_file = open(os.path.join(self.VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = 0 
            if obj.find('difficult')!=None:
                difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in self.classes or int(difficult)==1:
                continue
            cls_id = self.classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

            self.nums[self.classes.index(cls)] = self.nums[self.classes.index(cls)] + 1
        
    def generate_txt_in_image_sets(self):
        # generate txt in ImageSets
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(self.VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(self.VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = [xml for xml in temp_xml if xml.endswith(".xml")]

        num     = len(total_xml)  
        list_   = range(num)
        tv      = int(num * self.trainval_percent)  
        tr      = int(tv * self.train_percent)  
        trainval = random.sample(list_, tv)  
        train   = random.sample(trainval, tr)  

        print("train and val size", tv)
        print("train size", tr)
        ftrainval   = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath, 'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath, 'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath, 'val.txt'), 'w')  

        for i in list_:  
            name = total_xml[i][:-4] + '\n'
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  

        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    def generate_train_val_txt(self):
        # Generate 2007_train.txt and 2007_val.txt for training
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        type_index = 0
        for year, image_set in self.VOCdevkit_sets:
            image_ids = open(os.path.join(self.VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.jpg'%(os.path.abspath(self.VOCdevkit_path), year, image_id))

                self.convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            self.photo_nums[type_index] = len(image_ids)
            type_index += 1
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")

    def print_table(self, List1, List2):
        for i in range(len(List1[0])):
            print("|", end=' ')
            for j in range(len(List1)):
                print(List1[j][i].rjust(int(List2[j])), end=' ')
                print("|", end=' ')
            print()

    def process(self):
        # Perform appropriate processing according to annotation_mode
        if self.annotation_mode in [0, 1]:
            self.generate_txt_in_image_sets()

        if self.annotation_mode in [0, 2]:
            self.generate_train_val_txt()

        str_nums = [str(int(x)) for x in self.nums]
        tableData = [
            self.classes, str_nums
        ]
        colWidths = [0] * len(tableData)

        for i in range(len(tableData)):
            for j in range(len(tableData[i])):
                if len(tableData[i][j]) > colWidths[i]:
                    colWidths[i] = len(tableData[i][j])
        self.print_table(tableData, colWidths)

        if self.photo_nums[0] <= 500:
            print("Because the training dataset size is less than 500 and the amount of data is small, set more training epochs to ensure sufficient gradient descent times (steps).")

        if np.sum(self.nums) == 0:
            print("Couldn't get target in dataset. Please change classes_path to your own dataset and make sure the label names are correct, otherwise there will be no training effect!")


if __name__ == "__main__":
    preprocessor = VOCPreprocessor(annotation_mode=2, classes_path='model_data/voc_classes.txt', trainval_percent=0.9, train_percent=0.9, VOCdevkit_path='VOCdevkit')
    preprocessor.process()
